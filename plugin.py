import asyncio
import json
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import uuid

from memory_layers import (
    MemoryEntry, MemoryConfig, MemoryLayer,
    ShallowMemoryLayer, WorkingMemoryLayer, 
    DeepMemoryLayer, MetaMemoryLayer
)
from ranker import SpatioTemporalRanker
from budget import BudgetController


def make_openai_compatible_api_func(
    api_key: str,
    base_url: str,
    model: str,
    timeout_s: int = 60,
):
    base_url = (base_url or "").rstrip("/")
    if not base_url:
        raise ValueError("base_url is required")
    if not api_key:
        raise ValueError("api_key is required")
    if not model:
        raise ValueError("model is required")
    
    async def _call(prompt: str) -> str:
        import aiohttp
        
        url = f"{base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        }
        
        timeout = aiohttp.ClientTimeout(total=timeout_s)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, headers=headers, json=payload) as resp:
                data = await resp.json(content_type=None)
                if resp.status >= 400:
                    err = data.get("error", {})
                    msg = err.get("message") or str(data)
                    raise RuntimeError(f"LLM API error ({resp.status}): {msg}")
                choices = data.get("choices") or []
                if not choices:
                    raise RuntimeError(f"LLM API returned no choices: {data}")
                message = choices[0].get("message") or {}
                content = message.get("content")
                if not content:
                    raise RuntimeError(f"LLM API returned empty content: {data}")
                return str(content).strip()
    
    return _call


class SpatioTemporalMemoryPlugin:
    """OpenClaw插件：分层时空记忆系统"""
    
    def __init__(
        self,
        model_name: str = "openclaw-medium",
        memory_config: Optional[MemoryConfig] = None,
        enable_logging: bool = True
    ):
        """
        初始化插件
        
        Args:
            model_name: 使用的模型名称
            memory_config: 记忆系统配置
            enable_logging: 是否启用日志记录
        """
        self.model_name = model_name
        self.config = memory_config or MemoryConfig()
        self.enable_logging = enable_logging
        
        # 初始化各层记忆
        self.layers: Dict[str, MemoryLayer] = {
            "shallow": ShallowMemoryLayer(self.config),
            "working": WorkingMemoryLayer(self.config),
            "deep": DeepMemoryLayer(self.config),
            "meta": MetaMemoryLayer(self.config)
        }
        
        # 初始化排序器
        self.ranker = SpatioTemporalRanker(self.config)
        
        # 初始化预算控制器
        self.budget_controller = BudgetController(model_name)
        
        # 性能统计
        self.performance_stats = {
            "total_queries": 0,
            "total_latency_ms": 0,
            "average_latency_ms": 0,
            "total_tokens_saved": 0,
            "total_cost_saved": 0.0,
            "cache_hit_rate": 0.0,
            "memory_compression_ratio": 0.0
        }
        
        # 日志记录
        self.query_log = []
        self.max_log_entries = 1000
        
        if self.enable_logging:
            print(f"[SpatioTemporalMemoryPlugin] Initialized with model: {model_name}")
    
    def _log(self, message: str, level: str = "INFO"):
        """记录日志"""
        if self.enable_logging:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] [{level}] {message}")
    
    def _estimate_token_count(self, text: str) -> int:
        """估算文本token数量（简化版本）"""
        # 粗略估算：1个token ≈ 0.75个英文单词或1个中文字符
        words = text.split()
        english_tokens = len(words) * 0.75
        
        # 计算中文字符
        chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
        
        return int(english_tokens + chinese_chars)
    
    def _create_memory_entry(self, query: str, response: str, layer: str) -> MemoryEntry:
        """创建记忆条目"""
        entry_id = str(uuid.uuid4())
        timestamp = datetime.now()
        token_count = self._estimate_token_count(query) + self._estimate_token_count(response)
        
        return MemoryEntry(
            id=entry_id,
            query=query,
            response=response,
            timestamp=timestamp,
            layer=layer,
            token_count=token_count
        )
    
    def _build_context_from_memories(self, memories: List[MemoryEntry]) -> str:
        """从记忆条目构建上下文"""
        if not memories:
            return ""
        
        context_parts = []
        for memory in memories:
            context_parts.append(f"Previous conversation:\nQ: {memory.query}\nA: {memory.response}")
        
        return "\n\n".join(context_parts)
    
    def _log_query(self, query: str, context: str, response: str, 
                  memory_tokens: int, response_tokens: int, latency_ms: float):
        """记录查询日志"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "context_length": len(context),
            "response": response,
            "memory_tokens": memory_tokens,
            "response_tokens": response_tokens,
            "total_tokens": memory_tokens + response_tokens,
            "latency_ms": latency_ms
        }
        
        self.query_log.append(log_entry)
        
        # 保持日志大小
        if len(self.query_log) > self.max_log_entries:
            self.query_log.pop(0)
    
    async def retrieve_relevant_memories(self, query: str) -> Tuple[str, int, List[MemoryEntry]]:
        """
        检索相关记忆
        
        Args:
            query: 用户查询
            
        Returns:
            (上下文字符串, 使用的token数, 相关记忆列表)
        """
        start_time = time.time()
        
        # 估算响应长度
        response_estimate = self.budget_controller.estimate_response_length(query)
        
        # 计算记忆检索预算
        memory_budget = self.budget_controller.calculate_memory_budget(query, response_estimate)
        
        # 从各层检索记忆
        all_memories = {}
        for layer_name, layer in self.layers.items():
            if layer_name != "meta":  # 元记忆不直接参与检索
                memories = layer.retrieve(query, memory_budget // 4)  # 每层分配1/4预算
                all_memories[layer_name] = memories
        
        # 使用时空排序器选择最优记忆组合
        current_time = datetime.now()
        selected_memories = self.ranker.select_memories_within_budget(
            query, all_memories, memory_budget, current_time
        )
        
        # 构建上下文
        context = self._build_context_from_memories(selected_memories)
        memory_tokens = self._estimate_token_count(context)
        
        # 记录到元记忆
        meta_entry = self._create_memory_entry(query, f"Retrieved {len(selected_memories)} memories", "meta")
        self.layers["meta"].add(meta_entry)
        
        latency_ms = (time.time() - start_time) * 1000
        self._log(f"Memory retrieval completed in {latency_ms:.2f}ms, found {len(selected_memories)} memories")
        
        return context, memory_tokens, selected_memories
    
    async def process_query(self, query: str, openclaw_api_func) -> Dict[str, Any]:
        """
        处理用户查询
        
        Args:
            query: 用户查询
            openclaw_api_func: OpenClaw API调用函数
            
        Returns:
            处理结果
        """
        start_time = time.time()
        
        try:
            self._log(f"Processing query: {query[:100]}...")
            
            # 1. 检索相关记忆
            context, memory_tokens, relevant_memories = await self.retrieve_relevant_memories(query)
            
            # 2. 构建完整的prompt
            if context:
                full_prompt = f"Context from previous conversations:\n{context}\n\nCurrent query: {query}"
            else:
                full_prompt = query
            
            # 3. 调用OpenClaw API
            self._log(f"Calling OpenClaw API with {self._estimate_token_count(full_prompt)} tokens...")
            
            response = await openclaw_api_func(full_prompt)
            response_tokens = self._estimate_token_count(response)
            
            # 4. 检查预算约束
            total_tokens = memory_tokens + response_tokens
            budget_check = self.budget_controller.check_budget_constraints(memory_tokens, response_tokens)
            
            if not budget_check["within_limits"]:
                self._log(f"Budget exceeded: {total_tokens} > {budget_check['max_tokens']}", "WARNING")
            
            # 5. 更新预算统计
            self.budget_controller.update_usage_stats(total_tokens, memory_tokens)
            
            # 6. 异步更新记忆层
            await self._update_memories_async(query, response, relevant_memories)
            
            # 7. 记录性能统计
            latency_ms = (time.time() - start_time) * 1000
            self._update_performance_stats(latency_ms, memory_tokens, len(relevant_memories))
            
            # 8. 记录查询日志
            self._log_query(query, context, response, memory_tokens, response_tokens, latency_ms)
            
            result = {
                "query": query,
                "response": response,
                "context": context,
                "memory_tokens": memory_tokens,
                "response_tokens": response_tokens,
                "total_tokens": total_tokens,
                "latency_ms": latency_ms,
                "relevant_memories_count": len(relevant_memories),
                "budget_check": budget_check,
                "status": "success"
            }
            
            self._log(f"Query processed successfully in {latency_ms:.2f}ms")
            return result
            
        except Exception as e:
            self._log(f"Error processing query: {str(e)}", "ERROR")
            return {
                "query": query,
                "response": f"Error: {str(e)}",
                "status": "error",
                "error": str(e),
                "latency_ms": (time.time() - start_time) * 1000
            }
    
    async def _update_memories_async(self, query: str, response: str, relevant_memories: List[MemoryEntry]):
        """异步更新记忆层"""
        try:
            # 创建新的记忆条目
            shallow_entry = self._create_memory_entry(query, response, "shallow")
            working_entry = self._create_memory_entry(query, response, "working")
            deep_entry = self._create_memory_entry(query, response, "deep")
            
            # 添加到各层
            self.layers["shallow"].add(shallow_entry)
            self.layers["working"].add(working_entry)
            self.layers["deep"].add(deep_entry)
            
            # 记录层级转换
            for memory in relevant_memories:
                self.layers["meta"].record_layer_transition(memory.layer, "shallow")
            
            self._log(f"Memories updated across all layers")
            
        except Exception as e:
            self._log(f"Error updating memories: {str(e)}", "WARNING")
    
    def _update_performance_stats(self, latency_ms: float, memory_tokens: int, memories_count: int):
        """更新性能统计"""
        self.performance_stats["total_queries"] += 1
        self.performance_stats["total_latency_ms"] += latency_ms
        self.performance_stats["average_latency_ms"] = (
            self.performance_stats["total_latency_ms"] / self.performance_stats["total_queries"]
        )
        
        # 估算节省的token（假设没有记忆系统需要更多token）
        estimated_original_tokens = memory_tokens * 1.5  # 假设会增加50%
        tokens_saved = max(0, estimated_original_tokens - memory_tokens)
        self.performance_stats["total_tokens_saved"] += tokens_saved
        
        # 计算节省的成本
        cost_saved = self.budget_controller.get_cost_estimate(tokens_saved)
        self.performance_stats["total_cost_saved"] += cost_saved
    
    def cleanup_expired_memories(self):
        """清理过期记忆"""
        current_time = datetime.now()
        total_removed = 0
        
        for layer_name, layer in self.layers.items():
            if layer_name != "meta":  # 元记忆有特殊清理逻辑
                removed = layer.decay(current_time)
                total_removed += removed
                if removed > 0:
                    self._log(f"Removed {removed} expired memories from {layer_name} layer")
        
        return total_removed
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """获取记忆系统统计"""
        stats = {}
        total_entries = 0
        total_tokens = 0
        
        for layer_name, layer in self.layers.items():
            layer_stats = layer.get_stats()
            stats[layer_name] = layer_stats
            total_entries += layer_stats.get("entries", 0)
            total_tokens += layer_stats.get("total_tokens", 0)
        
        stats["total"] = {
            "entries": total_entries,
            "tokens": total_tokens,
            "layers": len(self.layers)
        }
        
        return stats
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        return {
            "performance": self.performance_stats,
            "budget": self.budget_controller.get_stats(),
            "ranker": self.ranker.get_stats(),
            "recent_queries": self.query_log[-10:] if self.query_log else []
        }
    
    def export_configuration(self) -> Dict[str, Any]:
        """导出配置"""
        return {
            "model_name": self.model_name,
            "memory_config": {
                "shallow_ttl": self.config.shallow_ttl,
                "working_ttl": self.config.working_ttl,
                "deep_persist_path": self.config.deep_persist_path,
                "max_shallow_entries": self.config.max_shallow_entries,
                "max_working_entries": self.config.max_working_entries,
                "lambda_decay": self.config.lambda_decay,
                "alpha_similarity": self.config.alpha_similarity,
                "beta_time": self.config.beta_time,
                "gamma_layer": self.config.gamma_layer,
                "token_budget_ratio": self.config.token_budget_ratio
            },
            "budget_config": self.budget_controller.export_config(),
            "performance_stats": self.performance_stats
        }
    
    def reset_statistics(self):
        """重置统计信息"""
        self.performance_stats = {
            "total_queries": 0,
            "total_latency_ms": 0,
            "average_latency_ms": 0,
            "total_tokens_saved": 0,
            "total_cost_saved": 0.0,
            "cache_hit_rate": 0.0,
            "memory_compression_ratio": 0.0
        }
        self.query_log.clear()
        self.budget_controller.usage_stats = {
            "total_queries": 0,
            "total_tokens_used": 0,
            "total_cost": 0.0,
            "average_tokens_per_query": 0,
            "budget_exceeded_count": 0
        }
        self._log("Statistics reset")


# 全局插件实例
_plugin_instance = None


def get_plugin(
    model_name: str = "openclaw-medium",
    memory_config: Optional[MemoryConfig] = None,
    enable_logging: bool = True
) -> SpatioTemporalMemoryPlugin:
    """获取插件实例（单例模式）"""
    global _plugin_instance
    
    if _plugin_instance is None:
        _plugin_instance = SpatioTemporalMemoryPlugin(
            model_name=model_name,
            memory_config=memory_config,
            enable_logging=enable_logging
        )
    
    return _plugin_instance


async def openclaw_with_memory(
    query: str,
    openclaw_api_func,
    model_name: str = "openclaw-medium",
    enable_logging: bool = True
) -> Dict[str, Any]:
    """
    主要的OpenClaw插件接口函数
    
    Args:
        query: 用户查询
        openclaw_api_func: OpenClaw API调用函数
        model_name: 模型名称
        enable_logging: 是否启用日志
        
    Returns:
        处理结果
    """
    plugin = get_plugin(model_name, enable_logging=enable_logging)
    return await plugin.process_query(query, openclaw_api_func)


def cleanup_plugin():
    """清理插件资源"""
    global _plugin_instance
    if _plugin_instance:
        _plugin_instance.cleanup_expired_memories()
        _plugin_instance = None
