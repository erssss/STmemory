from typing import Dict, Optional, Any
from dataclasses import dataclass
import json


@dataclass
class ModelConfig:
    """模型配置信息"""
    name: str
    max_tokens: int
    context_window: int
    cost_per_1k_tokens: float
    optimal_budget_ratio: float  # 建议使用比例


class BudgetController:
    """Token预算控制器"""
    
    # 预定义模型配置
    MODEL_CONFIGS = {
        "openclaw-small": ModelConfig(
            name="openclaw-small",
            max_tokens=4096,
            context_window=4096,
            cost_per_1k_tokens=0.001,
            optimal_budget_ratio=0.8
        ),
        "openclaw-medium": ModelConfig(
            name="openclaw-medium",
            max_tokens=8192,
            context_window=8192,
            cost_per_1k_tokens=0.002,
            optimal_budget_ratio=0.75
        ),
        "openclaw-large": ModelConfig(
            name="openclaw-large",
            max_tokens=16384,
            context_window=16384,
            cost_per_1k_tokens=0.004,
            optimal_budget_ratio=0.7
        ),
        "gpt-3.5-turbo": ModelConfig(
            name="gpt-3.5-turbo",
            max_tokens=4096,
            context_window=4096,
            cost_per_1k_tokens=0.0015,
            optimal_budget_ratio=0.8
        ),
        "gpt-4": ModelConfig(
            name="gpt-4",
            max_tokens=8192,
            context_window=8192,
            cost_per_1k_tokens=0.03,
            optimal_budget_ratio=0.75
        ),
        "claude-3-haiku": ModelConfig(
            name="claude-3-haiku",
            max_tokens=4096,
            context_window=4096,
            cost_per_1k_tokens=0.001,
            optimal_budget_ratio=0.8
        ),
        "claude-3-sonnet": ModelConfig(
            name="claude-3-sonnet",
            max_tokens=8192,
            context_window=8192,
            cost_per_1k_tokens=0.003,
            optimal_budget_ratio=0.75
        )
    }
    
    def __init__(self, model_name: str = "openclaw-medium", system_prompt_ratio: float = 0.1):
        """
        初始化预算控制器
        
        Args:
            model_name: 模型名称
            system_prompt_ratio: 系统提示词占用的token比例
        """
        self.model_name = model_name
        self.system_prompt_ratio = system_prompt_ratio
        self.model_config = self._get_model_config(model_name)
        self.usage_stats = {
            "total_queries": 0,
            "total_tokens_used": 0,
            "total_cost": 0.0,
            "average_tokens_per_query": 0,
            "budget_exceeded_count": 0
        }
    
    def _get_model_config(self, model_name: str) -> ModelConfig:
        """获取模型配置"""
        if model_name in self.MODEL_CONFIGS:
            return self.MODEL_CONFIGS[model_name]
        
        # 默认配置
        return ModelConfig(
            name=model_name,
            max_tokens=4096,
            context_window=4096,
            cost_per_1k_tokens=0.001,
            optimal_budget_ratio=0.8
        )
    
    def calculate_memory_budget(self, query: str, response_length_estimate: Optional[int] = None) -> int:
        """
        计算记忆检索的token预算
        
        Args:
            query: 用户查询
            response_length_estimate: 预估响应长度（token数）
            
        Returns:
            记忆检索的token预算
        """
        # 基础预算：模型最大token数的配置比例
        base_budget = int(self.model_config.max_tokens * self.model_config.optimal_budget_ratio)
        
        # 为系统提示词预留空间
        system_prompt_tokens = int(self.model_config.max_tokens * self.system_prompt_ratio)
        
        # 为响应预留空间
        if response_length_estimate:
            response_buffer = int(response_length_estimate * 1.2)  # 20%缓冲
        else:
            # 默认响应长度：基础预算的30%
            response_buffer = int(base_budget * 0.3)
        
        # 计算最终预算
        memory_budget = base_budget - system_prompt_tokens - response_buffer
        
        # 确保预算至少为100 tokens
        return max(100, memory_budget)
    
    def calculate_total_budget(self, include_memory: bool = True) -> int:
        """
        计算总token预算
        
        Args:
            include_memory: 是否包含记忆检索预算
            
        Returns:
            总token预算
        """
        if include_memory:
            return int(self.model_config.max_tokens * self.model_config.optimal_budget_ratio)
        else:
            # 仅用于响应生成的预算
            return int(self.model_config.max_tokens * (self.model_config.optimal_budget_ratio - 0.1))
    
    def estimate_response_length(self, query: str) -> int:
        """
        预估响应长度
        
        Args:
            query: 用户查询
            
        Returns:
            预估响应长度（token数）
        """
        # 简单的启发式方法
        query_length = len(query.split())
        
        # 基于查询类型预估
        if any(word in query.lower() for word in ["list", "enumerate", "examples"]):
            # 列表类查询通常需要更长响应
            estimated_length = min(query_length * 3, 500)
        elif any(word in query.lower() for word in ["explain", "describe", "how", "why"]):
            # 解释类查询
            estimated_length = min(query_length * 2, 300)
        elif any(word in query.lower() for word in ["what", "when", "where", "who"]):
            # 事实类查询
            estimated_length = min(query_length * 1.5, 200)
        else:
            # 默认情况
            estimated_length = min(query_length * 2, 250)
        
        return estimated_length
    
    def update_usage_stats(self, tokens_used: int, memory_tokens: int = 0):
        """
        更新使用统计
        
        Args:
            tokens_used: 总使用token数
            memory_tokens: 记忆检索使用的token数
        """
        self.usage_stats["total_queries"] += 1
        self.usage_stats["total_tokens_used"] += tokens_used
        self.usage_stats["total_cost"] += (tokens_used / 1000) * self.model_config.cost_per_1k_tokens
        
        # 更新平均使用量
        self.usage_stats["average_tokens_per_query"] = (
            self.usage_stats["total_tokens_used"] / self.usage_stats["total_queries"]
        )
    
    def check_budget_constraints(self, memory_tokens: int, response_tokens: int) -> Dict[str, Any]:
        """
        检查预算约束
        
        Args:
            memory_tokens: 记忆检索使用的token数
            response_tokens: 响应生成的token数
            
        Returns:
            约束检查结果
        """
        total_tokens = memory_tokens + response_tokens
        max_tokens = self.model_config.max_tokens
        
        constraints = {
            "within_limits": total_tokens <= max_tokens,
            "total_tokens": total_tokens,
            "max_tokens": max_tokens,
            "memory_tokens": memory_tokens,
            "response_tokens": response_tokens,
            "memory_ratio": memory_tokens / total_tokens if total_tokens > 0 else 0,
            "response_ratio": response_tokens / total_tokens if total_tokens > 0 else 0
        }
        
        if not constraints["within_limits"]:
            self.usage_stats["budget_exceeded_count"] += 1
        
        return constraints
    
    def optimize_budget_allocation(self, memory_tokens: int, response_tokens: int) -> Dict[str, int]:
        """
        优化预算分配
        
        Args:
            memory_tokens: 当前记忆检索使用的token数
            response_tokens: 当前响应生成的token数
            
        Returns:
            优化后的预算分配
        """
        total_tokens = memory_tokens + response_tokens
        max_tokens = self.model_config.max_tokens
        
        if total_tokens <= max_tokens:
            # 当前分配合理
            return {
                "memory_tokens": memory_tokens,
                "response_tokens": response_tokens,
                "optimized": False
            }
        
        # 需要优化，优先保证响应生成
        excess_tokens = total_tokens - max_tokens
        
        # 首先减少记忆检索预算
        new_memory_tokens = max(50, memory_tokens - excess_tokens)
        remaining_excess = excess_tokens - (memory_tokens - new_memory_tokens)
        
        # 如果还不够，减少响应预算
        new_response_tokens = response_tokens - remaining_excess
        
        # 确保响应预算至少为50 tokens
        if new_response_tokens < 50:
            new_response_tokens = 50
            new_memory_tokens = max_tokens - new_response_tokens
        
        return {
            "memory_tokens": new_memory_tokens,
            "response_tokens": new_response_tokens,
            "optimized": True
        }
    
    def get_cost_estimate(self, tokens: int) -> float:
        """
        估算成本
        
        Args:
            tokens: token数量
            
        Returns:
            预估成本（美元）
        """
        return (tokens / 1000) * self.model_config.cost_per_1k_tokens
    
    def get_savings_estimate(self, original_tokens: int, optimized_tokens: int) -> Dict[str, Any]:
        """
        计算节省估算
        
        Args:
            original_tokens: 原始token使用量
            optimized_tokens: 优化后的token使用量
            
        Returns:
            节省统计信息
        """
        tokens_saved = original_tokens - optimized_tokens
        savings_ratio = tokens_saved / original_tokens if original_tokens > 0 else 0
        cost_saved = self.get_cost_estimate(tokens_saved)
        
        return {
            "tokens_saved": tokens_saved,
            "savings_ratio": savings_ratio,
            "cost_saved": cost_saved,
            "percentage_saved": savings_ratio * 100
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """获取预算控制器统计信息"""
        return {
            "model_name": self.model_name,
            "model_config": {
                "max_tokens": self.model_config.max_tokens,
                "context_window": self.model_config.context_window,
                "cost_per_1k_tokens": self.model_config.cost_per_1k_tokens,
                "optimal_budget_ratio": self.model_config.optimal_budget_ratio
            },
            "system_prompt_ratio": self.system_prompt_ratio,
            "usage_stats": self.usage_stats,
            "estimated_monthly_cost": self._estimate_monthly_cost()
        }
    
    def _estimate_monthly_cost(self) -> float:
        """估算月度成本"""
        if self.usage_stats["total_queries"] == 0:
            return 0.0
        
        # 假设每天查询量相同，估算月度成本
        daily_queries = self.usage_stats["total_queries"]
        daily_cost = self.usage_stats["total_cost"]
        
        # 简单估算：按当前使用模式推算30天
        monthly_cost = (daily_cost / daily_queries) * daily_queries * 30
        
        return monthly_cost
    
    def export_config(self) -> Dict[str, Any]:
        """导出配置信息"""
        return {
            "model_name": self.model_name,
            "system_prompt_ratio": self.system_prompt_ratio,
            "model_config": {
                "name": self.model_config.name,
                "max_tokens": self.model_config.max_tokens,
                "context_window": self.model_config.context_window,
                "cost_per_1k_tokens": self.model_config.cost_per_1k_tokens,
                "optimal_budget_ratio": self.model_config.optimal_budget_ratio
            }
        }
    
    def load_config(self, config: Dict[str, Any]):
        """加载配置信息"""
        if "model_name" in config:
            self.model_name = config["model_name"]
            self.model_config = self._get_model_config(self.model_name)
        
        if "system_prompt_ratio" in config:
            self.system_prompt_ratio = config["system_prompt_ratio"]
        
        if "model_config" in config:
            model_config_data = config["model_config"]
            self.model_config = ModelConfig(**model_config_data)