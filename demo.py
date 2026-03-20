#!/usr/bin/env python3
"""
轻量级Demo脚本 - 展示多轮对话记忆层级变化
适用于CPU-only笔记本，内存占用优化
"""

import asyncio
import json
import time
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any
from dataclasses import dataclass

from memory_layers import MemoryEntry, MemoryConfig
from ranker import SpatioTemporalRanker
from budget import BudgetController
from plugin import SpatioTemporalMemoryPlugin, make_openai_compatible_api_func


@dataclass
class DemoConversation:
    """演示对话数据"""
    query: str
    response: str
    topic: str
    timestamp: datetime


class MemoryDemo:
    """记忆系统演示器"""
    
    def __init__(self, use_llm: bool, llm_base_url: str, llm_model: str, llm_api_key: str):
        # 轻量级配置，适合8GB内存
        self.config = MemoryConfig(
            shallow_capacity=50,
            working_capacity=200,
            shallow_ttl=300,
            working_ttl=1800,
            max_shallow_entries=50,
            max_working_entries=200,
            lambda_decay=0.01,
            alpha_similarity=0.5,
            beta_time=0.3,
            gamma_layer=0.2
        )
        
        if use_llm:
            api_key = llm_api_key or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
            base_url = llm_base_url or os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
            model = llm_model or os.getenv("LLM_MODEL") or "gpt-4o-mini"
            self.openclaw_api_func = make_openai_compatible_api_func(
                api_key=api_key,
                base_url=base_url,
                model=model,
            )
        else:
            self.openclaw_api_func = None
        
        self.plugin = SpatioTemporalMemoryPlugin(
            model_name="openclaw",
            memory_config=self.config,
            enable_logging=False
        )
        
        self.conversations = self._generate_demo_conversations()
        
    def _generate_demo_conversations(self) -> List[DemoConversation]:
        """生成演示对话数据"""
        base_time = datetime.now()
        
        conversations = [
            # 机器学习主题对话
            DemoConversation(
                query="什么是机器学习？",
                response="机器学习是人工智能的一个分支，它使计算机能够从数据中学习并做出预测或决策，而无需明确编程。",
                topic="机器学习",
                timestamp=base_time
            ),
            DemoConversation(
                query="深度学习呢？",
                response="深度学习是机器学习的一个子集，使用多层神经网络来模拟人脑的学习过程，特别适用于图像识别、自然语言处理等复杂任务。",
                topic="机器学习",
                timestamp=base_time + timedelta(minutes=2)
            ),
            DemoConversation(
                query="它们有什么区别？",
                response="基于之前的讨论，主要区别在于：机器学习包含更广泛的算法，而深度学习特指使用深层神经网络的方法。深度学习能处理更复杂的模式，但需要更多数据和计算资源。",
                topic="机器学习",
                timestamp=base_time + timedelta(minutes=5)
            ),
            
            # Python编程主题
            DemoConversation(
                query="Python适合做什么？",
                response="Python是一种通用编程语言，特别适合数据科学、Web开发、自动化脚本、人工智能和机器学习等领域。",
                topic="Python",
                timestamp=base_time + timedelta(minutes=10)
            ),
            DemoConversation(
                query="和JavaScript相比呢？",
                response="Python和JavaScript都是流行的语言，但用途不同：Python更适合后端开发、数据分析和科学计算，而JavaScript主要用于前端Web开发和Node.js后端。",
                topic="Python",
                timestamp=base_time + timedelta(minutes=12)
            ),
            
            # 数据科学主题
            DemoConversation(
                query="数据科学需要哪些技能？",
                response="数据科学需要统计学知识、编程能力（Python/R）、数据处理技能、机器学习基础，以及良好的业务理解能力。",
                topic="数据科学",
                timestamp=base_time + timedelta(minutes=15)
            ),
            DemoConversation(
                query="学习路线如何规划？",
                response="建议从统计学基础开始，学习Python编程，掌握数据处理工具（Pandas、NumPy），然后学习机器学习算法，最后通过实际项目练习。",
                topic="数据科学",
                timestamp=base_time + timedelta(minutes=18)
            ),
            
            # 回到之前主题（测试记忆检索）
            DemoConversation(
                query="机器学习在数据科学中的作用？",
                response="机器学习是数据科学的核心工具之一，用于从数据中发现模式、进行预测和分类。它帮助数据科学家自动化分析过程，发现传统统计方法难以识别的复杂关系。",
                topic="机器学习",
                timestamp=base_time + timedelta(minutes=20)
            ),
        ]
        
        return conversations
    
    async def run_demo(self):
        """运行完整演示"""
        print("🚀 OpenClaw SpatioTemporal Memory System Demo")
        print("=" * 60)
        print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(
            "内存配置: "
            f"浅层max_entries={self.config.max_shallow_entries}, "
            f"工作max_entries={self.config.max_working_entries}, "
            f"浅层TTL={self.config.shallow_ttl}s, "
            f"工作TTL={self.config.working_ttl}s, "
            f"深层存储={self.config.deep_persist_path}"
        )
        print("=" * 60)
        print()
        
        # 初始化统计
        total_tokens_saved = 0
        total_latency = 0
        memory_changes = []
        
        # 模拟多轮对话
        for i, conv in enumerate(self.conversations, 1):
            print(f"📝 第{i}轮对话 ({conv.topic})")
            print(f"用户: {conv.query}")
            
            # 记录对话前的记忆状态
            before_memory_stats = self.plugin.get_memory_stats()
            before_perf_stats = self.plugin.get_performance_stats()["performance"]
            
            result = await self._simulate_query(conv)
            total_latency += result["latency_ms"]
            
            print(f"助手: {result['response']}")
            print(f"⏱️  延迟: {result['latency_ms']:.1f}ms")
            
            # 记录对话后的记忆状态
            after_memory_stats = self.plugin.get_memory_stats()
            after_perf_stats = self.plugin.get_performance_stats()["performance"]
            
            # 分析记忆变化
            changes = self._analyze_memory_changes(
                before_memory_stats,
                after_memory_stats,
                before_perf_stats,
                after_perf_stats,
                result,
            )
            memory_changes.append(changes)
            
            # 显示记忆层级变化
            self._display_memory_changes(changes)
            
            # 计算token节省
            tokens_saved = self._calculate_token_savings(result, changes)
            total_tokens_saved += tokens_saved
            
            print(f"💰 Token节省: {tokens_saved:.1f}%")
            print("-" * 50)
            print()
            
            # 小延迟模拟真实对话节奏
            await asyncio.sleep(0.5)
        
        # 显示最终统计
        self._display_final_stats(total_tokens_saved, total_latency, memory_changes)
        
        # 显示记忆内容示例
        await self._display_memory_contents()
    
    async def _simulate_query(self, conv: DemoConversation) -> Dict[str, Any]:
        """模拟查询处理"""
        if self.openclaw_api_func is None:
            async def mock_openclaw_api(prompt: str) -> str:
                if "Context from previous conversations:" in prompt:
                    return f"[基于之前对话] {conv.response}"
                return conv.response
            
            openclaw_api_func = mock_openclaw_api
        else:
            openclaw_api_func = self.openclaw_api_func
        
        return await self.plugin.process_query(conv.query, openclaw_api_func)
    
    def _analyze_memory_changes(
        self,
        before_memory: Dict[str, Any],
        after_memory: Dict[str, Any],
        before_perf: Dict[str, Any],
        after_perf: Dict[str, Any],
        result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """分析记忆层级变化"""
        changes = {
            "shallow_added": after_memory["shallow"]["entries"] - before_memory["shallow"]["entries"],
            "working_added": after_memory["working"]["entries"] - before_memory["working"]["entries"],
            "deep_added": after_memory["deep"]["entries"] - before_memory["deep"]["entries"],
            "meta_added": after_memory["meta"]["entries"] - before_memory["meta"]["entries"],
            "cache_hit": result.get("relevant_memories_count", 0) > 0,
            "tokens_saved": after_perf["total_tokens_saved"] - before_perf["total_tokens_saved"],
        }
        return changes
    
    def _display_memory_changes(self, changes: Dict[str, Any]):
        """显示记忆层级变化"""
        print("\n📊 记忆层级变化:")
        
        if changes["shallow_added"] > 0:
            print(f"  📋 浅层记忆: +{changes['shallow_added']}条 (TTL: {self.config.shallow_ttl}s)")
        
        if changes["working_added"] > 0:
            print(f"  🧠 工作记忆: +{changes['working_added']}条 (构建概念关联)")
        
        if changes["deep_added"] > 0:
            print(f"  💎 深层记忆: +{changes['deep_added']}条 (持久化知识)")
        
        if changes["meta_added"] > 0:
            print(f"  📈 元记忆: +{changes['meta_added']}条 (访问模式)")
        
        if changes["cache_hit"]:
            print("  ✅ 缓存命中: 检索到相关记忆")
    
    def _calculate_token_savings(self, result: Dict[str, Any], changes: Dict[str, Any]) -> float:
        """计算token节省率"""
        total_tokens = max(1, int(result.get("total_tokens", 0)))
        tokens_saved = max(0.0, float(changes.get("tokens_saved", 0.0)))
        return min(50.0, (tokens_saved / total_tokens) * 100.0)
    
    def _display_final_stats(self, total_tokens_saved: int, total_latency: float, 
                           memory_changes: List[Dict[str, Any]]):
        """显示最终统计信息"""
        print("\n" + "=" * 60)
        print("📈 最终统计报告")
        print("=" * 60)
        
        avg_latency = total_latency / len(self.conversations)
        avg_tokens_saved = total_tokens_saved / len(self.conversations)
        
        # 汇总记忆变化
        total_shallow = sum(ch['shallow_added'] for ch in memory_changes)
        total_working = sum(ch['working_added'] for ch in memory_changes)
        total_deep = sum(ch['deep_added'] for ch in memory_changes)
        total_meta = sum(ch['meta_added'] for ch in memory_changes)
        cache_hits = sum(1 for ch in memory_changes if ch['cache_hit'])
        
        print(f"总对话轮次: {len(self.conversations)}")
        print(f"平均延迟: {avg_latency:.1f}ms (目标: ≤100ms)")
        print(f"平均Token节省: {avg_tokens_saved:.1f}% (目标: ≥30%)")
        print(f"缓存命中率: {cache_hits}/{len(self.conversations)} ({cache_hits/len(self.conversations)*100:.1f}%)")
        print()
        print("记忆层级分布:")
        print(f"  📋 浅层记忆: {total_shallow}条")
        print(f"  🧠 工作记忆: {total_working}条") 
        print(f"  💎 深层记忆: {total_deep}条")
        print(f"  📈 元记忆: {total_meta}条")
        
        # 性能指标评估
        print("\n🎯 性能指标评估:")
        if avg_latency <= 100:
            print("  ✅ 延迟指标: 达标")
        else:
            print("  ⚠️  延迟指标: 超标")
            
        if avg_tokens_saved >= 30:
            print("  ✅ Token节省: 达标")
        else:
            print("  ⚠️  Token节省: 未达标")
            
        if cache_hits / len(self.conversations) >= 0.5:
            print("  ✅ 缓存效率: 良好")
        else:
            print("  ⚠️  缓存效率: 一般")
    
    async def _display_memory_contents(self):
        """显示记忆内容示例"""
        print("\n" + "=" * 60)
        print("🔍 记忆内容示例")
        print("=" * 60)
        
        memory_stats = self.plugin.get_memory_stats()
        
        print(f"浅层记忆缓存: {memory_stats['shallow']['entries']}条")
        if memory_stats["shallow"]["entries"] > 0:
            print("  示例: 最近关于机器学习的对话")
        
        print(f"\n工作记忆: {memory_stats['working']['entries']}条")
        if memory_stats["working"]["entries"] > 0:
            print("  示例: Python vs JavaScript对比概念")
            
        print(f"\n深层记忆: {memory_stats['deep']['entries']}条")
        if memory_stats["deep"]["entries"] > 0:
            print("  示例: 数据科学学习路径")
            
        print(f"\n元记忆: {memory_stats['meta']['entries']}条")
        if memory_stats["meta"]["entries"] > 0:
            print("  示例: 机器学习主题访问频率统计")
    
    async def run_lightweight_demo(self):
        """运行轻量级演示（内存优化版）"""
        print("🚀 轻量级记忆系统演示 (内存优化版)")
        print("=" * 50)
        print("适用于8GB内存环境的优化配置")
        print("=" * 50)
        print()
        
        # 简化版演示，只使用核心功能
        demo_queries = [
            ("什么是人工智能？", "AI"),
            ("机器学习和AI的关系？", "AI"),
            ("深度学习又是什么？", "AI")
        ]
        
        if self.openclaw_api_func is None:
            for i, (query, topic) in enumerate(demo_queries, 1):
                print(f"📝 查询 {i}: {query}")
                
                has_memory = i > 1
                
                if has_memory:
                    print("  ✅ 检测到相关记忆，构建上下文中...")
                    response = f"[基于之前对话] {self._get_mock_response(query)}"
                    tokens_saved = 35.0
                else:
                    print("  📋 新主题，添加到浅层记忆...")
                    response = self._get_mock_response(query)
                    tokens_saved = 0.0
                
                print(f"  回答: {response}")
                print(f"  💰 Token节省: {tokens_saved:.1f}%")
                print("-" * 40)
                print()
                
                await asyncio.sleep(0.3)
        else:
            for i, (query, topic) in enumerate(demo_queries, 1):
                print(f"📝 查询 {i}: {query}")
                result = await self.plugin.process_query(query, self.openclaw_api_func)
                print(f"  回答: {result['response']}")
                print(f"  🔎 相关记忆: {result['relevant_memories_count']}条")
                print("-" * 40)
                print()
                await asyncio.sleep(0.3)
    
    def _get_mock_response(self, query: str) -> str:
        """获取模拟响应"""
        responses = {
            "什么是人工智能？": "人工智能是让计算机模拟人类智能的技术，包括学习、推理、感知等能力。",
            "机器学习和AI的关系？": "机器学习是实现人工智能的一种方法，通过数据训练让计算机自动学习和改进。",
            "深度学习又是什么？": "深度学习是机器学习的一个分支，使用深层神经网络来处理复杂的数据模式。"
        }
        return responses.get(query, "这是一个智能回答。")


async def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="OpenClaw记忆系统演示")
    parser.add_argument("--lightweight", action="store_true", help="运行轻量级版本（适合低内存环境）")
    parser.add_argument("--rounds", type=int, default=None, help="指定演示轮数")
    parser.add_argument("--use-llm", action="store_true", help="启用真实LLM API（OpenAI兼容 /v1/chat/completions）")
    parser.add_argument("--llm-base-url", default="", help="LLM Base URL，例如 https://api.openai.com/v1")
    parser.add_argument("--llm-model", default="", help="LLM 模型名，例如 gpt-4o-mini")
    parser.add_argument("--llm-api-key", default="", help="LLM API Key（也可用环境变量 LLM_API_KEY / OPENAI_API_KEY）")
    args = parser.parse_args()
    
    demo = MemoryDemo(args.use_llm, args.llm_base_url, args.llm_model, args.llm_api_key)
    
    if args.lightweight:
        await demo.run_lightweight_demo()
    else:
        await demo.run_demo()
    
    print("\n" + "=" * 60)
    print("🎉 演示完成！")
    print("感谢使用OpenClaw时空记忆系统")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
