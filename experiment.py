"""
实验代码：评估时空记忆系统性能
记录token使用量、延迟、BLEU等指标
"""

import asyncio
import json
import time
import statistics
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np

# 尝试导入BLEU评分相关库
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.tokenize import word_tokenize
    import nltk
    try:
        nltk.data.find("tokenizers/punkt")
        BLEU_AVAILABLE = True
    except LookupError:
        BLEU_AVAILABLE = False
except ImportError:
    BLEU_AVAILABLE = False

from plugin import SpatioTemporalMemoryPlugin, get_plugin, cleanup_plugin, make_openai_compatible_api_func
from memory_layers import MemoryConfig


class PerformanceEvaluator:
    """性能评估器"""
    
    def __init__(
        self,
        model_name: str = "openclaw-medium",
        use_llm: bool = False,
        llm_base_url: str = "",
        llm_model: str = "",
        llm_api_key: str = "",
    ):
        self.model_name = model_name
        self.results = {
            "experiments": [],
            "summary": {},
            "metadata": {
                "model_name": model_name,
                "tested_at": datetime.now().isoformat(),
                "bleu_available": BLEU_AVAILABLE
            }
        }
        
        # Mock API函数
        self.mock_api_responses = {
            "ai": "AI stands for Artificial Intelligence. It's the simulation of human intelligence in machines that are programmed to think and learn like humans.",
            "machine learning": "Machine Learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed.",
            "neural networks": "Neural networks are computing systems inspired by the biological neural networks that constitute animal brains.",
            "weather": "The weather today is quite pleasant with clear skies and mild temperatures around 22°C.",
            "memory": "I can recall our previous conversations using my layered memory system that stores information across different temporal scales.",
            "help": "I'm here to help you with questions about AI, technology, weather, and general topics. Feel free to ask me anything!",
            "default": "I understand your question and I'm happy to provide a helpful response based on my knowledge."
        }
        
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
    
    async def mock_openclaw_api(self, prompt: str) -> str:
        """改进的模拟OpenClaw API"""
        prompt_lower = prompt.lower()
        
        # 根据关键词选择响应
        for keyword, response in self.mock_api_responses.items():
            if keyword in prompt_lower:
                return response
        
        return self.mock_api_responses["default"]
    
    def calculate_bleu_score(self, reference: str, candidate: str) -> float:
        """计算BLEU分数"""
        if not BLEU_AVAILABLE:
            # 模拟BLEU分数
            # 简单的相似度计算
            ref_words = set(reference.lower().split())
            cand_words = set(candidate.lower().split())
            
            if not ref_words or not cand_words:
                return 0.0
            
            overlap = len(ref_words.intersection(cand_words))
            precision = overlap / len(cand_words) if cand_words else 0
            recall = overlap / len(ref_words) if ref_words else 0
            
            # 简单的F1分数作为BLEU的近似
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
                return f1
            else:
                return 0.0
        
        try:
            # 使用NLTK计算真实的BLEU分数
            reference_tokens = word_tokenize(reference.lower())
            candidate_tokens = word_tokenize(candidate.lower())
            
            smoothing = SmoothingFunction().method1
            bleu_score = sentence_bleu(
                [reference_tokens], 
                candidate_tokens,
                smoothing_function=smoothing
            )
            return bleu_score
        except Exception as e:
            print(f"Error calculating BLEU score: {e}")
            return 0.0
    
    async def run_single_experiment(
        self, 
        plugin: SpatioTemporalMemoryPlugin,
        query: str, 
        expected_response: Optional[str] = None
    ) -> Dict[str, Any]:
        """运行单个实验"""
        
        start_time = time.time()
        
        # 记录实验前的记忆状态
        memory_stats_before = plugin.get_memory_stats()
        
        # 处理查询
        openclaw_api_func = self.openclaw_api_func or self.mock_openclaw_api
        result = await plugin.process_query(query, openclaw_api_func)
        
        end_time = time.time()
        
        # 记录实验后的记忆状态
        memory_stats_after = plugin.get_memory_stats()
        
        # 计算BLEU分数（如果有预期响应）
        bleu_score = 0.0
        if expected_response and result["status"] == "success":
            bleu_score = self.calculate_bleu_score(expected_response, result["response"])
        
        # 计算记忆变化
        memory_changes = {
            "shallow_added": memory_stats_after["shallow"]["entries"] - memory_stats_before["shallow"]["entries"],
            "working_added": memory_stats_after["working"]["entries"] - memory_stats_before["working"]["entries"],
            "deep_added": memory_stats_after["deep"]["entries"] - memory_stats_before["deep"]["entries"],
            "meta_added": memory_stats_after["meta"]["entries"] - memory_stats_before["meta"]["entries"]
        }
        
        experiment_result = {
            "query": query,
            "expected_response": expected_response,
            "timestamp": datetime.now().isoformat(),
            "status": result["status"],
            "latency_ms": result["latency_ms"],
            "memory_tokens": result["memory_tokens"],
            "response_tokens": result["response_tokens"],
            "total_tokens": result["total_tokens"],
            "relevant_memories_count": result["relevant_memories_count"],
            "budget_check": result.get("budget_check", {}),
            "bleu_score": bleu_score,
            "memory_changes": memory_changes,
            "actual_response": result["response"] if result["status"] == "success" else None,
            "error": result.get("error", None)
        }
        
        return experiment_result
    
    async def run_baseline_comparison(self, queries: List[str]) -> Dict[str, Any]:
        """运行基线对比实验"""
        
        # 清理插件
        cleanup_plugin()
        
        # 创建带记忆的插件
        plugin_with_memory = SpatioTemporalMemoryPlugin(
            model_name=self.model_name,
            enable_logging=False
        )
        
        # 创建不带记忆的插件（禁用记忆功能）
        config_no_memory = MemoryConfig(
            max_shallow_entries=0,
            max_working_entries=0,
            lambda_decay=0.0,
            alpha_similarity=0.0,
            beta_time=0.0,
            gamma_layer=0.0
        )
        plugin_no_memory = SpatioTemporalMemoryPlugin(
            model_name=self.model_name,
            memory_config=config_no_memory,
            enable_logging=False
        )
        
        results_with_memory = []
        results_without_memory = []
        
        print(f"Running baseline comparison with {len(queries)} queries...")
        
        for i, query in enumerate(queries):
            if i % 10 == 0:
                print(f"Progress: {i}/{len(queries)}")
            
            # 测试带记忆的情况
            result_with = await self.run_single_experiment(plugin_with_memory, query)
            results_with_memory.append(result_with)
            
            # 测试不带记忆的情况
            result_without = await self.run_single_experiment(plugin_no_memory, query)
            results_without_memory.append(result_without)
        
        # 计算对比指标
        comparison = self._calculate_comparison_metrics(
            results_with_memory, 
            results_without_memory
        )
        
        return {
            "with_memory": results_with_memory,
            "without_memory": results_without_memory,
            "comparison": comparison
        }
    
    def _calculate_comparison_metrics(
        self, 
        with_memory: List[Dict], 
        without_memory: List[Dict]
    ) -> Dict[str, Any]:
        """计算对比指标"""
        
        # 提取关键指标
        latencies_with = [r["latency_ms"] for r in with_memory if r["status"] == "success"]
        latencies_without = [r["latency_ms"] for r in without_memory if r["status"] == "success"]
        
        tokens_with = [r["total_tokens"] for r in with_memory if r["status"] == "success"]
        tokens_without = [r["total_tokens"] for r in without_memory if r["status"] == "success"]
        
        bleu_with = [r["bleu_score"] for r in with_memory if r["bleu_score"] > 0]
        bleu_without = [r["bleu_score"] for r in without_memory if r["bleu_score"] > 0]
        
        # 计算平均值
        avg_latency_with = statistics.mean(latencies_with) if latencies_with else 0
        avg_latency_without = statistics.mean(latencies_without) if latencies_without else 0
        
        avg_tokens_with = statistics.mean(tokens_with) if tokens_with else 0
        avg_tokens_without = statistics.mean(tokens_without) if tokens_without else 0
        
        avg_bleu_with = statistics.mean(bleu_with) if bleu_with else 0
        avg_bleu_without = statistics.mean(bleu_without) if bleu_without else 0
        
        # 计算改进百分比
        latency_improvement = ((avg_latency_without - avg_latency_with) / avg_latency_without * 100) if avg_latency_without > 0 else 0
        token_reduction = ((avg_tokens_without - avg_tokens_with) / avg_tokens_without * 100) if avg_tokens_without > 0 else 0
        bleu_improvement = ((avg_bleu_with - avg_bleu_without) / avg_bleu_without * 100) if avg_bleu_without > 0 else 0
        
        return {
            "latency": {
                "with_memory": avg_latency_with,
                "without_memory": avg_latency_without,
                "improvement_percent": latency_improvement
            },
            "tokens": {
                "with_memory": avg_tokens_with,
                "without_memory": avg_tokens_without,
                "reduction_percent": token_reduction
            },
            "bleu": {
                "with_memory": avg_bleu_with,
                "without_memory": avg_bleu_without,
                "improvement_percent": bleu_improvement
            },
            "success_rate": {
                "with_memory": len([r for r in with_memory if r["status"] == "success"]) / len(with_memory) * 100,
                "without_memory": len([r for r in without_memory if r["status"] == "success"]) / len(without_memory) * 100
            }
        }
    
    async def run_memory_layer_analysis(self, queries: List[str]) -> Dict[str, Any]:
        """运行记忆层分析实验"""
        
        cleanup_plugin()
        plugin = SpatioTemporalMemoryPlugin(
            model_name=self.model_name,
            enable_logging=False
        )
        
        layer_usage = {
            "shallow": {"hits": 0, "total_queries": 0},
            "working": {"hits": 0, "total_queries": 0},
            "deep": {"hits": 0, "total_queries": 0},
            "meta": {"hits": 0, "total_queries": 0}
        }
        
        results = []
        
        print(f"Running memory layer analysis with {len(queries)} queries...")
        
        for i, query in enumerate(queries):
            if i % 10 == 0:
                print(f"Progress: {i}/{len(queries)}")
            
            # 记录查询前的记忆状态
            stats_before = plugin.get_memory_stats()
            
            # 处理查询
            result = await self.run_single_experiment(plugin, query)
            
            # 记录查询后的记忆状态
            stats_after = plugin.get_memory_stats()
            
            # 分析各层的使用情况
            for layer in ["shallow", "working", "deep", "meta"]:
                layer_usage[layer]["total_queries"] += 1
                if stats_after[layer]["access_count"] > stats_before[layer]["access_count"]:
                    layer_usage[layer]["hits"] += 1
            
            result["layer_analysis"] = {
                "stats_before": stats_before,
                "stats_after": stats_after,
                "layer_usage": layer_usage.copy()
            }
            
            results.append(result)
        
        # 计算最终的层使用统计
        final_layer_stats = {}
        for layer in layer_usage:
            hits = layer_usage[layer]["hits"]
            total = layer_usage[layer]["total_queries"]
            final_layer_stats[layer] = {
                "hits": hits,
                "total_queries": total,
                "hit_rate": (hits / total * 100) if total > 0 else 0
            }
        
        return {
            "results": results,
            "layer_statistics": final_layer_stats,
            "final_memory_stats": plugin.get_memory_stats()
        }
    
    async def run_performance_benchmark(self, test_scenarios: Dict[str, List[str]]) -> Dict[str, Any]:
        """运行性能基准测试"""
        
        benchmark_results = {
            "metadata": {
                "model_name": self.model_name,
                "tested_at": datetime.now().isoformat(),
                "scenarios": list(test_scenarios.keys())
            },
            "results": {}
        }
        
        for scenario_name, queries in test_scenarios.items():
            print(f"Running benchmark for scenario: {scenario_name}")
            
            if scenario_name == "baseline_comparison":
                result = await self.run_baseline_comparison(queries)
            elif scenario_name == "memory_layer_analysis":
                result = await self.run_memory_layer_analysis(queries)
            else:
                # 默认实验
                cleanup_plugin()
                plugin = SpatioTemporalMemoryPlugin(
                    model_name=self.model_name,
                    enable_logging=False
                )
                
                results = []
                for query in queries:
                    experiment_result = await self.run_single_experiment(plugin, query)
                    results.append(experiment_result)
                
                result = {
                    "results": results,
                    "summary": self._calculate_experiment_summary(results)
                }
            
            benchmark_results["results"][scenario_name] = result
        
        return benchmark_results
    
    def _calculate_experiment_summary(self, results: List[Dict]) -> Dict[str, Any]:
        """计算实验摘要"""
        
        successful_results = [r for r in results if r["status"] == "success"]
        
        if not successful_results:
            return {"error": "No successful results"}
        
        # 计算关键指标
        latencies = [r["latency_ms"] for r in successful_results]
        tokens = [r["total_tokens"] for r in successful_results]
        memory_tokens = [r["memory_tokens"] for r in successful_results]
        bleu_scores = [r["bleu_score"] for r in successful_results if r["bleu_score"] > 0]
        
        summary = {
            "total_queries": len(results),
            "successful_queries": len(successful_results),
            "success_rate": len(successful_results) / len(results) * 100,
            "latency": {
                "mean_ms": statistics.mean(latencies),
                "median_ms": statistics.median(latencies),
                "min_ms": min(latencies),
                "max_ms": max(latencies),
                "std_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0
            },
            "tokens": {
                "mean_total": statistics.mean(tokens),
                "mean_memory": statistics.mean(memory_tokens),
                "mean_response": statistics.mean([r["response_tokens"] for r in successful_results]),
                "total_saved": sum([r["memory_tokens"] for r in successful_results])  # 估算节省
            },
            "bleu": {
                "mean_score": statistics.mean(bleu_scores) if bleu_scores else 0,
                "median_score": statistics.median(bleu_scores) if bleu_scores else 0,
                "available": len(bleu_scores) > 0
            }
        }
        
        return summary
    
    def save_results(self, filename: str = None):
        """保存结果到文件"""
        if filename is None:
            filename = f"experiment_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to {filename}")
        return filename


async def main():
    """主实验函数"""
    
    print("=== 时空记忆系统性能评估实验 ===")
    print(f"BLEU评分可用: {BLEU_AVAILABLE}")
    
    import argparse
    
    parser = argparse.ArgumentParser(description="时空记忆系统性能评估实验")
    parser.add_argument("--use-llm", action="store_true", help="启用真实LLM API（OpenAI兼容 /v1/chat/completions）")
    parser.add_argument("--llm-base-url", default="", help="LLM Base URL，例如 https://api.openai.com/v1")
    parser.add_argument("--llm-model", default="", help="LLM 模型名，例如 gpt-4o-mini")
    parser.add_argument("--llm-api-key", default="", help="LLM API Key（也可用环境变量 LLM_API_KEY / OPENAI_API_KEY）")
    args = parser.parse_args()
    
    # 创建评估器
    evaluator = PerformanceEvaluator(
        model_name="openclaw-medium",
        use_llm=args.use_llm,
        llm_base_url=args.llm_base_url,
        llm_model=args.llm_model,
        llm_api_key=args.llm_api_key,
    )
    
    # 定义测试场景
    test_scenarios = {
        "basic_functionality": [
            "What is artificial intelligence?",
            "How does machine learning work?",
            "Tell me about neural networks",
            "What's the weather like today?",
            "Can you help me with something?"
        ],
        "memory_recall": [
            "Do you remember what we talked about earlier?",
            "What did I ask you about AI?",
            "Can you recall our previous conversation?",
            "What topics have we covered?",
            "Tell me about our discussion on machine learning"
        ],
        "baseline_comparison": [
            "What is artificial intelligence?",
            "How does machine learning relate to AI?",
            "Explain neural networks in simple terms",
            "What are the applications of AI?",
            "How can I learn more about machine learning?",
            "Tell me about deep learning",
            "What is computer vision?",
            "How does natural language processing work?"
        ],
        "memory_layer_analysis": [
            "What is AI?",
            "Tell me about machine learning",
            "How do neural networks work?",
            "What did we discuss earlier?",
            "Can you explain AI applications?",
            "What is deep learning?",
            "How is AI used in healthcare?",
            "Tell me something interesting about technology"
        ],
        "stress_test": [
            "AI ML DL NN CV NLP",
            "What is artificial intelligence in detail?",
            "Explain the difference between AI, ML, and DL",
            "How do convolutional neural networks work?",
            "What are the ethical implications of AI?",
            "Compare supervised and unsupervised learning",
            "What is transfer learning in deep learning?",
            "How does backpropagation work in neural networks?"
        ]
    }
    
    # 运行基准测试
    print("\n运行性能基准测试...")
    benchmark_results = await evaluator.run_performance_benchmark(test_scenarios)
    
    # 保存结果
    results_file = evaluator.save_results()
    
    # 打印摘要
    print("\n=== 实验结果摘要 ===")
    
    for scenario_name, result in benchmark_results["results"].items():
        print(f"\n{scenario_name.upper()}:")
        
        if "summary" in result:
            summary = result["summary"]
            if "error" not in summary:
                print(f"  成功率: {summary['success_rate']:.1f}%")
                print(f"  平均延迟: {summary['latency']['mean_ms']:.2f}ms")
                print(f"  平均Token数: {summary['tokens']['mean_total']:.1f}")
                print(f"  BLEU分数: {summary['bleu']['mean_score']:.3f}")
            else:
                print(f"  错误: {summary['error']}")
        
        elif "comparison" in result:
            comparison = result["comparison"]
            print(f"  Token减少: {comparison['tokens']['reduction_percent']:.1f}%")
            print(f"  延迟改进: {comparison['latency']['improvement_percent']:.1f}%")
            print(f"  BLEU改进: {comparison['bleu']['improvement_percent']:.1f}%")
            print(f"  记忆系统成功率: {comparison['success_rate']['with_memory']:.1f}%")
            print(f"  无记忆成功率: {comparison['success_rate']['without_memory']:.1f}%")
        
        elif "layer_statistics" in result:
            layer_stats = result["layer_statistics"]
            for layer, stats in layer_stats.items():
                print(f"  {layer.capitalize()}层命中率: {stats['hit_rate']:.1f}%")
    
    print(f"\n详细结果已保存到: {results_file}")
    
    # 生成简单的性能报告
    print("\n=== 性能指标达成情况 ===")
    
    # 检查延迟目标（≤100ms）
    if "basic_functionality" in benchmark_results["results"]:
        basic_summary = benchmark_results["results"]["basic_functionality"]["summary"]
        if "latency" in basic_summary:
            avg_latency = basic_summary["latency"]["mean_ms"]
            latency_target_met = avg_latency <= 100
            print(f"延迟目标 (≤100ms): {'✓' if latency_target_met else '✗'} ({avg_latency:.2f}ms)")
    
    # 检查token减少目标（≥30%）
    if "baseline_comparison" in benchmark_results["results"]:
        comparison = benchmark_results["results"]["baseline_comparison"]["comparison"]
        token_reduction = comparison["tokens"]["reduction_percent"]
        token_target_met = token_reduction >= 30
        print(f"Token减少目标 (≥30%): {'✓' if token_target_met else '✗'} ({token_reduction:.1f}%)")
    
    # 检查BLEU质量目标（≥baseline-2%）
    if "baseline_comparison" in benchmark_results["results"]:
        comparison = benchmark_results["results"]["baseline_comparison"]["comparison"]
        bleu_improvement = comparison["bleu"]["improvement_percent"]
        bleu_target_met = bleu_improvement >= -2  # 不低于基线2%
        print(f"回答质量目标 (BLEU≥baseline-2%): {'✓' if bleu_target_met else '✗'} ({bleu_improvement:.1f}%)")


if __name__ == "__main__":
    # 运行实验
    asyncio.run(main())
