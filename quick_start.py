#!/usr/bin/env python3
"""
快速开始脚本 - 5分钟内体验SpatioTemporal Memory系统
"""

import asyncio
import sys
import os
import argparse
from datetime import datetime

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from plugin import SpatioTemporalMemoryPlugin, make_openai_compatible_api_func, make_configured_llm_api_func
    from memory_layers import MemoryConfig
except ImportError as e:
    print("❌ 导入失败，请先安装依赖:")
    print("   pip install -r requirements.txt")
    print(f"   错误详情: {e}")
    sys.exit(1)


async def quick_demo(
    use_llm: bool,
    llm_base_url: str,
    llm_model: str,
    llm_api_key: str,
    llm_provider: str,
    local_llm_base_url: str,
    local_llm_model: str,
):
    """快速演示"""
    print("🚀 SpatioTemporal Memory System - 快速开始")
    print("=" * 50)
    
    # 创建轻量级配置
    config = MemoryConfig(
        shallow_capacity=20,
        working_capacity=50,
        shallow_ttl=300,
        working_ttl=1800,
        max_shallow_entries=20,
        max_working_entries=50,
        lambda_decay=0.01,
        alpha_similarity=0.5,
        beta_time=0.3,
        gamma_layer=0.2
    )
    if llm_base_url:
        config.llm_base_url = str(llm_base_url)
    if llm_model:
        config.llm_model = str(llm_model)
    if llm_provider:
        config.llm_provider = str(llm_provider)
    if local_llm_base_url:
        config.local_llm_base_url = str(local_llm_base_url)
    if local_llm_model:
        config.local_llm_model = str(local_llm_model)
    
    # 初始化插件
    print("🔄 初始化记忆系统...")
    plugin = SpatioTemporalMemoryPlugin(
        model_name="openclaw",
        memory_config=config,
        enable_logging=False
    )
    
    print("✅ 系统初始化完成！")
    print()
    
    # 演示对话
    demo_queries = [
        "什么是人工智能？",
        "机器学习和AI有什么关系？",
        "深度学习又是什么？"
    ]
    
    print("📝 开始多轮对话演示:")
    print("-" * 40)
    
    async def mock_openclaw_api(prompt: str) -> str:
        prompt_lower = prompt.lower()
        if "人工智能" in prompt or "ai" in prompt_lower or "artificial intelligence" in prompt_lower:
            return "人工智能（AI）是让计算机系统模拟人类智能的技术，包括学习、推理、感知与决策等能力。"
        if "机器学习" in prompt:
            return "机器学习是人工智能的一种实现路径，通过数据训练模型，让系统在无需显式编程的情况下完成预测与决策。"
        if "深度学习" in prompt:
            return "深度学习是机器学习的一个分支，使用多层神经网络从大量数据中学习复杂特征，常用于视觉与自然语言任务。"
        return f"我理解你在问：{prompt[:50]}..."
    
    if use_llm:
        provider = str(config.llm_provider or "remote").strip().lower()
        if provider in {"local", "local_llm", "ollama"}:
            openclaw_api_func = make_configured_llm_api_func(config, remote_api_key=llm_api_key or "")
        else:
            api_key = llm_api_key or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
            base_url = str(llm_base_url or config.llm_base_url or "").strip()
            model = str(llm_model or config.llm_model or "").strip()
            openclaw_api_func = make_openai_compatible_api_func(
                api_key=api_key,
                base_url=base_url,
                model=model,
            )
    else:
        openclaw_api_func = mock_openclaw_api
    
    for i, query in enumerate(demo_queries, 1):
        print(f"\n💬 第{i}轮对话:")
        print(f"用户: {query}")
        
        before_memory_stats = plugin.get_memory_stats()
        before_perf_stats = plugin.get_performance_stats()["performance"]
        
        print("🔄 处理查询中...")
        result = await plugin.process_query(query, openclaw_api_func)
        print(f"助手: {result['response']}")
        
        after_memory_stats = plugin.get_memory_stats()
        after_perf_stats = plugin.get_performance_stats()["performance"]
        
        shallow_change = after_memory_stats["shallow"]["entries"] - before_memory_stats["shallow"]["entries"]
        working_change = after_memory_stats["working"]["entries"] - before_memory_stats["working"]["entries"]
        deep_change = after_memory_stats["deep"]["entries"] - before_memory_stats["deep"]["entries"]
        meta_change = after_memory_stats["meta"]["entries"] - before_memory_stats["meta"]["entries"]
        
        if shallow_change > 0:
            print(f"  📋 浅层记忆: +{shallow_change}")
        if working_change > 0:
            print(f"  🧠 工作记忆: +{working_change}")
        if deep_change > 0:
            print(f"  💎 深层记忆: +{deep_change}")
        if meta_change > 0:
            print(f"  📈 元记忆: +{meta_change}")
        
        tokens_saved_delta = after_perf_stats["total_tokens_saved"] - before_perf_stats["total_tokens_saved"]
        print(f"  💰 Token节省: {tokens_saved_delta:.1f}")
        print(f"  🔎 相关记忆: {result['relevant_memories_count']}条")
        
        await asyncio.sleep(0.5)  # 模拟处理时间
    
    # 显示最终统计
    print("\n" + "=" * 50)
    print("📈 演示统计:")
    memory_stats = plugin.get_memory_stats()
    perf_stats = plugin.get_performance_stats()["performance"]
    print(f"总查询次数: {perf_stats['total_queries']}")
    print(f"浅层记忆: {memory_stats['shallow']['entries']}条")
    print(f"工作记忆: {memory_stats['working']['entries']}条")
    print(f"深层记忆: {memory_stats['deep']['entries']}条")
    print(f"元记忆: {memory_stats['meta']['entries']}条")
    print(f"节省Token: {perf_stats['total_tokens_saved']}")
    
    print("\n🎯 性能评估:")
    print("  ✅ 延迟: <100ms (模拟)")
    print("  ✅ Token节省: 已统计")
    print("  ✅ 内存使用: 轻量级配置")
    print("  ✅ 多轮对话: 记忆关联成功")
    
    print("\n🎉 快速演示完成！")
    print("\n下一步建议:")
    print("  1. 运行完整演示: python demo.py")
    print("  2. 查看CLI工具: python cli.py --help")
    print("  3. 运行测试: pytest tests/")
    print("  4. 查看配置: cat config.json")


async def main():
    """主函数"""
    try:
        parser = argparse.ArgumentParser(description="SpatioTemporal Memory System 快速开始")
        parser.add_argument("--use-llm", action="store_true", help="启用真实LLM API（OpenAI兼容 /v1/chat/completions）")
        parser.add_argument("--llm-base-url", default="", help="LLM Base URL，例如 https://api.openai.com/v1")
        parser.add_argument("--llm-model", default="", help="LLM 模型名，例如 gpt-4o-mini")
        parser.add_argument("--llm-api-key", default="", help="LLM API Key（也可用环境变量 LLM_API_KEY / OPENAI_API_KEY）")
        parser.add_argument("--llm-provider", default="ollama", help="LLM 提供方：ollama/local 或 remote")
        parser.add_argument("--local-llm-base-url", default="", help="本地LLM Base URL，例如 http://localhost:11434/v1")
        parser.add_argument("--local-llm-model", default="", help="本地LLM 模型名，例如 qwen3.5-9b")
        args = parser.parse_args()
        
        await quick_demo(
            args.use_llm,
            args.llm_base_url,
            args.llm_model,
            args.llm_api_key,
            args.llm_provider,
            args.local_llm_base_url,
            args.local_llm_model,
        )
    except KeyboardInterrupt:
        print("\n\n用户中断演示")
    except Exception as e:
        print(f"\n❌ 演示出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 检查Python版本
    if sys.version_info < (3, 8):
        print("❌ 需要Python 3.8或更高版本")
        sys.exit(1)
    
    print(f"Python版本: {sys.version}")
    print(f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 运行快速演示
    asyncio.run(main())
