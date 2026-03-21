import asyncio
import click
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional

from plugin import SpatioTemporalMemoryPlugin, get_plugin, cleanup_plugin
from memory_layers import MemoryConfig
from budget import BudgetController
from memory_layers import DeepMemoryLayer
from ranker import SpatioTemporalRanker


# Mock OpenClaw API函数
async def mock_openclaw_api(prompt: str) -> str:
    """模拟OpenClaw API调用"""
    # 简单的模拟响应
    if "hello" in prompt.lower():
        return "Hello! How can I help you today?"
    elif "weather" in prompt.lower():
        return "The weather is nice today. Sunny with a light breeze."
    elif "memory" in prompt.lower():
        return "I can help you with memory-related questions. What would you like to know?"
    elif "help" in prompt.lower():
        return "I can help you with various tasks. Feel free to ask me anything!"
    else:
        return f"I understand you're asking about: {prompt[:50]}... Let me think about that."


@click.group()
def cli():
    """分层时空记忆系统 CLI工具"""
    pass


@cli.command()
@click.option('--query', '-q', required=True, help='用户查询')
@click.option('--model', '-m', default='openclaw-medium', help='模型名称')
@click.option('--config', '-c', help='配置文件路径')
@click.option('--verbose', '-v', is_flag=True, help='详细输出')
async def query(query: str, model: str, config: Optional[str], verbose: bool):
    """处理单个查询"""
    try:
        # 加载配置
        memory_config = None
        if config:
            with open(config, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
                memory_config = MemoryConfig(**config_data.get('memory_config', {}))
        
        # 获取插件实例
        plugin = get_plugin(model_name=model, memory_config=memory_config, enable_logging=verbose)
        
        click.echo(f"Processing query: {query}")
        click.echo(f"Using model: {model}")
        
        # 处理查询
        result = await plugin.process_query(query, mock_openclaw_api)
        
        # 输出结果
        click.echo(f"\nResponse: {result['response']}")
        click.echo(f"Status: {result['status']}")
        click.echo(f"Latency: {result['latency_ms']:.2f}ms")
        click.echo(f"Memory tokens: {result['memory_tokens']}")
        click.echo(f"Response tokens: {result['response_tokens']}")
        click.echo(f"Total tokens: {result['total_tokens']}")
        click.echo(f"Relevant memories: {result['relevant_memories_count']}")
        
        if verbose and 'budget_check' in result:
            budget_check = result['budget_check']
            click.echo(f"Budget check: {'✓' if budget_check['within_limits'] else '✗'}")
            click.echo(f"Memory ratio: {budget_check['memory_ratio']:.2%}")
            click.echo(f"Response ratio: {budget_check['response_ratio']:.2%}")
        
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)


@cli.command()
@click.option('--count', '-n', default=5, help='对话轮数')
@click.option('--model', '-m', default='openclaw-medium', help='模型名称')
@click.option('--verbose', '-v', is_flag=True, help='详细输出')
async def demo(count: int, model: str, verbose: bool):
    """运行演示对话"""
    click.echo("=== 分层时空记忆系统演示 ===\n")
    
    # 获取插件实例
    plugin = get_plugin(model_name=model, enable_logging=verbose)
    
    # 演示查询序列
    demo_queries = [
        "Hello, can you help me?",
        "What's the weather like today?",
        "Do you remember what I asked about earlier?",
        "Can you explain how memory works?",
        "What did we talk about in our previous conversations?"
    ]
    
    # 限制查询数量
    queries = demo_queries[:count]
    
    for i, query in enumerate(queries, 1):
        click.echo(f"\n--- 对话 {i}/{len(queries)} ---")
        click.echo(f"用户: {query}")
        
        try:
            result = await plugin.process_query(query, mock_openclaw_api)
            
            click.echo(f"助手: {result['response']}")
            click.echo(f"延迟: {result['latency_ms']:.2f}ms | Token: {result['total_tokens']} | 记忆: {result['relevant_memories_count']}")
            
            # 显示记忆层级变化
            if verbose:
                stats = plugin.get_memory_stats()
                click.echo(f"记忆统计 - 浅层: {stats['shallow']['entries']} | 工作: {stats['working']['entries']} | 深层: {stats['deep']['entries']}")
            
            # 短暂延迟以模拟真实对话
            await asyncio.sleep(0.5)
            
        except Exception as e:
            click.echo(f"错误: {str(e)}", err=True)
    
    # 显示最终统计
    click.echo("\n=== 演示完成 ===")
    stats = plugin.get_performance_stats()
    
    click.echo(f"总查询数: {stats['performance']['total_queries']}")
    click.echo(f"平均延迟: {stats['performance']['average_latency_ms']:.2f}ms")
    click.echo(f"节省Token: {stats['performance']['total_tokens_saved']}")
    click.echo(f"节省成本: ${stats['performance']['total_cost_saved']:.4f}")


@cli.command()
@click.option('--model', '-m', default='openclaw-medium', help='模型名称')
@click.option('--output', '-o', help='输出文件路径')
def stats(model: str, output: Optional[str]):
    """显示系统统计信息"""
    plugin = get_plugin(model_name=model, enable_logging=False)
    
    memory_stats = plugin.get_memory_stats()
    performance_stats = plugin.get_performance_stats()


@cli.command()
@click.option('--config', '-c', help='配置文件路径')
@click.option('--db', help='深层记忆SQLite路径')
@click.option('--provider', default=None, help='向量库后端: numpy/sqlite')
@click.option('--dedup/--no-dedup', default=True, help='是否执行精确去重')
@click.option('--compress/--no-compress', default=True, help='是否执行聚类压缩')
@click.option('--threshold', default=0.88, show_default=True, help='聚类相似度阈值')
@click.option('--min-cluster-size', default=3, show_default=True, help='最小簇大小')
@click.option('--max-clusters', default=5, show_default=True, help='最多处理簇数量')
@click.option('--delete-sources/--keep-sources', default=True, help='压缩后是否删除源记忆')
def optimize_deep(
    config: Optional[str],
    db: Optional[str],
    provider: Optional[str],
    dedup: bool,
    compress: bool,
    threshold: float,
    min_cluster_size: int,
    max_clusters: int,
    delete_sources: bool,
):
    memory_config = None
    if config:
        with open(config, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
            memory_config = MemoryConfig(**config_data.get('memory_config', {}))
    memory_config = memory_config or MemoryConfig()
    if db:
        memory_config.deep_persist_path = str(db)
    if provider:
        memory_config.deep_vector_store_provider = str(provider)

    layer = DeepMemoryLayer(memory_config)
    try:
        ranker = SpatioTemporalRanker(memory_config)
        layer.set_encoder(ranker._encode_texts)
    except Exception:
        pass

    stats = layer.optimize(
        deduplicate_exact=bool(dedup),
        compress=bool(compress),
        similarity_threshold=float(threshold),
        min_cluster_size=int(min_cluster_size),
        max_clusters=int(max_clusters),
        delete_sources=bool(delete_sources),
    )
    click.echo(json.dumps(stats, ensure_ascii=False, indent=2))
    
    stats_data = {
        "timestamp": datetime.now().isoformat(),
        "memory_stats": memory_stats,
        "performance_stats": performance_stats
    }
    
    if output:
        with open(output, 'w', encoding='utf-8') as f:
            json.dump(stats_data, f, indent=2, ensure_ascii=False)
        click.echo(f"统计信息已保存到: {output}")
    else:
        click.echo("=== 记忆系统统计 ===")
        click.echo(f"总条目数: {memory_stats['total']['entries']}")
        click.echo(f"总Token数: {memory_stats['total']['tokens']}")
        
        for layer_name in ['shallow', 'working', 'deep', 'meta']:
            if layer_name in memory_stats:
                layer = memory_stats[layer_name]
                click.echo(f"{layer_name.capitalize()}层: {layer['entries']} 条目, {layer.get('total_tokens', 0)} tokens")
        
        click.echo("\n=== 性能统计 ===")
        perf = performance_stats['performance']
        click.echo(f"总查询数: {perf['total_queries']}")
        click.echo(f"平均延迟: {perf['average_latency_ms']:.2f}ms")
        click.echo(f"节省Token: {perf['total_tokens_saved']}")
        click.echo(f"节省成本: ${perf['total_cost_saved']:.4f}")


@cli.command()
@click.option('--output', '-o', default='config.json', help='输出文件路径')
def generate_config(output: str):
    """生成配置文件"""
    config = {
        "model_name": "openclaw-medium",
        "memory_config": {
            "shallow_ttl": 300,
            "working_ttl": 1800,
            "deep_persist_path": "deep_memory.db",
            "max_shallow_entries": 100,
            "max_working_entries": 50,
            "lambda_decay": 0.1,
            "alpha_similarity": 0.6,
            "beta_time": 0.3,
            "gamma_layer": 0.1,
            "token_budget_ratio": 0.8,
            "llm_provider": "ollama",
            "llm_base_url": "https://api.openai.com/v1",
            "llm_model": "gpt-4o-mini",
            "local_llm_base_url": "http://localhost:11434/v1",
            "local_llm_model": "qwen3.5-9b",
            "local_llm_timeout_s": 60,
            "local_llm_check": True,
            "deep_enable_vector_index": True,
            "deep_vector_store_provider": "numpy",
            "deep_vector_distance": "cosine",
            "qdrant_url": "http://localhost:6333",
            "qdrant_collection": "stmemory_deep"
        },
        "budget_config": {
            "system_prompt_ratio": 0.1
        }
    }
    
    with open(output, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    click.echo(f"配置文件已生成: {output}")


@cli.command()
@click.option('--model', '-m', default='openclaw-medium', help='模型名称')
@click.option('--interval', '-i', default=60, help='清理间隔（秒）')
async def cleanup(model: str, interval: int):
    """清理过期记忆"""
    plugin = get_plugin(model_name=model, enable_logging=True)
    
    click.echo(f"开始清理过期记忆，间隔: {interval}秒")
    
    while True:
        try:
            removed = plugin.cleanup_expired_memories()
            if removed > 0:
                click.echo(f"清理完成，移除 {removed} 条过期记忆")
            
            await asyncio.sleep(interval)
            
        except KeyboardInterrupt:
            click.echo("清理任务已停止")
            break
        except Exception as e:
            click.echo(f"清理错误: {str(e)}", err=True)


@cli.command()
def benchmark():
    """运行性能基准测试"""
    click.echo("=== 性能基准测试 ===")
    
    # 模拟多个查询
    test_queries = [
        "Hello",
        "What's the weather?",
        "Tell me about memory",
        "How does this work?",
        "Can you help me?",
        "What time is it?",
        "What's new?",
        "Goodbye"
    ]
    
    plugin = get_plugin(enable_logging=False)
    
    # 测试记忆检索性能
    start_time = time.time()
    
    for query in test_queries:
        asyncio.run(plugin.retrieve_relevant_memories(query))
    
    retrieval_time = (time.time() - start_time) * 1000
    
    # 测试完整查询性能
    start_time = time.time()
    
    async def run_test_queries():
        for query in test_queries:
            await plugin.process_query(query, mock_openclaw_api)
    
    asyncio.run(run_test_queries())
    
    total_time = (time.time() - start_time) * 1000
    avg_time = total_time / len(test_queries)
    
    click.echo(f"记忆检索平均延迟: {retrieval_time/len(test_queries):.2f}ms")
    click.echo(f"完整查询平均延迟: {avg_time:.2f}ms")
    click.echo(f"总测试时间: {total_time:.2f}ms")
    
    stats = plugin.get_performance_stats()
    click.echo(f"节省Token: {stats['performance']['total_tokens_saved']}")
    click.echo(f"节省成本: ${stats['performance']['total_cost_saved']:.4f}")


@cli.command()
def reset():
    """重置所有统计和记忆"""
    if click.confirm("确定要重置所有统计和记忆吗？此操作不可恢复。"):
        plugin = get_plugin()
        plugin.reset_statistics()
        
        # 清理所有记忆层
        for layer_name in ['shallow', 'working', 'deep', 'meta']:
            if layer_name in plugin.layers:
                plugin.layers[layer_name] = type(plugin.layers[layer_name])(plugin.config)
        
        click.echo("所有统计和记忆已重置")
    else:
        click.echo("操作已取消")


if __name__ == '__main__':
    cli()
