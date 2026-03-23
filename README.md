# SpatioTemporal Memory System for OpenClaw

一个面向OpenClaw的玩具级智能体插件，通过分层时空记忆系统降低调用成本并提升推理效率。

## 🏗️ 架构概览

```
┌─────────────────────────────────────────────────────────────┐
│                    OpenClaw Plugin                          │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│              SpatioTemporalMemoryPlugin                      │
│  ┌───────────────────────────────────────────────────────┐    │
│  │              BudgetController                        │    │
│  │  - Token预算管理 (B: 模型上限的80%)                │    │
│  │  - 成本估算与优化                                   │    │
│  └─────────────────────┬─────────────────────────────┘    │
│                        │                                   │
│  ┌─────────────────────▼────────────────────────────────┐   │
│  │          SpatioTemporalRanker                        │   │
│  │  ┌────────────────────────────────────────────────┐   │   │
│  │  │  Context Construction Algorithm               │   │   │
│  │  │  score = α·语义相似度 + β·w(t) + γ·P(i,j)    │   │   │
│  │  │  w(t) = exp(-λΔt) 时间衰减权重               │   │   │
│  │  │  P(i,j) 层级转移概率矩阵                     │   │   │
│  │  └────────────────────┬─────────────────────────┘   │   │
│  │                       │                            │   │
│  │  ┌────────────────────▼────────────────────────┐  │   │
│  │  │    Semantic Similarity (Sentence-BERT)   │  │   │
│  │  └────────────────────────────────────────────┘  │   │
│  └──────────────────────┬─────────────────────────────┘   │
│                         │                                  │
└─────────────────────────┼──────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│              Memory Layer Architecture                     │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐      │
│  │ ShallowMemory │ │WorkingMemory │ │  DeepMemory  │      │
│  │   (TTL≈5min) │ │  (TTL≈30min) │ │ (Persistent) │      │
│  │  内存存储     │ │ Redis+LRU    │ │ SQLite/JSON  │      │
│  └──────┬───────┘ └──────┬───────┘ └──────┬───────┘      │
│         │                │                │               │
│  ┌──────▼────────────────▼────────────────▼────────────┐  │
│  │              MetaMemoryLayer                      │  │
│  │  记录"何时、为何、如何"访问各层                   │  │
│  │  用于动态层级调度                                  │  │
│  └────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 快速开始

### 环境要求
- Python 3.8+
- 8GB 内存（推荐）
- CPU-only 运行支持

### 安装依赖
```bash
cd STmemory
pip install -r requirements.txt
```

### 基本使用

#### 1. 命令行交互
```bash
# 启动交互式对话
python cli.py query

# 查看系统状态
python cli.py stats

# 运行演示
python cli.py demo

# 性能基准测试
python cli.py benchmark
```

#### 2. Python API
```python
from plugin import SpatioTemporalMemoryPlugin

# 初始化插件
plugin = SpatioTemporalMemoryPlugin(
    model="openclaw",
    config_path="config.json"
)

# 异步查询
response = await plugin.query("用户问题")
print(response)

# 获取统计信息
stats = plugin.get_stats()
print(f"Token节省: {stats['token_savings']:.1f}%")
```

## ⚙️ 配置参数

### 核心参数
```json
{
  "memory": {
    "shallow_capacity": 100,      // 浅层记忆容量
    "working_capacity": 500,      // 工作记忆容量  
    "deep_capacity": 2000,        // 深层记忆容量
    "ttl_shallow": 300,           // 浅层TTL (秒)
    "ttl_working": 1800,         // 工作记忆TTL (秒)
    "decay_lambda": 0.01,       // 时间衰减系数λ
    "alpha": 0.5,                 // 语义相似度权重
    "beta": 0.3,                  // 时间衰减权重
    "gamma": 0.2                  // 层级转移权重
  },
  "budget": {
    "model": "openclaw",          // LLM模型
    "budget_ratio": 0.8,         // Token预算比例
    "safety_margin": 0.1         // 安全余量
  },
  "performance": {
    "max_latency_ms": 100,       // 最大延迟
    "target_compression": 5.0,   // 目标压缩率
    "bleu_threshold": 0.98        // BLEU质量阈值
  }
}
```

### 模型配置
支持以下LLM模型：
- `openclaw` (默认)
- `gpt-3.5-turbo`
- `gpt-4`
- `claude`

## 📊 性能指标

### 目标指标
- ✅ **Token节省**: ≥30%
- ✅ **响应延迟**: ≤100ms
- ✅ **压缩比率**: ≥5×
- ✅ **回答质量**: BLEU ≥ baseline-2%

### 实际表现（示例）
```
=== 性能统计 ===
总查询次数: 100
Token节省: 35.2%
平均延迟: 78ms
压缩比率: 6.8×
BLEU分数: 0.96
缓存命中率: 82%
```

## 🧪 测试与验证

### 单元测试
```bash
# 运行所有测试
pytest tests/ -v

# 覆盖率报告
pytest tests/ --cov=. --cov-report=html

# 特定测试
pytest tests/test_memory_layers.py::test_shallow_memory_crud -v
```

### 性能实验
```bash
# 基准测试
python experiment.py --mode benchmark

# BLEU评估
python experiment.py --mode bleu

# 记忆层级分析
python experiment.py --mode memory-analysis
```

### 压力测试
```bash
# 多轮对话测试
python cli.py stress-test --rounds 50

# 并发测试
python cli.py stress-test --concurrent 10
```

## 🔧 内存管理

### 各层特性

| 层级 | 存储方式 | TTL | 容量 | 主要功能 |
|------|----------|-----|------|----------|
| 浅层 | 内存 | 5分钟 | 100条 | 快速缓存最近对话 |
| 工作 | Redis+LRU | 30分钟 | 500条 | 去重摘要关键信息 |
| 深层 | SQLite/JSON | 永久 | 2000条 | 结构化知识图谱 |
| 元记忆 | 内存+文件 | 持久 | 无限制 | 访问模式记录 |

### 内存优化策略
- 自动TTL过期清理
- LRU缓存淘汰
- 向量索引压缩
- 异步持久化

## 📈 使用示例

### 多轮对话演示
```bash
$ python cli.py demo

=== OpenClaw SpatioTemporal Memory Demo ===

用户: 什么是机器学习？
助手: 机器学习是人工智能的一个分支...
[记忆已添加到浅层缓存]

用户: 深度学习呢？
助手: 深度学习是机器学习的一个子集...
[检测到相关主题，激活工作记忆]

用户: 它们有什么区别？
[检索到之前的对话，构建上下文]
助手: 基于之前的讨论，主要区别在于...
[节省token: 45%]

=== 记忆层级变化 ===
浅层记忆: +2条 (TTL: 4:59)
工作记忆: +1条 (构建概念关联)
深层记忆: 0条 (未达到持久化阈值)
```

### API集成示例
```python
import asyncio
from plugin import SpatioTemporalMemoryPlugin

async def main():
    plugin = SpatioTemporalMemoryPlugin()
    
    # 多轮对话
    queries = [
        "Python和JavaScript有什么区别？",
        "哪个更适合数据科学？",
        "学习曲线如何？"
    ]
    
    for query in queries:
        response = await plugin.query(query)
        print(f"Q: {query}")
        print(f"A: {response[:100]}...")
        print()
    
    # 查看节省的token
    stats = plugin.get_stats()
    print(f"总共节省token: {stats['total_tokens_saved']}")

if __name__ == "__main__":
    asyncio.run(main())
```

## 🔍 调试与监控

### 日志级别
```python
import logging

# 设置调试级别
logging.basicConfig(level=logging.DEBUG)

# 查看详细日志
python cli.py query --debug
```

### 性能分析
```bash
# 内存使用分析
python -m memory_profiler cli.py demo

# 时间分析
python -m cProfile cli.py benchmark
```

## 🤝 贡献指南

1. Fork仓库并创建分支: `feature/your-feature`
2. 编写测试: 确保覆盖率≥90%
3. 运行测试: `pytest tests/`
4. 提交PR: 包含详细的变更说明

## 📄 许可证

MIT License - 详见LICENSE文件

## 🆘 常见问题

### Q: 内存使用过高怎么办？
A: 调整`shallow_capacity`和`working_capacity`参数，或降低`decay_lambda`值。

### Q: BLEU分数低于预期？
A: 检查`alpha`、`beta`、`gamma`权重配置，可能需要调整记忆检索策略。

### Q: 延迟超过100ms？
A: 考虑启用Redis缓存，或优化向量索引参数。

### 联系方式
如有问题，请在GitHub Issues中提交。
