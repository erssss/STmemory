# STmemory：分层时空记忆系统（Inference-time Memory）

![build](https://img.shields.io/badge/build-manual-lightgrey)
![version](https://img.shields.io/badge/version-1.0.0-blue)
![license](https://img.shields.io/badge/license-MIT-green)
![coverage](https://img.shields.io/badge/coverage-unknown-lightgrey)

STmemory 是一个“推理期（inference-time）”记忆增强系统：通过分层时空记忆（Shallow/Working/Deep/Meta）+ Token 预算 + 时空排序，在不把全量历史塞进 prompt 的前提下，为每次新 query 构造紧凑上下文并调用 OpenAI-compatible LLM API。

## 功能概览

- 分层记忆：LRU+TTL（shallow/working）+ SQLite 持久化（deep）+ 访问统计（meta）
- 检索与排序：语义相似 + 时间近因 + 层级偏好融合打分，并在预算内选择子集
- 向量检索后端：numpy/sqlite/qdrant 可选切换
- 可视化演示：浏览器 UI 观测检索/上下文/写回/统计
- 基准测试：LOCOMO 集成跑（可阈值门禁，支持 mock）

## 架构速览

```mermaid
flowchart TB
  U[User] --> P[SpatioTemporalMemoryPlugin]
  P --> B[BudgetController]
  P --> L[Memory Layers]
  L --> SM[Shallow]
  L --> WM[Working]
  L --> DM[Deep (SQLite + VectorStore)]
  L --> MM[Meta]
  P --> R[Ranker]
  P --> API[OpenAI-compatible LLM]
  API --> P
```

更完整的架构/时序/状态图见：[docs/architecture/overview.md](docs/architecture/overview.md)。

## 安装指南

### 系统要求

- Python >= 3.8
- 推荐内存：8GB+
- 可选：Qdrant（当 `deep_vector_store_provider="qdrant"`）

### 安装依赖

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

LOCOMO 基准额外依赖：

```bash
python -m pip install -r benchmark/requirements.txt
```

### 验证步骤

```bash
python quick_start.py
pytest -q
```

## 5 分钟上手（完整示例）

运行：

```bash
python quick_start.py
```

预期输出（截断示例）：

```text
Python版本: ...
🚀 SpatioTemporal Memory System - 快速开始
🔄 初始化记忆系统...
✅ 系统初始化完成！
📝 开始多轮对话演示:
用户: 什么是人工智能？
助手: ...
  📋 浅层记忆: +1
  🧠 工作记忆: +1
  💎 深层记忆: +1
  📈 元记忆: +1
```

## 使用方式

### Python API（最常用）

```python
import asyncio
from plugin import SpatioTemporalMemoryPlugin, make_configured_llm_api_func
from memory_layers import MemoryConfig

async def main():
    cfg = MemoryConfig(llm_provider="remote")
    plugin = SpatioTemporalMemoryPlugin(model_name="openclaw-medium", memory_config=cfg, enable_logging=False)

    llm = make_configured_llm_api_func(cfg)
    result = await plugin.process_query("我们上次聊到的时间衰减公式是什么？", llm)
    print(result["response"])

asyncio.run(main())
```

### CLI（演示与运维）

```bash
python cli.py --help
python cli.py demo -n 3
python cli.py query -q "hello" -v
```

CLI 参数说明见：[docs/modules/cli.md](docs/modules/cli.md)。

### 可视化演示（Web UI）

```bash
python -m visual_demo.server --host 127.0.0.1 --port 8765
```

服务说明与 API 见：
- [docs/services/visual_demo.md](docs/services/visual_demo.md)
- [docs/api/http.md](docs/api/http.md)

## 模块索引（功能/参数/原理摘要）

| 模块 | 功能 | 核心参数（示例） | 原理摘要 | 文档 |
|---|---|---|---|---|
| plugin | 端到端编排：预算→检索→排序→LLM→写回 | llm_provider / local_llm_* | 检索后拼接 context，再调用 OpenAI-compatible | [docs/modules/plugin.md](docs/modules/plugin.md) |
| memory_layers | 分层记忆与 Deep 检索 | shallow_ttl / deep_enable_vector_index | 实体/向量/时间候选 + 词面/时间融合重排 | [docs/modules/memory_layers.md](docs/modules/memory_layers.md) |
| ranker | 时空排序与预算内选择 | alpha/beta/gamma/lambda_decay | score=α·sim+β·time+γ·layer；贪心近似 knapsack | [docs/modules/ranker.md](docs/modules/ranker.md) |
| budget | token 预算控制 | system_prompt_ratio | 分配 system/response/memory 预算并检查超限 | [docs/modules/budget.md](docs/modules/budget.md) |
| vector_store | 向量库抽象与实现 | deep_vector_store_provider / qdrant_* | numpy 全扫；qdrant ANN + 可选重排 | [docs/modules/vector_store.md](docs/modules/vector_store.md) |
| compression | 压缩与 token 估算 | compression_target_ratio | 句子打分 + 装包 + 迭代收紧 | [docs/modules/compression.md](docs/modules/compression.md) |
| temporal_model | 时间解析与匹配分 | - | 规则解析 TimeRange；三角形匹配分 | [docs/modules/temporal_model.md](docs/modules/temporal_model.md) |

## 性能基准测试（LOCOMO）

基准文档：[docs/benchmark/locomo.md](docs/benchmark/locomo.md)

本 README 的数据来自以下命令（mock 模式，CPU 环境）：

```bash
/usr/bin/time -v python -m benchmark.locomo.run --use-mock-openai --probe-size 1  --max-workers 2 --max-qa 1 --output-dir benchmark/results/locomo_readme_probe1
/usr/bin/time -v python -m benchmark.locomo.run --use-mock-openai --probe-size 3  --max-workers 2 --max-qa 1 --output-dir benchmark/results/locomo_readme_probe3
/usr/bin/time -v python -m benchmark.locomo.run --use-mock-openai --probe-size 10 --max-workers 2 --max-qa 1 --output-dir benchmark/results/locomo_readme_probe10
```

结果摘要（mock 下质量指标为 1.0，延迟统计为 0.0；更适合用于回归门禁与产物链路验证）：

| 数据规模（probe-size） | 端到端耗时（wall） | 峰值 RSS（GiB，time -v） | BLEU（mean） | F1（mean） |
|---:|---:|---:|---:|---:|
| 1  | 5.92s | 0.94 | 1.00 | 1.00 |
| 3  | 7.78s | 0.95 | 1.00 | 1.00 |
| 10 | 10.15s | 0.98 | 1.00 | 1.00 |

## API 参考

- Python API：核心类为 `SpatioTemporalMemoryPlugin`（见 [plugin.py](plugin.py)）
  - `process_query/retrieve_relevant_memories/vector_search/get_performance_stats`
- HTTP API：可视化服务与 LOCOMO server 端点见：[docs/api/http.md](docs/api/http.md)

## 文档入口

- 文档盘点与迁移入口：[docs/document-inventory.md](docs/document-inventory.md)
- 架构与流程：[docs/architecture/overview.md](docs/architecture/overview.md)
- 配置与调参：[docs/configuration/overview.md](docs/configuration/overview.md)
- 模块参数/原理索引：[docs/modules/overview.md](docs/modules/overview.md)

## 许可证

MIT License（见 LICENSE）。
