# LOCOMO 基准测试（集成跑、参数、产物与指标）

本项目内置 LOCOMO 集成跑：自动拉起本地服务 → 执行 add/search → 生成评测产物 → 汇总报告与阈值判定。

核心入口：
- 集成跑： [benchmark/locomo/run.py](../../benchmark/locomo/run.py)
- 服务端： [benchmark/locomo/server.py](../../benchmark/locomo/server.py)

## 1. 快速开始（推荐：mock）

```bash
python -m benchmark.locomo.run --use-mock-openai
```

该模式不依赖外部 LLM；会使用 server 的 mock OpenAI 端点回答与判别。

## 2. 集成跑参数（run.py）

命令：`python -m benchmark.locomo.run [options]`

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---:|---|
| --config | str | None | JSON 配置：可包含 `memory_config` 与 thresholds 等 |
| --host | str | 127.0.0.1 | 本地 server host |
| --port | int | 8000 | 本地 server 端口 |
| --output-dir | str | None | 输出目录；缺省则写入 `benchmark/results/locomo/<timestamp>` |
| --dataset | str | benchmark/locomo/dataset/locomo10.json | 数据集路径 |
| --model | str | qwen3.5-9b | LLM 模型名（用于 runner/报告标记） |
| --llm-provider | str | ollama | `ollama/openai_compat/anthropic`（内部会归一化） |
| --openai-base-url | str | env OPENAI_BASE_URL 或空 | openai-compatible base url |
| --openai-api-key | str | env OPENAI_API_KEY 或空 | openai-compatible api key |
| --minimax-api-key | str | env MINIMAX_API_KEY/ANTHROPIC_API_KEY 或空 | anthropic/minimax key（为空会回退 mock） |
| --top-k | int | 30 | search 阶段的 top_k |
| --filter-memories | bool | False | 传给 LOCOMO 方法脚本的过滤开关 |
| --is-graph | bool | False | 传给 LOCOMO 方法脚本的 graph 模式开关 |
| --max-workers | int | 自适应 | 并发 worker 上限（与 CPU/内存相关） |
| --use-mock-openai / --use-mock-llm | bool | False | 强制使用 mock |
| --probe-size | int | 1 | 抽样探测大小（用于 smoke run） |
| --probe-seed | int | 42 | 探测抽样种子 |
| --subset-index | int | None | 只跑数据集中的某个 item |
| --max-qa | int | 1 | 每个 item 最多评测 QA 数 |
| --skip-probe | bool | False | 跳过 probe |
| --run-full | bool | False | 预留：当前实现会强制置 False |
| --max-add-p95-s | float | None | add 阶段 p95 延迟阈值 |
| --max-search-p95-s | float | None | search 阶段 p95 延迟阈值 |
| --fail-on-threshold | bool | False | 超阈值时以失败退出 |

参数定义位置见：[run.py](../../benchmark/locomo/run.py#L336-L363)。

## 3. 服务端参数（server.py）

命令：`python -m benchmark.locomo.server [options]`

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---:|---|
| --config | str | None | 用于覆盖 `MemoryConfig`（会允许旧字段映射） |
| --host | str | 127.0.0.1 | 监听地址 |
| --port | int | 8000 | 监听端口 |
| --enable-mock-openai | bool | False | 启用 `/v1/chat/completions` mock |
| --dataset | str | None | mock answers 用数据集路径（缺省用 locomo10.json） |

服务 API 见：[docs/api/http.md](../api/http.md)。

## 4. 产物目录结构（output-dir）

典型产物（以具体运行目录为准）：
- `report.json`：汇总报告（含阈值判定、质量摘要、延迟统计、路径指针等）
- `evaluation.txt`：评测日志与摘要（含 latency 解析片段）
- `evaluation_metrics.csv`：质量指标表
- `metrics.jsonl`：逐条记录（含 judge latency 等）

可视化 demo 会读取上述产物并展示（见 [visual_demo/README.md](../../visual_demo/README.md)）。

## 5. 指标口径（概要）

### 5.1 延迟

run.py 会解析：
- answer latency：来自 experiments 输出（按字段 `response_time`）
- judge latency：来自 `metrics.jsonl` 中 `type=judge` 的记录

并给出 `avg/p95/max` 统计（见 [run.py](../../benchmark/locomo/run.py#L156-L169)）。

### 5.2 质量

质量汇总由 `summarize_metrics(evaluation_metrics.csv)` 生成（见 [reporting.py](../../benchmark/locomo/reporting.py)）。

## 6. 阈值判定与失败策略

阈值来源：
- CLI 参数 `--max-add-p95-s/--max-search-p95-s`
- 或 `--config` JSON 内的 `thresholds.max_add_p95_s/max_search_p95_s`

当 `--fail-on-threshold` 为 True 且指标超阈值，集成跑会以失败退出（用于 CI/回归门禁）。
