# cli：命令行工具（参数说明）

对应源码：[cli.py](../../cli.py)。

## 1. 模块职责

- 提供最小可用的命令行入口，用于：
  - 单次查询演示（默认使用 mock LLM）
  - 多轮 demo
  - 导出统计
  - Deep 记忆优化（去重/聚类压缩）
  - 生成配置模板

## 2. 命令一览

CLI 入口为 click group：`python cli.py <command> [options]`。

### 2.1 query：处理单个查询

命令：

```bash
python cli.py query -q "..." [-m openclaw-medium] [-c path/to/config.json] [-v]
```

参数：

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---:|---|
| --query / -q | str | 必填 | 用户问题 |
| --model / -m | str | openclaw-medium | 传给 BudgetController 的模型名 |
| --config / -c | str | None | JSON 配置路径（读取 `memory_config` 并构造 `MemoryConfig(**...)`） |
| --verbose / -v | bool | False | 打印详细日志与预算比例 |

输出（stdout）包含：响应、延迟、token 使用、命中记忆条数、预算检查结果等。

### 2.2 demo：多轮演示

```bash
python cli.py demo [-n 5] [-m openclaw-medium] [-v]
```

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---:|---|
| --count / -n | int | 5 | 演示轮数（最多 5，取内置 demo_queries 前 N 条） |
| --model / -m | str | openclaw-medium | 同上 |
| --verbose / -v | bool | False | 额外打印分层记忆条数变化 |

### 2.3 stats：显示系统统计信息

```bash
python cli.py stats [-m openclaw-medium] [-o output.json]
```

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---:|---|
| --model / -m | str | openclaw-medium | 同上 |
| --output / -o | str | None | 若提供则写入 JSON，否则打印到 stdout |

### 2.4 optimize-deep：Deep 记忆去重/聚类压缩

```bash
python cli.py optimize-deep [--config cfg.json] [--db deep.db] [--provider numpy|sqlite] \
  [--dedup/--no-dedup] [--compress/--no-compress] \
  [--threshold 0.88] [--min-cluster-size 3] [--max-clusters 5] \
  [--delete-sources/--keep-sources]
```

参数：

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---:|---|
| --config | str | None | 读取 `memory_config` 覆盖默认值 |
| --db | str | None | 覆盖 `deep_persist_path` |
| --provider | str | None | 覆盖 `deep_vector_store_provider`（仅支持 numpy/sqlite） |
| --dedup/--no-dedup | bool | True | 是否执行精确去重 |
| --compress/--no-compress | bool | True | 是否执行聚类压缩 |
| --threshold | float | 0.88 | 聚类相似度阈值 |
| --min-cluster-size | int | 3 | 最小簇大小 |
| --max-clusters | int | 5 | 最多处理簇数量 |
| --delete-sources/--keep-sources | bool | True | 是否删除源条目 |

该命令会直接构造 `DeepMemoryLayer` 并尽力注入 encoder，然后调用 `layer.optimize(...)`。

### 2.5 generate-config：生成配置模板

```bash
python cli.py generate-config [-o config.json]
```

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---:|---|
| --output / -o | str | config.json | 输出文件路径 |

生成的 JSON 同时包含 `memory_config` 与 `budget_config`，作为快速起步模板。
