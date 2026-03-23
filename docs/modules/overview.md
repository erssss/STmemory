# 核心模块索引（参数/原理入口）

本项目为“分层时空记忆 + 预算 + 排序”的推理增强系统，并不包含传统意义上的模型训练管线（trainer/dataset/optimizer 等）。核心能力集中在根目录若干 Python 文件与两个服务入口（可视化与基准）。

## 模块列表

| 模块 | 作用 | 关键配置入口 | 文档 |
|---|---|---|---|
| 编排器 | 端到端请求链路：预算→检索→排序→LLM→写回 | MemoryConfig + BudgetController | [plugin.md](plugin.md) |
| 分层记忆 | Shallow/Working/Deep/Meta 的数据结构与策略 | MemoryConfig | [memory_layers.md](memory_layers.md) |
| 排序器 | 时空综合评分 + 预算内选择 | alpha/beta/gamma/lambda_decay | [ranker.md](ranker.md) |
| 预算控制 | 预算计算、超限检查、成本估算 | BudgetController(model_name, system_prompt_ratio) | [budget.md](budget.md) |
| 向量库 | numpy/sqlite/qdrant 向量后端 | deep_vector_store_* + qdrant_* | [vector_store.md](vector_store.md) |
| 压缩 | token 估算与可解释压缩 | enable_compression + compression_* | [compression.md](compression.md) |
| 时间模型 | 时间范围解析与匹配分 | deep_enable_time_filter（Deep） | [temporal_model.md](temporal_model.md) |
| CLI | 命令行演示与运维入口 | config.json / memory_config | [cli.md](cli.md) |
