# 配置与调参指南（MemoryConfig/服务/基准）

本项目“实际生效”的运行配置核心是 `MemoryConfig`（dataclass），由各入口在运行时构造并注入插件/服务。

## 1. 配置来源与优先级

### 1.1 权威配置：MemoryConfig（生效）

以下入口会读取 JSON 并构造 `MemoryConfig(**cfg["memory_config"])`：
- CLI： [cli.py](../cli.py#L46-L53)
- LOCOMO server： [benchmark/locomo/server.py](../benchmark/locomo/server.py#L392-L401)
- 可视化 demo：直接构造（并可通过参数/环境变量注入 LLM）

示例模板：
- [config.memory_config.example.json](../config.memory_config.example.json)
- [memory_config.full.example.json](memory_config.full.example.json)

### 1.2 示例/历史配置：config.json（不保证生效）

根目录的 [config.json](../config.json) 结构与当前主链路读取方式不同，更多用于文档/示例表达。建议把需要长期维护的配置统一迁移到 `memory_config` 结构中。

## 2. MemoryConfig 字段分组（速查）

详细字段说明请以模块文档为准：
- 分层记忆与 Deep 检索：`docs/modules/memory_layers.md`
- 排序器：`docs/modules/ranker.md`
- 向量库：`docs/modules/vector_store.md`
- 压缩：`docs/modules/compression.md`

这里提供“调参入口级”速查与建议。

### 2.1 容量与 TTL（低延迟优先）

- `max_shallow_entries`：增大可提升“最近对话召回”，但会增加排序候选数与内存占用
- `max_working_entries`：增大会提升关键词召回覆盖，但会增加 O(n log n) 的检索成本
- `shallow_ttl/working_ttl`：缩短可降低噪声与内存占用，代价是跨轮可用性下降

### 2.2 排序权重（质量/近因/层级）

- `alpha_similarity`：语义相关性权重（embedding 质量好时优先加大）
- `beta_time` + `lambda_decay`：近因偏好（对长会话建议降低 lambda）
- `gamma_layer`：层级偏好（Deep 噪声高时可降低）

### 2.3 Deep 检索（召回/速度/可解释）

推荐先从保守配置开始：
- `deep_enable_vector_index=true`
- `deep_vector_candidates=50`
- `deep_enable_time_filter=true`
- `deep_expand_temporal_neighbors=1`
- `deep_time_weight=0.25`，`deep_lexical_weight=0.25`

若追求速度：
- 减小 `deep_vector_candidates`
- 减小 `deep_time_candidates_limit`
- 关闭 `deep_expand_temporal_neighbors`

若追求时间指令遵循（例如“昨天我们说过…”）：
- 增大 `deep_time_weight`
- 适当增大 `deep_time_candidates_limit`

### 2.4 压缩（预算/信息保真）

- `enable_compression=true` 通常建议开启（能显著降低 context token）
- `compressed_max_chars` 与 `compression_target_ratio` 控制压缩强度；压得过狠会损失可读性与事实细节
- `deep_return_compressed=true` 会在构造 context 时优先使用压缩后的 query/response

### 2.5 向量库后端（扩展性）

- 开发/小规模：`deep_vector_store_provider="numpy"`
- 单机持久化：`"sqlite"`
- 生产/大规模：`"qdrant"`（并配置 `qdrant_url/qdrant_collection/...`）

## 3. LLM 配置（remote/local）

### 3.1 local（Ollama/OpenAI-compatible，默认）

通过 MemoryConfig：
- `llm_provider="ollama"`（或 `local`/`local_llm`）
- `local_llm_base_url`（默认 `http://localhost:11434/v1`）
- `local_llm_model`
- `local_llm_check=true`（首次调用前探测 `/v1/models`）

### 3.2 remote（可选）

通过 MemoryConfig：
- `llm_provider="remote"`
- `llm_base_url`（例如 `https://api.openai.com/v1` 或私有网关）
- `llm_model`

通过环境变量：
- `LLM_API_KEY` 或 `OPENAI_API_KEY`

## 4. LOCOMO 集成跑配置

LOCOMO 集成跑入口：[benchmark/locomo/run.py](../benchmark/locomo/run.py)。

配置样例：
- [benchmark/locomo/config.example.json](../benchmark/locomo/config.example.json)

说明：
- `run.py --config` 支持同时承载 `memory_config`（传给 server）以及阈值/并发等跑分参数。
