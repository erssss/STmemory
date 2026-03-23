# 使用 Qdrant 的向量数据库部署与运维

本项目的 DeepMemory 向量检索通过 `VectorStore` 抽象接入向量数据库。选择 `qdrant` provider 时：

- 记忆条目（query/response/metadata/时间戳）仍以 SQLite 作为持久化来源（兼容原有逻辑与时间过滤/邻居扩展）。
- 向量索引与相似度检索由 Qdrant 负责（支持 HNSW、Cosine/Dot/Euclid 等距离）。

## 1. 安装

```bash
pip install -r requirements.txt
```

Qdrant 运行方式二选一：

- Docker 运行 Qdrant Server（推荐生产/多人并发）。
- 本地模式（`qdrant-client` 的 `location=":memory:"` 或 `path=...`）用于开发/测试。

## 2. 启动 Qdrant（Docker）

```bash
docker run --rm -p 6333:6333 -p 6334:6334 qdrant/qdrant:latest
```

健康检查：

```bash
curl -s http://localhost:6333/ | cat
```

## 3. 配置（MemoryConfig）

将以下字段放入你的配置文件 `memory_config` 中（可参考 `config.memory_config.example.json`）：

- `deep_vector_store_provider`: 设为 `"qdrant"`
- `deep_vector_distance`: `"cosine" | "dot" | "euclid"`
- `qdrant_url`: 例如 `"http://localhost:6333"`
- `qdrant_collection`: 例如 `"stmemory_deep"`
- `qdrant_timeout_s`: REST 请求超时（秒）
- `qdrant_hnsw_m / qdrant_hnsw_ef_construct / qdrant_full_scan_threshold`: 索引构建参数
- `qdrant_search_hnsw_ef / qdrant_search_exact`: 查询参数
- `deep_vector_rerank_prefetch`: 当请求的距离算法与索引距离不一致时，用于重排的候选数
- `deep_vector_auto_reindex`: `true/false`，在 encoder 注入后补齐缺失向量
- `deep_vector_auto_reindex_max_entries`: 自动补齐的最大条目数
- `deep_store_embeddings_in_sqlite`: `true/false`，是否把 embedding 同时写入 SQLite（建议保持 `true` 以便快速恢复与调试）

说明：Qdrant collection 会在首次写入向量时自动创建，向量维度取第一条向量的长度。

## 4. 启动服务与调用示例

### 4.1 LOCOMO 兼容服务

```bash
python -m benchmark.locomo.server --config path/to/your_config.json --port 8000
```

向量搜索 API：

```bash
curl -s http://127.0.0.1:8000/vector_search \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"u1","query":"hello","top_k":5,"distance":"cosine"}' | cat
```

向量库监控（统计）：

```bash
curl -s 'http://127.0.0.1:8000/vector_store/stats?user_id=u1' | cat
```

### 4.2 可视化 Demo 服务

```bash
python -m visual_demo.server --port 8765
```

向量搜索 API：

```bash
curl -s http://127.0.0.1:8765/api/vector_search \
  -H 'Content-Type: application/json' \
  -d '{"query":"hello","top_k":5,"distance":"euclid"}' | cat
```

向量库监控（统计）：

```bash
curl -s http://127.0.0.1:8765/api/vector_store/stats | cat
```

## 5. 性能调优建议

- 优先把 `deep_vector_distance` 设置为与你主要检索方式一致的距离（通常为 `cosine`）。
- 通过 `qdrant_search_hnsw_ef` 调整召回/速度：越大召回越好但越慢。
- 大数据量时避免 `qdrant_search_exact=true`，除非需要精确结果。
- 如果你需要在同一个索引上支持不同距离算法的查询，本项目会通过 `deep_vector_rerank_prefetch` 拉取候选后做重排；可按场景增大该值提升重排质量。

## 6. 日志与监控

- Qdrant 后端日志使用 Python `logging`（logger 名：`stmemory.vector_store.qdrant`）。
- 通过上面的 `*/vector_store/stats` 接口可获得 upsert/search/delete 次数与累计耗时、最近错误等统计。

## 7. 数据迁移与恢复

- 从旧的 `numpy/sqlite` 向量后端迁移到 `qdrant`：只需切换配置并确保 Deep 层能拿到 encoder（默认插件初始化会注入）。
- 若 SQLite 中存在历史 embedding，会在 Deep 层启动加载时自动写入向量库；若历史 embedding 缺失，会在 `set_encoder()` 之后按 `deep_vector_auto_reindex*` 配置补齐。

