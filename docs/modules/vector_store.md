# vector_store：向量存储抽象与实现（参数与原理）

对应源码目录：[vector_store/](../../vector_store)。

## 1. 模块职责

- 抽象出统一接口 `VectorStoreBase`：`upsert/search/delete`（见 [base.py](../../vector_store/base.py)）。
- 提供三种 provider：
  - `numpy`：内存向量库（开发/小规模）
  - `sqlite`：把向量存于 SQLite（与 DeepMemory 同库）
  - `qdrant`：外部向量数据库（HNSW/可观测性/可扩展）

DeepMemory 的向量检索通过 `VectorStoreFactory.create()` 注入（见 [memory_layers.py](../../memory_layers.py#L392-L412)）。

## 2. 对外接口（输入/输出）

### 2.1 VectorStoreBase

| 方法 | 输入 | 输出 | 说明 |
|---|---|---|---|
| upsert | ids: List[str], vectors: List[np.ndarray], payloads?: List[dict] | None | 批量写入/更新 |
| search | query_vector: np.ndarray, limit: int, filters?: dict | List[OutputData] | 相似度检索 |
| delete | ids: List[str] | None | 删除 |

`search` 输出 `OutputData`：
- `id`：与 DeepMemory 条目 id 对应
- `score`：相似度分数（不同距离下可能是 dot/cos 或 -l2）
- `payload`：可选字段（Qdrant 会回传 payload）

## 3. 配置参数（MemoryConfig）

DeepMemory 创建向量库时使用这些字段（定义见 [MemoryConfig](../../memory_layers.py#L72-L92)）：

| 参数 | 类型 | 默认值 | 取值范围/约束 | 影响 |
|---|---|---:|---|---|
| deep_vector_store_provider | str | numpy | `numpy/sqlite/qdrant` | provider 选择 |
| deep_vector_store_table | str | memories | - | provider=sqlite 的表名 |
| deep_vector_distance | str | cosine | `cosine/dot/euclid` | provider=qdrant 的索引距离；也影响查询重排 |
| deep_vector_rerank_prefetch | int | 50 | ≥1 | Qdrant 查询距离与索引距离不一致时的预取候选 |
| deep_store_embeddings_in_sqlite | bool | True | - | 是否同时把 embedding 写入 SQLite |

Qdrant 专项字段（provider=qdrant）：
- `qdrant_location`（本地模式，可选）
- `qdrant_url/qdrant_api_key/qdrant_collection/qdrant_timeout_s/qdrant_prefer_grpc`
- HNSW/优化参数：`qdrant_hnsw_m/qdrant_hnsw_ef_construct/qdrant_full_scan_threshold/qdrant_indexing_threshold/qdrant_on_disk_payload`
- 查询参数：`qdrant_search_hnsw_ef/qdrant_search_exact`

详见 [vector-db-qdrant.md](../vector-db-qdrant.md) 与 [factory.py](../../vector_store/factory.py#L18-L45)。

## 4. 算法与实现要点

### 4.1 numpy provider

- 存在于进程内内存结构中；适合 demo/小规模。
- 查询通常为全量扫描 + 相似度计算（复杂度近似 O(n·d)）。

### 4.2 sqlite provider

- 用 SQLite 表保存向量；适合单机持久化但不适合大规模 ANN。
- 查询性能与实现细节受 SQLite schema 与索引影响（需要结合 sqlite_store 实现评估）。

### 4.3 qdrant provider（推荐生产/大规模）

实现见 [qdrant_store.py](../../vector_store/qdrant_store.py)。

关键机制：
- **Collection 自建**：首次 upsert 时根据首条向量维度创建 collection，并校验后续维度一致（[qdrant_store.py](../../vector_store/qdrant_store.py#L123-L162)）。
- **距离一致性与重排**：
  - 若查询想要的距离 `wanted_distance` 与索引距离不一致，会使用 `with_vectors=True` 预取更多候选，再在客户端用 cos/dot/-l2 重排并截断 top-k（[qdrant_store.py](../../vector_store/qdrant_store.py#L226-L273)）。
- **可观测性**：统计 upsert/search/delete 的调用次数、耗时与 last_error（[VectorStoreStats](../../vector_store/qdrant_store.py#L18-L37)），上层通过 `get_vector_store_stats()` 暴露。

## 5. 复杂度分析（概要）

记 n 为向量条目数，d 为向量维度：
- numpy：search 近似 O(n·d)，空间 O(n·d)
- qdrant：search 近似 O(log n)~O(n^ρ)（ANN 视索引/ef 等而定），空间 O(n·d) + 索引开销

## 6. 适用场景

- 单机 demo/快速验证：`numpy`
- 单机持久化、小规模：`sqlite`
- 生产/并发/大规模：`qdrant`
