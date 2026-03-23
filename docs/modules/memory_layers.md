# memory_layers：分层记忆模块（参数与算法原理）

对应源码：[memory_layers.py](../../memory_layers.py)。

## 1. 模块职责

- 定义统一的记忆条目结构 `MemoryEntry` 与配置 `MemoryConfig`。
- 实现四层记忆：
  - Shallow：最近性缓存（LRU + TTL）
  - Working：摘要化/关键词检索（TTL）
  - Deep：SQLite 持久化 +（可选）向量检索 + 时间过滤/邻居扩展 + 轻量实体索引
  - Meta：访问日志/层级转移统计

## 2. 对外接口（输入/输出）

### 2.1 数据结构：MemoryEntry

| 字段 | 类型 | 含义 |
|---|---|---|
| id | str | 唯一 ID |
| query | str | 写入内容（可为用户 query 或一段记忆文本） |
| response | str | 关联响应（允许为空字符串） |
| timestamp | datetime | 时间戳 |
| layer | str | 所属层级：`shallow/working/deep/meta` |
| token_count | int | 近似 token 计数（用于预算） |
| embedding | Optional[np.ndarray] | 向量（可为空） |
| metadata | Optional[dict] | 附加字段（如摘要、压缩文本、原始 token 等） |

### 2.2 统一接口：MemoryLayer（抽象基类）

| 方法 | 输入 | 输出 | 语义 |
|---|---|---|---|
| add | entry: MemoryEntry | bool | 写入一条记忆 |
| retrieve | query: str, budget: int | List[MemoryEntry] | 在 token 预算内检索候选 |
| decay | now: datetime | int | 衰减/清理并返回删除条数 |
| get_stats | - | Dict[str,Any] | 统计信息 |

## 3. 配置参数（MemoryConfig）分组说明

`MemoryConfig` 是本项目“权威生效配置源”，由上层（如 [plugin.py](../../plugin.py) 或各服务）构造并传入。核心字段定义见 [memory_layers.py](../../memory_layers.py#L30-L98)。

### 3.1 通用容量/TTL

| 参数 | 类型 | 默认值 | 取值范围/约束 | 依赖/影响 |
|---|---|---:|---|---|
| shallow_ttl | int | 300 | >0（秒） | Shallow 的过期清理阈值 |
| working_ttl | int | 1800 | >0（秒） | Working 的过期清理阈值 |
| max_shallow_entries | int | 100 | ≥1 | Shallow LRU 容量上限（条数） |
| max_working_entries | int | 50 | ≥1 | Working 容量上限（条数） |

备注：`shallow_capacity/working_capacity` 在当前实现中未作为硬约束参与淘汰（以 `max_*_entries` 为准）。

### 3.2 摘要（Working）

| 参数 | 类型 | 默认值 | 取值范围/约束 | 依赖/影响 |
|---|---|---:|---|---|
| use_llm_summary | bool | False | - | True 时会尝试用外部摘要模型，否则走本地截断摘要 |
| summary_llm_model | str | MiniMax-M2.7 | 非空 | 仅 `use_llm_summary=True` 时使用 |
| summary_llm_max_tokens | int | 256 | ≥1 | 仅 LLM 摘要路径使用 |
| summary_max_chars | int | 200 | ≥20 | 摘要最大字符数（两条路径都生效） |

外部依赖：LLM 摘要路径会尝试导入 `anthropic` 并从 `ANTHROPIC_API_KEY` 或 `MINIMAX_API_KEY` 取 key。

### 3.3 压缩（Deep）

| 参数 | 类型 | 默认值 | 取值范围/约束 | 依赖/影响 |
|---|---|---:|---|---|
| enable_compression | bool | True | - | True 时写入 Deep 会生成 `compressed_query/response` |
| compressed_max_chars | int | 180 | ≥40 | 压缩目标上限（字符数） |
| compression_min_ratio | float | 0.5 | (0,1] | 压缩下限比例（过短会回退裁剪） |
| compression_target_ratio | float | 0.5 | (0,1] | 目标 token 比例（会迭代收紧 max_chars） |
| compression_min_chars | int | 60 | ≥40 | 迭代收紧下限 |
| deep_return_compressed | bool | True | - | 上层构造 context 时优先使用压缩文本 |

### 3.4 Deep 检索（向量/时间/词面）

| 参数 | 类型 | 默认值 | 取值范围/约束 | 依赖/影响 |
|---|---|---:|---|---|
| deep_enable_vector_index | bool | True | - | True 且 encoder 可用时启用向量检索候选 |
| deep_vector_candidates | int | 50 | ≥1 | 向量检索候选数 |
| deep_enable_time_filter | bool | True | - | True 时解析 query 的时间范围并加入候选 |
| deep_time_candidates_limit | int | 300 | ≥1 | 时间窗口候选扫描上限 |
| deep_expand_temporal_neighbors | int | 1 | ≥0 | 对候选在时间序列上做邻居扩展（hop 数） |
| deep_time_weight | float | 0.25 | [0,1] 且与 `deep_lexical_weight` 之和 ≤1 | 深检索重排时的时间匹配权重 |
| deep_lexical_weight | float | 0.25 | [0,1] 且与 `deep_time_weight` 之和 ≤1 | 深检索重排时的词面重合权重 |
| deep_vector_rerank_prefetch | int | 50 | ≥1 | 仅向量库 provider=Qdrant 且“查询距离≠索引距离”时，用于重排的预取候选数 |

### 3.5 向量库（VectorStore）

| 参数 | 类型 | 默认值 | 取值范围/约束 | 依赖/影响 |
|---|---|---:|---|---|
| deep_vector_store_provider | str | numpy | `numpy/sqlite/qdrant` | Deep 向量索引后端实现 |
| deep_vector_store_table | str | memories | - | provider=sqlite 时用作表名 |
| deep_vector_distance | str | cosine | `cosine/dot/euclid` | provider=qdrant 时作为索引距离；也会影响查询 |
| deep_store_embeddings_in_sqlite | bool | True | - | True 时 embedding 也写入 SQLite（用于恢复/调试） |

Qdrant 相关字段详见 [vector-db-qdrant.md](../vector-db-qdrant.md)。

### 3.6 Deep 优化器（可选）

| 参数 | 类型 | 默认值 | 取值范围/约束 | 依赖/影响 |
|---|---|---:|---|---|
| deep_optimizer_enable | bool | False | - | True 时写入 Deep 后可能触发聚类/去重/压缩优化 |
| deep_optimizer_similarity_threshold | float | 0.88 | (0,1] | 聚类相似度阈值 |
| deep_optimizer_min_cluster_size | int | 3 | ≥2 | 最小簇大小 |
| deep_optimizer_max_clusters | int | 5 | ≥1 | 最大簇数 |
| deep_optimizer_delete_sources | bool | True | - | True 时会删除被合并的源条目 |

## 4. 核心算法原理（含伪代码/公式）

### 4.1 Shallow：LRU + TTL

- 写入：超过容量/粗略 token 上限时，弹出最旧条目（OrderedDict）。
- 检索：从最新往回取，直到预算用尽，再恢复为时间正序。

伪代码：

```text
add(entry):
  if entry.id exists: remove old, update token sum
  while too_many_entries or token_sum_too_large:
    pop_oldest()
  insert(entry)

retrieve(query, budget):
  for entry in reverse(time_order):
    if tokens + entry.tokens <= budget: take
    else break
  return taken_in_time_order
```

复杂度：
- add：摊还 O(1)
- retrieve：O(k)（k 为实际扫描/返回条目数，最坏 O(n)）
- decay：O(n)

适用场景：
- 需要极低延迟的“最近对话缓存”

### 4.2 Working：摘要写入 + 关键词匹配检索

- 摘要：默认走本地截断；可选走外部 LLM 摘要。
- 检索：按 query 分词后，计算命中比例 `match_count / |query_words|`，在预算内选取分数>0 的条目。

伪代码：

```text
retrieve(query, budget):
  query_words = set(query.lower().split())
  for entry in memories:
    score = count(word in entry_text for word in query_words) / |query_words|
  sort desc by score
  take while score>0 and tokens within budget
```

复杂度：
- retrieve：O(n * |Q| + n log n)

适用场景：
- 需要“解释性强、依赖少”的中期记忆筛选

### 4.3 Deep：候选生成（实体/向量/时间）+ 重排 + 预算内截断

Deep 的 `retrieve(query, budget)`（见 [memory_layers.py](../../memory_layers.py#L760-L859)）由三段组成：

1) 候选生成
- 实体候选：从 `_extract_entities(query)` 得到实体集合，在 `knowledge_graph[entity]` 中取最近 200 个 id，给予基础分 `0.2`
- 向量候选：若启用向量索引且 encoder 可用，则对 query 编码并向量检索 `deep_vector_candidates`，把命中分数作为候选基础分
- 时间候选：若解析出时间范围，则在时间排序列表中按窗口扫描（上限 `deep_time_candidates_limit`），给予基础分 `0.15`

2) 邻居扩展
- 对每个候选，在时间序列上左右扩展 `deep_expand_temporal_neighbors` hop，并把分数按 `0.6^hop` 衰减注入邻居。

3) 重排（融合时间匹配与词面重合）

记：
- `base`：候选基础分（来自实体/向量/时间）
- `t_bonus`：时间匹配分（[temporal_model.py](../../temporal_model.py)）
- `lex`：词面重合分（正则抽取 term 后的归一化 overlap）
- `w_time = deep_time_weight`，`w_lex = deep_lexical_weight`，`w_base = 1 - w_time - w_lex`

则重排分数：

\[
score = w_{base}\cdot base + w_{time}\cdot t\_bonus + w_{lex}\cdot lex
\]

最后按 `score` 降序、时间戳次序排序，并在 `budget` 内贪心装入返回结果。

复杂度（记 n 为 Deep 条目数，c 为候选数）：
- 实体候选：O(|E| * 200)（|E| 为抽取实体数）
- 向量检索：取决于 provider（numpy 近似 O(n·d)，Qdrant 为 ANN 近似亚线性）
- 时间候选：O(min(n, deep_time_candidates_limit))
- 重排：O(c log c)
- 预算截断：O(c)

适用场景：
- 需要持久化、可跨会话累计的长期记忆；并希望同时利用语义相似、时间约束与可解释的词面信号。

### 4.4 Meta：层级转移统计

Meta 主要记录访问日志与层级转移次数，用于后续更新排序器的转移矩阵（见 [ranker.py](../../ranker.py#L292-L297)）。

## 5. 异常与边界条件

- Deep 向量检索仅在 encoder 注入后生效：插件初始化时会尝试 `deep.set_encoder(ranker._encode_texts)`（见 [plugin.py](../../plugin.py#L269-L272)）。
- Working LLM 摘要路径依赖第三方库与 API key；失败会自动回退到本地截断摘要。
- `deep_time_weight + deep_lexical_weight > 1` 时，`w_base` 会被截断为非负（实现中对每个权重做了 [0,1] 裁剪，并用 `1-w_time-w_lex` 计算）。
