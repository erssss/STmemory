# ranker：时空排序器（参数与算法原理）

对应源码：[ranker.py](../../ranker.py)。

## 1. 模块职责

- 将来自不同记忆层的候选条目统一评分（语义相似 + 时间近因 + 层级偏好）。
- 在 token 预算约束下选择一组条目，用于构建上下文（context）。

## 2. 对外接口（输入/输出）

### 2.1 主要类

- `SpatioTemporalRanker(config: MemoryConfig, model_name: str = "all-MiniLM-L6-v2")`

### 2.2 关键方法

| 方法 | 输入 | 输出 | 说明 |
|---|---|---|---|
| compute_spatiotemporal_score | query, entry, current_time, current_layer | ScoredMemory | 单条打分（含分解项） |
| rank_memories | query, memories, current_time, current_layer | List[ScoredMemory] | 批量打分并按 score 排序 |
| select_memories_within_budget | query, all_memories, budget, current_time | List[MemoryEntry] | 汇总所有层候选并在预算内挑选 |
| update_transition_matrix | meta_layer | - | 用 Meta 统计更新转移矩阵 |

## 3. 配置参数（MemoryConfig）

字段定义见 [MemoryConfig](../../memory_layers.py#L30-L98)。本模块主要使用：

| 参数 | 类型 | 默认值 | 取值范围/约束 | 依赖/影响 |
|---|---|---:|---|---|
| lambda_decay | float | 0.1 | ≥0 | 时间衰减系数 λ（越大越“偏近因”） |
| alpha_similarity | float | 0.6 | [0,1] | 语义相似权重 α |
| beta_time | float | 0.3 | [0,1] | 时间权重 β |
| gamma_layer | float | 0.1 | [0,1] | 层级权重 γ |
| embedding_device | Optional[str] | None | `cpu/cuda/...` | SentenceTransformer 设备选择 |

约束建议（调参指导）：
- 推荐让 `alpha_similarity + beta_time + gamma_layer = 1`，以便分值可解释且权重直观；当前实现不会强制归一化。
- `lambda_decay` 的“时间单位”为秒（见 [compute_time_decay](../../ranker.py#L106-L111)），因此对长对话/长会话建议适当降低。

## 4. 核心算法原理

### 4.1 语义相似度

默认路径：
- 使用 `SentenceTransformer` 生成 query 与 entry 文本的向量，并计算余弦相似度（[compute_semantic_similarity](../../ranker.py#L113-L136)）。
- entry 向量会缓存到 `MemoryEntry.embedding`，降低后续重复计算成本。

降级路径：
- 若模型加载失败，会回退到 `_HashEmbeddingModel`（[ranker.py](../../ranker.py#L29-L70)），本质是 token 哈希计数向量（可跑通但质量较弱）。

### 4.2 时间近因（指数衰减）

\[
w(t) = \exp(-\lambda \cdot \Delta t)
\]

其中 \(\Delta t\) 以秒计（[compute_time_decay](../../ranker.py#L106-L111)）。

如果 query 显式包含时间范围（由 [temporal_model.py](../../temporal_model.py) 解析），会把时间分数改写为：

\[
time\_score = w(t)\cdot (0.2 + 0.8\cdot temporal\_match)
\]

（见 [compute_spatiotemporal_score](../../ranker.py#L163-L168)）

### 4.3 层级偏好（转移矩阵）

初始化转移矩阵（[ranker.py](../../ranker.py#L81-L97)）：
- 自环：0.6
- 相邻层：0.3
- 其他：0.1

层级分数：
\[
layer\_score = P(current\_layer \rightarrow entry.layer)
\]

可通过 `MetaMemoryLayer.get_transition_matrix()` 的统计更新（[ranker.py](../../ranker.py#L292-L297)）。

### 4.4 综合评分函数

综合分：

\[
score = \alpha \cdot sim + \beta \cdot time + \gamma \cdot layer
\]

对应实现：[compute_spatiotemporal_score](../../ranker.py#L152-L186)。

### 4.5 预算内选择：贪心近似 Knapsack

`select_memories_within_budget()` 的选择步骤：
1) 合并各层候选为一个列表
2) 批量计算相似度/综合分（向量模型存在时会进行批量编码与矩阵余弦）
3) 按 score 降序排序
4) 依次加入，直到 token 超预算（[ranker.py](../../ranker.py#L279-L286)）
5) 最终按时间戳升序重排，保证上下文连贯（[ranker.py](../../ranker.py#L287-L289)）

伪代码：

```text
select(all_entries, budget):
  score each entry
  sort by score desc
  for entry in sorted:
    if used + entry.tokens <= budget:
      take(entry); used += entry.tokens
    else: break
  sort taken by timestamp asc
  return taken
```

该问题形式上接近 0/1 背包（最大化价值、受 token 成本约束）。当前实现选择“排序 + 贪心装入”：
- 优点：O(n log n) 可解释、延迟可控、工程简单
- 局限：不保证最优，尤其当 token 成本差异很大时，更合理的近似是按 `score/token_cost` 排序

## 5. 复杂度分析

记 n 为候选总数，d 为向量维度：
- 语义编码：O(n·d)（具体由模型推理决定）
- 余弦相似：O(n·d)（矩阵乘/向量点积）
- 排序：O(n log n)
- 预算装入：O(n)

空间复杂度：
- 若缓存 embedding：O(n·d)

## 6. 适用场景与调参建议

- `alpha_similarity` 适合在“主题相关性”强时增大；若 embedding 质量较差（离线/回退哈希），可适当降低。
- `beta_time` 与 `lambda_decay` 共同控制近因偏好；对长会话建议降低 `lambda_decay`，避免旧记忆全被压到 0。
- `gamma_layer` 与转移矩阵用于表达“层级偏好”；当 Deep 记忆很大但召回噪声高时，可降低 Deep 的转移概率或降低 `gamma_layer`。
