# compression：记忆压缩与 token 估算（参数与原理）

对应源码：[compression.py](../../compression.py)。

## 1. 模块职责

- 提供轻量 `estimate_tokens(text)`：用于预算估算（英文词数 + 中文字符数 + 标点启发式）。
- 提供 `MemoryCompressor`：对长文本进行“可解释压缩”，生成压缩结果与压缩比。

该模块主要被 DeepMemory 使用（见 [memory_layers.py](../../memory_layers.py#L384-L390)）。

## 2. 对外接口（输入/输出）

### 2.1 estimate_tokens

| 输入 | 输出 | 说明 |
|---|---|---|
| text: str | int | 近似 token 数，非严格 tokenizer |

### 2.2 MemoryCompressor

初始化参数：

| 参数 | 类型 | 默认值 | 取值范围/约束 |
|---|---|---:|---|
| max_chars | int | 200 | ≥40 |
| min_ratio | float | 0.5 | (0,1] |
| target_ratio | float | 0.5 | (0,1] |
| min_chars | int | 60 | ≥40 |

主要方法：

| 方法 | 输入 | 输出 | 说明 |
|---|---|---|---|
| compress_text | text | CompressionResult | 在 `max_chars` 内压缩 |
| compress_to_target | text | CompressionResult | 迭代收紧 `max_chars` 逼近 `target_ratio` |

`CompressionResult`：
- `raw_text/compressed_text`
- `raw_tokens/compressed_tokens`
- `ratio = compressed_tokens / raw_tokens`

## 3. 核心算法原理

### 3.1 句子切分 + 句子打分 + 装包

1) 归一化空白；若长度已 ≤ `max_chars`，直接返回
2) 以中英文标点切分句子
3) 提取全局“实体集合”（启发式 token）
4) 对每个句子打分：
   - `ent_score`：句子实体与全局实体的交集数量
   - `digit_score`：包含数字加分
   - `date_score`：包含日期模式加分
   - `len_pen`：长度归一化（更长句子略加分，最多 1）
5) 按句子分数降序选择句子拼接，直到达到 `max_chars`
6) 若压缩太短（`len(comp) < len(raw)*min_ratio`），回退到“直接裁剪 + 省略号”

对应实现：[MemoryCompressor._compress_with_limit](../../compression.py#L70-L127)。

### 3.2 目标压缩比的自适应收紧

`compress_to_target()` 会在 `ratio > target_ratio` 且 `max_chars > min_chars` 时，将 `max_chars *= 0.8` 继续压缩，直到达到目标或无法进一步收紧（[compression.py](../../compression.py#L129-L142)）。

## 4. 复杂度分析

记 s 为句子数，L 为文本长度：
- 句子切分与特征抽取：O(L)
- 句子排序：O(s log s)
- 选择装包：O(s)

空间复杂度：O(s + L)

## 5. 适用场景与注意事项

- 适合“长文本记忆”在上下文预算下的快速压缩与可解释裁剪。
- 由于 `estimate_tokens` 为启发式，压缩比与真实 tokenizer 可能存在偏差；若需要严格预算，应在工程层引入模型 tokenizer 并替换估算函数。
