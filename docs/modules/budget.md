# budget：Token 预算控制（参数与算法原理）

对应源码：[budget.py](../../budget.py)。

## 1. 模块职责

- 将模型的 token 上限（max_tokens/context_window）转换为“本次可用预算”。
- 在运行时记录 token 使用统计与粗略成本估计。
- 为上层提供预算检查（是否超限）与简单的分配优化建议。

## 2. 对外接口（输入/输出）

### 2.1 主要类

- `BudgetController(model_name: str = "openclaw-medium", system_prompt_ratio: float = 0.1)`

### 2.2 关键方法

| 方法 | 输入 | 输出 | 说明 |
|---|---|---|---|
| calculate_memory_budget | query, response_length_estimate | int | 计算“记忆检索预算” |
| estimate_response_length | query | int | 启发式估计响应 token |
| check_budget_constraints | memory_tokens, response_tokens | Dict[str,Any] | 预算检查结果 |
| update_usage_stats | tokens_used, memory_tokens | - | 更新统计 |
| get_cost_estimate | tokens | float | 成本估算（按预设单价） |

在上层使用位置：
- [plugin.py](../../plugin.py#L384-L389) 计算预算并分配给各层检索

## 3. 参数说明

### 3.1 初始化参数

| 参数 | 类型 | 默认值 | 取值范围/约束 | 影响 |
|---|---|---:|---|---|
| model_name | str | openclaw-medium | 任意字符串 | 选择预置模型配置，否则走默认配置 |
| system_prompt_ratio | float | 0.1 | [0,1) | 为 system prompt 预留比例 |

### 3.2 预置模型配置（MODEL_CONFIGS）

`MODEL_CONFIGS` 维护每个模型的：
- `max_tokens`：硬上限
- `context_window`：上下文窗口（当前实现与 max_tokens 等同使用）
- `cost_per_1k_tokens`：粗略成本
- `optimal_budget_ratio`：建议使用比例（预算上限）

详见：[budget.py](../../budget.py#L19-L70)。

## 4. 核心算法原理

### 4.1 记忆预算计算

记：
- `T = max_tokens`
- `r = optimal_budget_ratio`
- `p = system_prompt_ratio`
- `R = response_length_estimate`（可为空）

则：
- `base_budget = floor(T * r)`
- `system_prompt_tokens = floor(T * p)`
- `response_buffer = floor(1.2 * R)`，若 R 为空则 `floor(0.3 * base_budget)`
- `memory_budget = base_budget - system_prompt_tokens - response_buffer`
- 下限保护：`max(100, memory_budget)`

对应实现：[calculate_memory_budget](../../budget.py#L105-L134)。

### 4.2 响应长度估计（启发式）

根据 query 中的关键词类别（list/explain/what...）选择倍率并截断到上限（200~500），见 [estimate_response_length](../../budget.py#L151-L178)。

## 5. 复杂度分析

- 预算计算与检查均为 O(1)。
- 统计更新为 O(1)。

## 6. 适用场景与注意事项

- 该模块不做精确 tokenizer 计数，依赖上层的近似 token 估算（例如 [compression.py](../../compression.py#L6-L13)），因此预算是“工程近似”而非严格上限保证。
- 若接入不同模型且 token 上限不同，应在 `MODEL_CONFIGS` 中补齐配置，以避免默认 4096 导致预算偏小/偏大。
