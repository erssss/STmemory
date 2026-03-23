# temporal_model：时间范围解析与时间匹配（参数与原理）

对应源码：[temporal_model.py](../../temporal_model.py)。

## 1. 模块职责

- 从 query 中解析出时间范围 `TimeRange(start,end)`。
- 计算某条记忆时间戳与时间范围的匹配分数（用于排序与 Deep 检索重排）。

被使用位置：
- 排序器： [ranker.py](../../ranker.py#L163-L168)
- Deep 检索： [memory_layers.py](../../memory_layers.py#L765-L846)

## 2. 对外接口（输入/输出）

| 方法 | 输入 | 输出 | 说明 |
|---|---|---|---|
| parse_time_range_from_query | query: str, now?: datetime | TimeRange | 解析“最近/今天/昨天/具体日期/年份”等 |
| temporal_match_score | entry_time: datetime, time_range: TimeRange, now?: datetime | float | [0,1] 的匹配分 |

## 3. 时间解析规则（概览）

### 3.1 中文相对时间

- 刚才/刚刚：`[now-10min, now]`
- 最近/近期：`[now-7days, now]`
- 今天：当天 00:00~23:59
- 昨天/前天：对应日期整天
- 上周：`[now-7days, now]`
- 上个月：`[now-30days, now]`
- 去年：上一年整年

### 3.2 绝对日期

- `YYYY-MM-DD` 或 `YYYY/MM/DD`
- `Month Day, Year` / `Day Month, Year`
- `YYYY` + 伴随“年/in/during”等提示词：视作该年整年

实现见：[parse_time_range_from_query](../../temporal_model.py#L21-L117)。

## 4. 匹配分数（temporal_match_score）

如果 `entry_time` 不在 `[start,end]` 内，匹配分为 0。

否则将范围映射到一个“以中点为峰值”的三角形分布：

- `mid = (start + end)/2`
- `span = end - start`
- `dist = |entry_time - mid|`
- 分数：

\[
score = \max\left(0, 1 - \frac{dist}{span/2}\right)
\]

实现见：[temporal_match_score](../../temporal_model.py#L120-L134)。

## 5. 复杂度分析

- 时间解析：O(|query|)（正则匹配与字符串扫描）
- 匹配分：O(1)

## 6. 适用场景与局限

- 适合“粗粒度时间约束”与“时间偏好”信号融合（排序/重排）。
- 当前规则以启发式为主，不处理复杂自然语言时间表达（如“上上周三下午”）；若需要更强时间理解，可替换为专用时间解析器并保持 `TimeRange` 接口不变。
