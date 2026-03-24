# 性能评估报告（模板与复现方法）

## 评估目标

- 功能性：启用隔离/窗口/污染检测后，跨 scope 召回显著下降，冲突能被标记并可被合并修正
- 性能：search/add 延迟增量可控；时间窗口过滤能降低 SQLite 扫描量

## 评估维度

### 在线路径

- `Memory.add`
  - simple add：embedding + insert（+topic/窗口/隔离 metadata 注入）
  - intelligent add：fact 抽取 + 候选检索 + 决策 + 动作执行（+污染检测）
- `Memory.search`
  - embedding + search（+session/topic/time_window filters）
  - 结果加工与 access_count 更新

### 离线路径

- `Memory.optimize`：必须指定 user_id，避免跨域压缩/删除
- `Memory.maintain`：批量 list + TTL 判定 + archive/delete

## 推荐实验方法

1) 基线（关闭增强开关）与增强（打开 isolation/topic/time_window/pollution/maintenance）对比
2) 在 LOCOMO 数据集实验脚本上重复运行同一参数组合，比较：
   - 召回质量指标（如有）
   - 跨 user/run 的错误召回率（污染指标）
3) 采集 telemetry/audit
   - 每次 search 的 results_count 与过滤后数量
   - intelligent add 的候选数量与冲突命中数量
   - maintenance 的 scanned/archived/deleted/errors

## 结论记录（待填）

- search 延迟（P50/P95）：基线 __ / 增强 __
- add 延迟（P50/P95）：基线 __ / 增强 __
- 跨 scope 召回率：基线 __ / 增强 __
- 冲突检测命中与抽样准确率：__

