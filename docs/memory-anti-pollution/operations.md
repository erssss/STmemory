# 运维与监控方案（防记忆污染增强）

## 维护任务调度

建议由上层服务/任务调度系统定期触发：

- 高频：每 1–6 小时触发一次 `Memory.maintain(user_id=..., dry_run=...)`（按活跃用户分批）
- 低频：每日触发一次冲突组巡检（对 `_conflict_state=conflicting` 的组调用 `resolve_conflicts` 生成建议）

默认建议先 `dry_run=true` 运行一段时间，观察统计与抽样结果后再切换到 `dry_run=false`。

## 日志与审计

已使用现有审计/遥测基础设施：

- 审计：`audit.log`（关键行为：search/add/intelligent_add/maintenance）
- 遥测：Telemetry 事件上报（如启用）

建议补充的关注点：

- 统计每次维护任务的 `scanned/deleted/archived/errors`
- 统计冲突组的新增速率与 resolved 速率
- 统计疑似 prompt 注入命中（suspect）数量

## 关键告警（建议）

- `maintenance.errors > 0`：维护任务执行异常
- `pollution.detected` 短期激增：可能存在输入污染源或 scope 丢失导致候选集越界
- `conflict_group` 长期堆积：冲突未被治理（需要合并/修正策略介入）

## 容量与归档策略

- 主库只保留活跃记忆：通过 TTL 归档或删除控制增长
- 归档保留策略：
  - 优先归档（`metadata.archived=true`），保留可追溯性
  - 在稳定运行后，可对长期归档再做冷存储导出（JSON/CSV）并从主库清理

