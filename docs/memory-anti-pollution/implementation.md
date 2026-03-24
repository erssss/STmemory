# 实施与迁移指南（防记忆污染增强）

## 配置启用

本增强方案默认保持兼容行为：在不改动现有调用代码的前提下，通过配置逐步打开隔离、时间窗口、污染检测与维护任务。

关键配置项（建议）：

```json
{
  "isolation": {
    "mode": "compat",
    "default_user_id": "user",
    "enforce_on_search": true,
    "enforce_on_intelligent_add": true
  },
  "topic_isolation": {
    "enabled": true,
    "provider": "llm",
    "mode": "soft",
    "confidence_threshold": 0.7
  },
  "time_window": {
    "enabled": true,
    "default_days": 30
  },
  "pollution_detection": {
    "enabled": true,
    "run_on_intelligent_add": true,
    "prompt_injection_scan": true,
    "mark_only": true
  },
  "maintenance": {
    "enabled": true,
    "allow_global": false,
    "dry_run_default": true,
    "archive_enabled": true,
    "ttl_days": { "working": 30, "short_term": 180, "long_term": 3650 }
  }
}
```

## 关键行为变更（兼容模式）

- `add/search`：当 `user_id` 为空时自动回退到 `isolation.default_user_id`，避免无 scope 的全库写入/检索带来的污染扩散。
- `optimize`：未传 `user_id` 会拒绝执行（除非 `maintenance.allow_global=true`）。
- `search`：在 SQLite 后端下默认排除 `metadata.archived=true` 的记忆（可通过显式 filters 覆盖）。

## 新增 API

- `Memory.detect_pollution(facts=[...], user_id=..., run_id=..., filters=...)`
- `Memory.resolve_conflicts(user_id=..., group_id=...)`
- `Memory.merge_conflicts(user_id=..., group_id=..., strategy="summarize")`
- `Memory.maintain(user_id=..., run_id=..., dry_run=...)`

## 数据字段约定（metadata）

系统会写入以下字段（均位于 `payload["metadata"]`）：

- `_session_id`：会话隔离键（默认映射 run_id）
- `_topic_id/_topic_path/_topic_confidence`：主题隔离键
- `_conflict_group_id/_conflict_state/_conflict_key/_conflict_score/_conflict_reason`：冲突标记
- `_consistency_state/_suspect_reason/_suspect_pattern`：污染/注入嫌疑标记
- `archived`：归档标记

时间戳字段位于 payload 顶层：

- `created_at_ts/updated_at_ts`：用于窗口过滤与维护任务判断

