# HTTP API 参考（服务端点、参数、错误码）

本项目包含两套 HTTP 服务：
- 可视化演示服务：`visual_demo/server.py`（面向浏览器 UI）
- LOCOMO 兼容基准服务：`benchmark/locomo/server.py`（面向 benchmark 脚本）

## 1. 可视化演示服务（visual_demo）

入口：[visual_demo/server.py](../../visual_demo/server.py)  
默认：`http://127.0.0.1:8765`

### 1.1 GET /

- 描述：返回前端页面
- 响应：HTML

### 1.2 GET /api/demo/state

- 描述：获取当前演示引擎状态
- 响应 200：

```json
{ "status": "ok", "state": { "step": 0, "history": [], "stats": {} } }
```

### 1.3 POST /api/demo/reset

- 描述：重置演示状态
- 请求体：空 JSON `{}` 或任意 JSON
- 响应 200：同 `/api/demo/state`

### 1.4 POST /api/demo/step

- 描述：执行演示脚本的下一步（内部调用插件处理）
- 请求体：空 JSON `{}` 或任意 JSON
- 响应 200：包含本步执行结果（字段随 demo 引擎实现变化）

### 1.5 POST /api/chat

- 描述：对话接口（调用 `SpatioTemporalMemoryPlugin.process_query`）
- 请求体：

```json
{ "query": "你的问题" }
```

- 响应 200：为 `process_query` 的结果结构（见 [plugin.py](../../plugin.py#L506-L517)）
- 响应 400：`{ "error": "query required" }`

### 1.6 POST /api/vector_search

- 描述：对 Deep 向量库做调试检索
- 请求体：

```json
{ "query": "hello", "top_k": 10, "distance": "cosine" }
```

- 参数说明：
  - `top_k`：int，默认 10，范围 [1,100]
  - `distance`：`cosine|dot|euclid`
- 响应 200：

```json
{ "results": [ { "id": "...", "score": 0.12, "memory": "...", "timestamp": "...", "metadata": {} } ] }
```

- 响应 400：`{ "error": "query required" }`

### 1.7 GET /api/vector_store/stats

- 描述：返回向量库统计（provider、collection、计数/耗时/错误等）
- 响应 200：`plugin.get_vector_store_stats()` 输出

### 1.8 LOCOMO 产物浏览与 websocket

- GET /api/locomo/runs：列出本机可见的 LOCOMO run 目录
- GET /api/locomo/run?dir=...：读取某个 run 的产物并返回结构化数据
- GET /ws/locomo：WebSocket 通道（启动/停止 smoke run、推送日志、加载 run）

错误码：
- 400：缺少必填参数（如 `dir required`）

## 2. LOCOMO 兼容服务（benchmark/locomo/server）

入口：[benchmark/locomo/server.py](../../benchmark/locomo/server.py)  
默认：`http://127.0.0.1:8000`

### 2.1 GET /healthz

- 描述：健康检查
- 响应 200：

```json
{ "status": "ok" }
```

### 2.2 POST /memories

- 描述：写入记忆（LOCOMO 的 add 阶段使用）
- 请求体：

```json
{
  "user_id": "u1",
  "messages": [ { "content": "..." } ],
  "metadata": { "timestamp": "8:30 pm on 28 Feb, 2024" }
}
```

- 响应 200：

```json
{ "status": "success", "added": 10 }
```

- 错误码：
  - 400：`user_id is required` / `messages must be a list`

### 2.3 DELETE /memories?user_id=...

- 描述：删除某个 user 的全部记忆（按 user_id 清理插件实例）
- 响应 200：`{ "status": "success" }`
- 错误码：
  - 400：`user_id is required`

### 2.4 POST /search

- 描述：检索记忆（LOCOMO 的 search 阶段使用）
- 请求体：

```json
{ "user_id": "u1", "query": "question", "top_k": 10 }
```

- 响应 200：

```json
{
  "results": [
    { "memory": "...", "metadata": { "timestamp": "2024-02-28 20:30" }, "score": 0.42 }
  ],
  "relations": []
}
```

- 错误码：
  - 400：`user_id and query are required`

说明：该端点会额外受 `locomo_search_context_token_budget` 影响，可能在 token 预算内截断返回条目数（见 [server.py](../../benchmark/locomo/server.py#L164-L185)）。

### 2.5 POST /vector_search

- 描述：调用 Deep 的向量搜索（用于诊断/对比）
- 请求体：

```json
{ "user_id": "u1", "query": "hello", "top_k": 10, "distance": "cosine" }
```

- 响应 200：同 `plugin.vector_search()` 输出
- 错误码：
  - 400：`user_id and query are required`

### 2.6 GET /vector_store/stats?user_id=...

- 描述：返回该 user 对应插件的向量库统计
- 响应 200：同 `plugin.get_vector_store_stats()`
- 错误码：
  - 400：`user_id is required`

### 2.7 POST /v1/chat/completions（可选 mock OpenAI）

- 描述：当 server 以 `--enable-mock-openai` 启动时，提供 mock 的 OpenAI-compatible 端点（用于 LOCOMO judge/answer）
- 请求体：OpenAI chat/completions 兼容结构（至少包含 `messages`）
- 响应 200：OpenAI chat.completion 兼容响应（含 `usage`）
- 错误码：
  - 404：mock_openai disabled
