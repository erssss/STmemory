# 可视化演示服务（visual_demo）

入口源码：[visual_demo/server.py](../../visual_demo/server.py)  
配套说明：[visual_demo/README.md](../../visual_demo/README.md)

## 1. 服务定位

- 提供一个无需前端构建工具的交互式 Web UI，用于观测：
  - 分层记忆：检索/排序/上下文构建/写回
  - 向量检索：Deep 向量搜索结果与向量库统计
  - 基准产物：LOCOMO run 目录加载与 smoke run 日志推送

## 2. 启动方式

### 2.1 Mock/离线模式（推荐）

```bash
python -m visual_demo.server --host 127.0.0.1 --port 8765
```

访问：`http://127.0.0.1:8765/`

### 2.2 使用真实 LLM

```bash
python -m visual_demo.server --use-llm \
  --llm-base-url https://api.openai.com/v1 \
  --llm-model gpt-4o-mini
```

API key：
- 优先读取 `--llm-api-key`
- 或从环境变量 `LLM_API_KEY/OPENAI_API_KEY` 读取

## 3. HTTP API 与 WebSocket

完整端点与请求/响应结构见：[docs/api/http.md](../api/http.md)。

WebSocket（`/ws/locomo`）消息约定（摘要）：
- client -> server：
  - `{ "action": "start", "max_workers": 4 }`
  - `{ "action": "stop" }`
  - `{ "action": "load", "dir": "/abs/path/to/run" }`
- server -> client：
  - `{ "type": "ack", "action": "start", "dir": "..." }`
  - `{ "type": "ack", "action": "stop" }`
  - `{ "type": "run_loaded", "run": {...} }`
  - `{ "type": "error", "message": "..." }`

## 4. 常见问题

- 页面打不开：检查端口占用，或用 `--port` 更换端口。
- 向量检索为空：确认 Deep 启用了向量索引，且 encoder 已注入（首次运行可能需要写入一些 Deep 条目）。
- smoke run 无法启动：检查本机 Python 依赖是否齐全，以及 benchmark/locomo 相关依赖是否已安装。
