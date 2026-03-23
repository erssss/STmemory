# STmemory 可视化演示（交互式 Web UI）

该演示用于直观展示当前仓库已实现的核心能力在真实运行时的效果，包括：

- 分层记忆工作流：检索 → 预算控制 → 排序选择 → Context 构建 → 生成/响应 → 写回多层记忆 → 指标统计
- 数据处理过程：每轮输入/输出、Context 预览、相关记忆命中数量、预算检查结果
- 性能与体验指标：延迟、Token 使用与节省趋势
- 功能完整性验证：LOCOMO benchmark 结果加载与阈值判定（PASS/FAIL）、样本级验证表格

## 启动

在仓库根目录执行：

```bash
python -m visual_demo.server --host 127.0.0.1 --port 8765
```

浏览器打开：

```text
http://127.0.0.1:8765/
```

## 使用说明

### 记忆工作流

- 点击“下一步”：按内置多轮对话脚本逐步执行，并将每轮的指标以图表方式动态更新
- 点击“自动播放”：连续执行多轮对话，观察记忆层随时间的积累与命中情况
- 在输入框提问并“发送”：实时触发一次 `process_query`，用于演示用户交互响应与检索/写回的联动效果

### LOCOMO 基准

- “刷新列表”：扫描本机可用的 LOCOMO 运行输出目录（默认扫描 `benchmark/results/locomo` 与 `/tmp/stmemory_locomo*`）
- “加载”：读取并展示 `report.json / evaluation_metrics.csv / evaluation.txt / token1.json / token2.json` 等产物
- “启动 Smoke Run”：在后台执行 `python -m benchmark.locomo.run --use-mock-openai ...`，并通过 WebSocket 实时推送日志与最终结果

## 依赖

演示服务端使用 `aiohttp`（仓库现有依赖），前端图表使用 CDN 方式加载 `Plotly.js` 与 `Mermaid`，无需额外安装前端构建工具。

