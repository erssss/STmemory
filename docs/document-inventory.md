# 文档盘点（STmemory，排除 powermem）

本文件用于记录仓库内现有技术文档的位置与定位，并给出本次文档重构后的推荐入口与迁移关系。

## 现有文档清单（排除 powermem）

| 路径 | 类型 | 覆盖内容 | 备注 |
|---|---|---|---|
| [README.md](../README.md) | 概览/使用 | 项目介绍、安装、快速开始、部分参数示例 | 含部分与当前实现不一致的历史描述 |
| [TECHNICAL_GUIDE.md](../TECHNICAL_GUIDE.md) | 原理/源码导读 | 模块边界、端到端链路、关键算法解释、可运行手册 | 内容较全面，但未按“模块参数/算法复杂度/API 参考”系统化拆分 |
| [docs/vector-db-qdrant.md](vector-db-qdrant.md) | 部署/运维 | Qdrant 接入、参数、调用示例、调优 | 可保留为向量库专项文档 |
| [visual_demo/README.md](../visual_demo/README.md) | Demo 说明 | Web UI 启动与交互、LOCOMO 产物加载与 smoke run | 可迁移为服务/API 文档的一部分 |
| [project_summary.md](../project_summary.md) | 报告/总结 | 项目总结与指标 | 建议明确其“报告”定位，避免与权威技术文档口径冲突 |

## 文档型配置模板

| 路径 | 用途 | 说明 |
|---|---|---|
| [config.memory_config.example.json](../config.memory_config.example.json) | MemoryConfig 样例 | 偏向 DeepMemory + Qdrant + 压缩/优化器 |
| [benchmark/locomo/config.example.json](../benchmark/locomo/config.example.json) | LOCOMO 集成跑参数样例 | 跑分阈值、并发与 mock 开关 |
| [config.json](../config.json) | 历史/示例配置 | 结构与当前主链路的 `memory_config` 不一致，建议在新文档中说明其定位与是否生效 |

## 重构后的推荐入口（本次新增）

- 架构与运行流程：`docs/architecture/overview.md`
- 配置与调参：`docs/configuration/overview.md`
- 模块参数与算法原理：
  - `docs/modules/plugin.md`
  - `docs/modules/memory_layers.md`
  - `docs/modules/ranker.md`
  - `docs/modules/budget.md`
  - `docs/modules/vector_store.md`
  - `docs/modules/compression.md`
  - `docs/modules/temporal_model.md`
- 服务与 API 参考：`docs/api/http.md`
- 基准测试（LOCOMO）：`docs/benchmark/locomo.md`
