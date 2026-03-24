from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_DIR = _REPO_ROOT / "src"
if _SRC_DIR.exists():
    sys.path.insert(0, str(_SRC_DIR))


@dataclass(frozen=True)
class StepResult:
    name: str
    data: Dict[str, Any]


class StepRunner:
    def __init__(self, *, pause: bool) -> None:
        self._pause = pause
        self._step_no = 0

    def step(self, title: str, detail: str) -> None:
        self._step_no += 1
        header = f"STEP {self._step_no:02d} | {title}"
        line = "=" * len(header)
        print(f"\n{line}\n{header}\n{line}\n{detail}\n")
        if self._pause:
            input("按回车继续...")


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, default=str)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _setup_logging(output_dir: Path, console_level: str) -> None:
    _ensure_dir(output_dir)

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    for h in list(root.handlers):
        root.removeHandler(h)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    file_handler = logging.FileHandler(output_dir / "demo.log", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(fmt)

    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(getattr(logging, console_level.upper(), logging.INFO))
    console_handler.setFormatter(fmt)

    root.addHandler(file_handler)
    root.addHandler(console_handler)

    logging.getLogger("httpx").setLevel(logging.INFO)
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)


def _build_sdk_config(*, db_path: Path, audit_path: Path) -> Dict[str, Any]:
    return {
        "vector_store": {
            "provider": "sqlite",
            "config": {
                "database_path": str(db_path),
                "collection_name": "memories_demo",
            },
        },
        "llm": {
            "provider": "ollama",
            "config": {
                "ollama_base_url": "http://localhost:11434",
                "model": "qwen3.5:9b",
                "temperature": 0.1,
            },
        },
        "embedder": {
            "provider": "ollama",
            "config": {
                "OLLAMA_EMBEDDING_BASE_URL": "http://localhost:11434",
                "model": "nomic-embed-text",
                "embedding_dims": 512,
            },
        },
        "intelligent_memory": {
            "enabled": True,  # 强制开启事实抽取
            "fallback_to_simple_add": True,
        },
        "telemetry": {
            "enable_telemetry": False,
        },
        "audit": {
            "enabled": True,
            "log_file": str(audit_path),
            "log_level": "INFO",
            "retention_days": 7,
        },
        "logging": {
            "level": "DEBUG",
            "file": "./logs/stmem.log",
        },
    }


def _set_env_for_server_demo(*, db_path: Path) -> None:
    os.environ["POWERMEM_SERVER_AUTH_ENABLED"] = "false"
    os.environ["POWERMEM_SERVER_RATE_LIMIT_ENABLED"] = "false"

    os.environ["DATABASE_PROVIDER"] = "sqlite"
    os.environ["SQLITE_PATH"] = str(db_path)
    os.environ["SQLITE_COLLECTION"] = "memories_api_demo"

    os.environ["INTELLIGENT_MEMORY_ENABLED"] = "false"
    os.environ["EMBEDDING_PROVIDER"] = "ollama"
    os.environ["EMBEDDING_MODEL"] = "nomic-embed-text"
    os.environ["EMBEDDING_DIMS"] = "512"
    os.environ["OLLAMA_EMBEDDING_BASE_URL"] = "http://localhost:11434"

    os.environ["LLM_PROVIDER"] = "ollama"
    os.environ["LLM_MODEL"] = "qwen3.5:9b"
    os.environ["OLLAMA_LLM_BASE_URL"] = "http://localhost:11434"


def _load_locomo_memories(
    *,
    dataset_path: Path,
    item_index: int,
    session_no: int,
    max_messages: int,
) -> List[Tuple[str, Dict[str, Any]]]:
    raw = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError("LOCOMO dataset 必须是非空 JSON 数组")

    if item_index < 0 or item_index >= len(raw):
        raise ValueError(f"item_index 超出范围: {item_index} (len={len(raw)})")

    item = raw[item_index]
    if not isinstance(item, dict):
        raise ValueError("LOCOMO dataset 的每条样本必须是 JSON 对象")

    conversation = item.get("conversation")
    if not isinstance(conversation, dict):
        raise ValueError("LOCOMO 样本缺少 conversation 字段")

    session_key = f"session_{session_no}"
    session = conversation.get(session_key)
    if not isinstance(session, list) or not session:
        raise ValueError(f"LOCOMO conversation 缺少或为空: {session_key}")

    date_time = conversation.get(f"{session_key}_date_time")
    speaker_a = conversation.get("speaker_a")
    speaker_b = conversation.get("speaker_b")

    take_n = len(session) if max_messages <= 0 else min(max_messages, len(session))
    memories: List[Tuple[str, Dict[str, Any]]] = []
    for msg in session[:take_n]:
        if not isinstance(msg, dict):
            continue
        text = msg.get("text")
        speaker = msg.get("speaker")
        if not isinstance(text, str) or not text.strip():
            continue
        if not isinstance(speaker, str) or not speaker.strip():
            speaker = "unknown"

        metadata: Dict[str, Any] = {
            "dataset": "data/locomo",
            "locomo_item_index": item_index,
            "locomo_session_no": session_no,
            "session_date_time": date_time,
            "speaker": speaker,
            "speaker_a": speaker_a,
            "speaker_b": speaker_b,
            "dia_id": msg.get("dia_id"),
            "img_url": msg.get("img_url"),
            "query": msg.get("query"),
            "blip_caption": msg.get("blip_caption"),
        }
        metadata = {k: v for k, v in metadata.items() if v is not None}

        content = f"{speaker}: {text}"
        memories.append((content, metadata))

    if not memories:
        raise ValueError("未能从 LOCOMO session 中抽取到可写入的文本消息")

    return memories


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="PowerMem demo: 分步输出 + 多级日志 + 数据流展示"
    )
    parser.add_argument("--pause", action="store_true", help="每个步骤暂停等待回车")
    parser.add_argument(
        "--with-api",
        action="store_true",
        help="额外演示 API Server 端到端数据流（TestClient 进程内调用）",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="输出目录（默认 demo/full_walkthrough/outputs/<timestamp>/）",
    )
    parser.add_argument(
        "--console-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="控制台日志级别（文件日志固定为 DEBUG）",
    )
    parser.add_argument(
        "--data-source",
        default="locomo",
        choices=["locomo", "simple"],
        help="Demo 数据来源（默认使用 data/locomo）",
    )
    parser.add_argument(
        "--dataset-path",
        default="",
        help="LOCOMO 数据集路径（默认 data/locomo/dataset/locomo10.json）",
    )
    parser.add_argument(
        "--locomo-item-index",
        type=int,
        default=0,
        help="选取 LOCOMO 数据集中的第几条样本（从 0 开始）",
    )
    parser.add_argument(
        "--locomo-session-no",
        type=int,
        default=1,
        help="选取 conversation 的第几段 session（例如 1/2/3...）",
    )
    parser.add_argument(
        "--max-messages",
        type=int,
        default=3,
        help="从选中 session 中最多抽取多少条消息写入（<=0 表示全部）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅加载并展示 Demo 将写入的数据，不执行 SDK 调用",
    )
    args = parser.parse_args(argv)

    base_dir = Path(__file__).resolve().parent
    default_output_dir = base_dir / "outputs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir).resolve() if args.output_dir else default_output_dir
    _ensure_dir(output_dir)

    _setup_logging(output_dir, args.console_level)
    log = logging.getLogger("demo")

    runner = StepRunner(pause=args.pause)
    results: List[StepResult] = []

    db_path = output_dir / "stmem_demo.db"
    audit_path = output_dir / "audit.log"
    run_config = _build_sdk_config(db_path=db_path, audit_path=audit_path)
    (output_dir / "run_config.json").write_text(_json_dumps(run_config), encoding="utf-8")

    runner.step(
        "初始化 Demo 环境",
        "准备输出目录、日志系统与 SDK 配置快照，确保每个关键节点都有可追踪输出。",
    )
    log.info("输出目录: %s", output_dir)
    log.debug("运行配置快照:\n%s", _json_dumps(run_config))

    runner.step(
        "加载 Demo 数据",
        "从 data/locomo 数据集中抽取对话消息作为写入内容（可通过参数切换为简单示例数据）。",
    )
    try:
        if args.data_source == "simple":
            memories_to_add = [
                ("用户喜欢咖啡", {"topic": "preference", "priority": "high"}),
                ("用户更偏好 Python 而不是 Java", {"topic": "tech", "priority": "medium"}),
                ("用户在上海工作，是一名软件工程师", {"topic": "profile", "priority": "high"}),
            ]
        else:
            default_dataset_path = (
                _REPO_ROOT / "data" / "locomo" / "dataset" / "locomo10.json"
            )
            dataset_path = (
                Path(args.dataset_path).resolve()
                if args.dataset_path
                else default_dataset_path
            )
            memories_to_add = _load_locomo_memories(
                dataset_path=dataset_path,
                item_index=args.locomo_item_index,
                session_no=args.locomo_session_no,
                max_messages=args.max_messages,
            )
        (output_dir / "demo_data_preview.json").write_text(
            _json_dumps(
                {
                    "data_source": args.data_source,
                    "count": len(memories_to_add),
                    "items": [
                        {"content": c, "metadata": m} for (c, m) in memories_to_add
                    ],
                }
            ),
            encoding="utf-8",
        )
        log.info("本次将写入 %s 条记忆（预览已写入 demo_data_preview.json）", len(memories_to_add))
    except Exception:
        log.exception("加载 Demo 数据失败")
        return 2

    if args.dry_run:
        runner.step("完成（dry-run）", "已完成数据加载与预览写入；跳过 SDK 调用。")
        return 0

    runner.step(
        "初始化 PowerMem SDK",
        "创建 Memory 实例：VectorStore=SQLite（本地持久化），Embedder=Ollama（本地 embedding），LLM=Ollama（本地 qwen3.5:9b）。",
    )
    try:
        from stmem import Memory
    except ModuleNotFoundError as e:
        log.error("运行环境缺少依赖，无法导入 stmem：%s", e)
        print(
            "\n无法继续执行真实 SDK 演示：当前 Python 环境缺少依赖。\n"
            "建议在可写的虚拟环境中安装依赖后再运行：\n\n"
            "  python -m venv .venv\n"
            "  source .venv/bin/activate\n"
            "  python -m pip install -U pip\n"
            "  python -m pip install -e .\n\n"
            "然后重新执行：\n"
            "  python demo/full_walkthrough/run_demo.py\n"
        )
        return 2
    try:
        memory = Memory(config=run_config)
        log.info("Memory 初始化完成")
    except ImportError as e:
        log.error("初始化失败：缺少依赖：%s", e)
        print(
            "\n无法继续执行真实 SDK 演示：当前 Python 环境缺少依赖（例如 ollama）。\n"
            "建议在虚拟环境中安装依赖后再运行：\n\n"
            "  python -m venv .venv\n"
            "  source .venv/bin/activate\n"
            "  python -m pip install -U pip\n"
            "  python -m pip install -e .\n"
            "  python -m pip install ollama\n\n"
            "并确保本机 Ollama 已启动且已拉取模型：\n"
            "  ollama serve\n"
            "  ollama pull qwen3.5:9b\n"
            "  ollama pull nomic-embed-text\n\n"
            "然后重新执行：\n"
            "  python demo/full_walkthrough/run_demo.py\n"
        )
        return 2
    except Exception:
        log.exception("Memory 初始化失败")
        return 2

    user_id = "demo_user"
    runner.step(
        "写入记忆（Add）",
        "写入多条带 metadata 的记忆；默认 infer=True，会触发 LLM 进行事实抽取与记忆合并决策。",
    )
    added_ids: List[Any] = []
    for content, metadata in memories_to_add:
        log.debug("准备写入: content=%s metadata=%s", content, metadata)
        print("\n[DEMO INPUT] 写入记忆输入:")
        print(json.dumps({"content": content, "metadata": metadata}, ensure_ascii=False, indent=2))
        try:
            r = memory.add(content, user_id=user_id, metadata=metadata)
            print("[DEMO OUTPUT] 写入记忆输出:")
            print(json.dumps(r, ensure_ascii=False, indent=2))
            log.info("写入成功: %s", r)
            # 展示事实抽取结果（如果有）
            if isinstance(r, dict) and "facts" in r:
                print("[DEMO FACTS] 事实抽取:")
                print(json.dumps(r["facts"], ensure_ascii=False, indent=2))
            if isinstance(r, dict):
                mid = None
                results_block = r.get("results")
                if isinstance(results_block, list) and results_block:
                    first = results_block[0]
                    if isinstance(first, dict):
                        mid = first.get("id") or first.get("memory_id")
                if mid is None:
                    mid = r.get("id") or r.get("memory_id")
                if mid is not None:
                    added_ids.append(mid)
        except Exception:
            log.exception("写入失败: content=%s", content)

    results.append(StepResult("added_ids", {"ids": added_ids}))

    runner.step(
        "检索记忆（Search）",
        "执行语义检索：query 文本会被 embed 成向量，随后在 SQLiteVectorStore 中做相似度排序返回 Top-K。",
    )
    search_query = (
        "What did Caroline do yesterday?"
        if args.data_source == "locomo"
        else "用户的偏好是什么？"
    )
    search_resp = memory.search(search_query, user_id=user_id, limit=5)
    log.info("search 响应:\n%s", _json_dumps(search_resp))
    results.append(StepResult("search", {"query": search_query, "response": search_resp}))

    runner.step(
        "读取与更新（Get/Update）",
        "通过 ID 精确读取，再更新内容；更新后再次读取验证数据确实发生变化。",
    )
    updated_info: Dict[str, Any] = {"updated": False}
    if added_ids:
        target_id = added_ids[0]
        before = memory.get(target_id, user_id=user_id)
        log.debug("更新前 get(%s): %s", target_id, _json_dumps(before))
        updated_content = (
            "Caroline: I went to an LGBTQ support group yesterday and it was powerful."
            if args.data_source == "locomo"
            else "用户喜欢咖啡（且偏好手冲）"
        )
        updated = memory.update(target_id, updated_content, user_id=user_id)
        after = memory.get(target_id, user_id=user_id)
        log.info("update 返回: %s", updated)
        log.info("更新后 get(%s): %s", target_id, _json_dumps(after))
        updated_info = {
            "id": target_id,
            "before": before,
            "update_result": updated,
            "after": after,
        }
    else:
        log.warning("未获取到可更新的 memory_id，跳过更新步骤")
    results.append(StepResult("update", updated_info))

    runner.step(
        "删除与异常分级（Delete + warning/error）",
        "删除一条记忆，并重复删除同一条，触发可预期的 warning；同时保留异常栈到 demo.log 便于定位。",
    )
    delete_info: Dict[str, Any] = {"deleted": False}
    if added_ids:
        del_id = added_ids[-1]
        ok1 = memory.delete(del_id, user_id=user_id)
        log.info("第一次 delete(%s) 返回: %s", del_id, ok1)
        ok2 = memory.delete(del_id, user_id=user_id)
        if ok2:
            log.warning("第二次 delete(%s) 仍返回 True（实现可能是幂等删除）", del_id)
        else:
            log.warning("第二次 delete(%s) 返回 False（对象已不存在，符合预期）", del_id)
        delete_info = {"id": del_id, "first": ok1, "second": ok2}
    else:
        log.error("added_ids 为空，无法演示 delete")
    results.append(StepResult("delete", delete_info))

    runner.step(
        "审计日志（Audit）",
        "审计日志以 JSONL 形式写入文件；这里读取末尾若干行用于直观展示“可追溯事件流”。",
    )
    audit_tail: List[str] = []
    try:
        if audit_path.exists():
            lines = audit_path.read_text(encoding="utf-8").splitlines()
            audit_tail = lines[-5:]
            log.info("audit.log 最后 5 行:\n%s", "\n".join(audit_tail))
        else:
            log.warning("audit.log 未生成：%s", audit_path)
    except Exception:
        log.exception("读取 audit.log 失败")
    results.append(StepResult("audit_tail", {"lines": audit_tail}))

    if args.with_api:
        runner.step(
            "API Server 端到端演示（可选）",
            "通过 TestClient 在同一进程内调用 FastAPI app，展示路由/Middleware/Service/SDK/Storage 串联的数据流与日志。",
        )
        _set_env_for_server_demo(db_path=db_path)

        try:
            try:
                from fastapi.testclient import TestClient
            except ModuleNotFoundError:
                TestClient = None  # type: ignore

            if TestClient is None:
                log.info("fastapi 不可用，改用 Service 层直调方式演示 API 数据流")
                from server.services.memory_service import MemoryService
                from server.services.search_service import SearchService

                api_config = json.loads(json.dumps(run_config))
                api_config["vector_store"]["config"]["collection_name"] = "memories_api_demo"
                api_config["audit"]["log_file"] = str(output_dir / "audit_api.log")

                mem_svc = MemoryService(config=api_config)
                search_svc = SearchService(config=api_config)

                created = mem_svc.create_memory(
                    content="API(直调)写入：用户喜欢深色模式",
                    user_id=user_id,
                    infer=False,
                )
                log.info("MemoryService.create_memory 返回:\n%s", _json_dumps(created))

                searched = search_svc.search_memories(
                    query="深色模式",
                    user_id=user_id,
                    limit=3,
                )
                log.info("SearchService.search_memories 返回:\n%s", _json_dumps(searched))

                error_case: Dict[str, Any] = {}
                try:
                    search_svc.search_memories(query="   ", user_id=user_id, limit=1)
                except Exception as e:
                    log.warning("SearchService 空 query 触发错误（用于演示校验/错误链路）：%s", e)
                    error_case = {"error": str(e), "type": type(e).__name__}

                results.append(
                    StepResult(
                        "api_service",
                        {"create": created, "search": searched, "error_case": error_case},
                    )
                )
            else:
                from server.middleware.logging import setup_logging as setup_server_logging
                from server.main import app

                setup_server_logging()
                client = TestClient(app)

                create_body = {"content": "API 写入：用户喜欢深色模式", "user_id": user_id}
                r1 = client.post("/api/v1/memories", json=create_body)
                log.info(
                    "POST /api/v1/memories status=%s body=%s", r1.status_code, r1.text
                )

                search_body = {"query": "深色模式", "user_id": user_id, "limit": 3}
                r2 = client.post("/api/v1/memories/search", json=search_body)
                log.info(
                    "POST /api/v1/memories/search status=%s body=%s",
                    r2.status_code,
                    r2.text,
                )

                r3 = client.post("/api/v1/memories", json={})
                log.warning(
                    "POST /api/v1/memories(空 body) status=%s body=%s",
                    r3.status_code,
                    r3.text,
                )

                results.append(
                    StepResult(
                        "api",
                        {
                            "create": {"status": r1.status_code, "body": r1.json()},
                            "search": {"status": r2.status_code, "body": r2.json()},
                            "validation_error": {
                                "status": r3.status_code,
                                "body": r3.json(),
                            },
                        },
                    )
                )
        except Exception:
            log.exception("API 演示失败（不影响 SDK 主流程）")

    snapshot = {r.name: r.data for r in results}
    (output_dir / "result_snapshot.json").write_text(
        _json_dumps(snapshot), encoding="utf-8"
    )

    runner.step(
        "完成",
        "已生成 demo.log / audit.log / SQLite DB / 结果快照。可打开 outputs 目录对照查看每一步的输入、输出与日志分级。",
    )
    log.info("结果快照已写入: %s", output_dir / "result_snapshot.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
