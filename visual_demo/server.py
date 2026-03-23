import argparse
import asyncio
import csv
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

from aiohttp import web, WSMsgType

from memory_layers import MemoryConfig
from plugin import SpatioTemporalMemoryPlugin, make_openai_compatible_api_func


@dataclass(frozen=True)
class DemoConversation:
    query: str
    response: str
    topic: str
    timestamp: datetime


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _static_dir() -> str:
    return os.path.join(os.path.dirname(__file__), "static")


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(x) for x in obj]
    return str(obj)


def _safe_read_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _safe_read_text(path: str, limit_bytes: int = 200_000) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read(limit_bytes)
    except Exception:
        return ""


def _safe_list_locomo_runs() -> List[Dict[str, Any]]:
    roots = [
        os.path.join(_repo_root(), "benchmark", "results", "locomo"),
        "/tmp",
    ]
    out: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for root in roots:
        if not os.path.isdir(root):
            continue
        try:
            names = os.listdir(root)
        except Exception:
            continue
        for name in names:
            if root == "/tmp" and not str(name).startswith("stmemory_locomo"):
                continue
            d = os.path.join(root, name)
            if not os.path.isdir(d):
                continue
            report_path = os.path.join(d, "report.json")
            metrics_path = os.path.join(d, "evaluation_metrics.json")
            if not (os.path.exists(report_path) or os.path.exists(metrics_path)):
                continue
            rp = os.path.realpath(d)
            if rp in seen:
                continue
            seen.add(rp)
            st = None
            try:
                st = os.stat(d)
            except Exception:
                st = None
            out.append(
                {
                    "dir": rp,
                    "name": os.path.basename(rp),
                    "mtime": int(getattr(st, "st_mtime", 0) or 0),
                    "has_report": os.path.exists(report_path),
                    "has_metrics": os.path.exists(metrics_path),
                }
            )
    out.sort(key=lambda x: int(x.get("mtime") or 0), reverse=True)
    return out


def _is_allowed_locomo_dir(path: str) -> bool:
    rp = os.path.realpath(path)
    allow1 = os.path.realpath(os.path.join(_repo_root(), "benchmark", "results", "locomo"))
    allow2 = os.path.realpath("/tmp")
    return rp.startswith(allow1 + os.sep) or rp == allow1 or rp.startswith(allow2 + os.sep) or rp == allow2


def _read_locomo_evaluation_csv(csv_path: str, limit_rows: int = 2000) -> List[Dict[str, Any]]:
    try:
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            out: List[Dict[str, Any]] = []
            for i, row in enumerate(reader):
                if i >= limit_rows:
                    break
                out.append(
                    {
                        "conversation_id": row.get("conversation_id", ""),
                        "category": row.get("category", ""),
                        "question": row.get("question", ""),
                        "answer": row.get("answer", ""),
                        "response": row.get("response", ""),
                        "bleu_score": row.get("bleu_score", ""),
                        "f1_score": row.get("f1_score", ""),
                        "llm_score": row.get("llm_score", ""),
                    }
                )
            return out
    except Exception:
        return []


def _load_locomo_run(dir_path: str) -> Dict[str, Any]:
    if not _is_allowed_locomo_dir(dir_path):
        raise web.HTTPForbidden(text="dir not allowed")
    d = os.path.realpath(dir_path)
    report = _safe_read_json(os.path.join(d, "report.json"))
    metrics = _safe_read_json(os.path.join(d, "evaluation_metrics.json"))
    evaluation_txt = _safe_read_text(os.path.join(d, "evaluation.txt"))
    token1 = _safe_read_json(os.path.join(d, "token1.json"))
    token2 = _safe_read_json(os.path.join(d, "token2.json"))
    csv_rows = _read_locomo_evaluation_csv(os.path.join(d, "evaluation_metrics.csv"))
    return {
        "dir": d,
        "report": report or {},
        "evaluation_metrics": metrics or {},
        "evaluation_txt": evaluation_txt,
        "token1": token1 or {},
        "token2": token2 or {},
        "evaluation_csv_rows": csv_rows,
    }


def _default_demo_conversations() -> List[DemoConversation]:
    base_time = datetime.now()
    return [
        DemoConversation(
            query="什么是机器学习？",
            response="机器学习是人工智能的一个分支，它使计算机能够从数据中学习并做出预测或决策，而无需明确编程。",
            topic="机器学习",
            timestamp=base_time,
        ),
        DemoConversation(
            query="深度学习呢？",
            response="深度学习是机器学习的一个子集，使用多层神经网络来模拟人脑的学习过程，特别适用于图像识别、自然语言处理等复杂任务。",
            topic="机器学习",
            timestamp=base_time + timedelta(minutes=2),
        ),
        DemoConversation(
            query="它们有什么区别？",
            response="主要区别在于：机器学习包含更广泛的算法，而深度学习特指使用深层神经网络的方法。深度学习能处理更复杂的模式，但需要更多数据和计算资源。",
            topic="机器学习",
            timestamp=base_time + timedelta(minutes=5),
        ),
        DemoConversation(
            query="Python适合做什么？",
            response="Python是一种通用编程语言，特别适合数据科学、Web开发、自动化脚本、人工智能和机器学习等领域。",
            topic="Python",
            timestamp=base_time + timedelta(minutes=10),
        ),
        DemoConversation(
            query="和JavaScript相比呢？",
            response="Python和JavaScript都很流行，但用途不同：Python更适合后端开发、数据分析和科学计算；JavaScript主要用于前端Web开发以及Node.js后端。",
            topic="Python",
            timestamp=base_time + timedelta(minutes=12),
        ),
        DemoConversation(
            query="数据科学需要哪些技能？",
            response="数据科学需要统计学知识、编程能力（Python/R）、数据处理技能、机器学习基础，以及良好的业务理解能力。",
            topic="数据科学",
            timestamp=base_time + timedelta(minutes=15),
        ),
        DemoConversation(
            query="学习路线如何规划？",
            response="建议从统计学基础开始，学习Python编程，掌握Pandas/NumPy等数据处理工具，再学习机器学习算法，最后通过实际项目练习。",
            topic="数据科学",
            timestamp=base_time + timedelta(minutes=18),
        ),
        DemoConversation(
            query="机器学习在数据科学中的作用？",
            response="机器学习是数据科学的核心工具之一，用于从数据中发现模式、进行预测和分类，帮助自动化分析并发现复杂关系。",
            topic="机器学习",
            timestamp=base_time + timedelta(minutes=20),
        ),
    ]


class DemoEngine:
    def __init__(self, use_llm: bool, llm_base_url: str, llm_model: str, llm_api_key: str):
        self.use_llm = bool(use_llm)
        self.llm_base_url = llm_base_url
        self.llm_model = llm_model
        self.llm_api_key = llm_api_key
        self._lock = asyncio.Lock()
        self.reset()

    def _new_plugin(self) -> SpatioTemporalMemoryPlugin:
        cfg = MemoryConfig(
            shallow_capacity=50,
            working_capacity=200,
            shallow_ttl=300,
            working_ttl=1800,
            max_shallow_entries=50,
            max_working_entries=200,
            lambda_decay=0.01,
            alpha_similarity=0.5,
            beta_time=0.3,
            gamma_layer=0.2,
        )
        if self.llm_base_url:
            cfg.llm_base_url = str(self.llm_base_url)
        if self.llm_model:
            cfg.llm_model = str(self.llm_model)
        return SpatioTemporalMemoryPlugin(model_name="visual-demo", memory_config=cfg, enable_logging=False)

    def reset(self) -> None:
        self.plugin = self._new_plugin()
        self.conversations = _default_demo_conversations()
        self.turn_idx = 0
        self.history: List[Dict[str, Any]] = []

        if self.use_llm:
            api_key = self.llm_api_key or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
            base_url = str(self.llm_base_url or self.plugin.config.llm_base_url or "").strip()
            model = str(self.llm_model or self.plugin.config.llm_model or "").strip()
            self.openclaw_api_func = make_openai_compatible_api_func(api_key=api_key, base_url=base_url, model=model)
        else:
            self.openclaw_api_func = None

    async def _mock_llm(self, prompt: str) -> str:
        q = prompt
        if "Current query:" in prompt:
            q = prompt.split("Current query:", 1)[-1].strip()
        if q:
            return f"（演示回复）我理解你的问题：{q}"
        return "（演示回复）"

    async def step(self) -> Dict[str, Any]:
        async with self._lock:
            if self.turn_idx >= len(self.conversations):
                return {"done": True, "turn_idx": self.turn_idx, "total_turns": len(self.conversations)}

            conv = self.conversations[self.turn_idx]
            before_memory = self.plugin.get_memory_stats()
            before_perf = self.plugin.get_performance_stats().get("performance", {})
            api_func = self.openclaw_api_func or self._mock_llm
            result = await self.plugin.process_query(conv.query, api_func)
            after_memory = self.plugin.get_memory_stats()
            after_perf = self.plugin.get_performance_stats().get("performance", {})

            changes = {
                "topic": conv.topic,
                "timestamp": _iso(conv.timestamp),
                "shallow_added": (after_memory.get("shallow", {}).get("entries", 0) - before_memory.get("shallow", {}).get("entries", 0)),
                "working_added": (after_memory.get("working", {}).get("entries", 0) - before_memory.get("working", {}).get("entries", 0)),
                "deep_added": (after_memory.get("deep", {}).get("entries", 0) - before_memory.get("deep", {}).get("entries", 0)),
                "meta_added": (after_memory.get("meta", {}).get("entries", 0) - before_memory.get("meta", {}).get("entries", 0)),
                "tokens_saved_delta": (after_perf.get("total_tokens_saved", 0) - before_perf.get("total_tokens_saved", 0)),
                "cache_hit": int(result.get("relevant_memories_count", 0) or 0) > 0,
            }

            payload = {
                "done": False,
                "turn_idx": self.turn_idx,
                "total_turns": len(self.conversations),
                "configuration": self.plugin.export_configuration(),
                "conversation": {
                    "query": conv.query,
                    "topic": conv.topic,
                    "expected_response": conv.response,
                    "timestamp": _iso(conv.timestamp),
                },
                "result": result,
                "memory_stats": after_memory,
                "performance": self.plugin.get_performance_stats(),
                "changes": changes,
            }
            self.history.append(payload)
            self.turn_idx += 1
            return payload

    async def chat(self, query: str) -> Dict[str, Any]:
        api_func = self.openclaw_api_func or self._mock_llm
        result = await self.plugin.process_query(query, api_func)
        return {
            "result": result,
            "configuration": self.plugin.export_configuration(),
            "memory_stats": self.plugin.get_memory_stats(),
            "performance": self.plugin.get_performance_stats(),
        }

    def state(self) -> Dict[str, Any]:
        return {
            "turn_idx": self.turn_idx,
            "total_turns": len(self.conversations),
            "configuration": self.plugin.export_configuration(),
            "memory_stats": self.plugin.get_memory_stats(),
            "performance": self.plugin.get_performance_stats(),
            "history_tail": [h.get("result", {}) for h in self.history[-10:]],
        }


class LocomoRunner:
    def __init__(self):
        self._lock = asyncio.Lock()
        self._ws_clients: Set[web.WebSocketResponse] = set()
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._current_dir: Optional[str] = None

    async def _broadcast(self, payload: Dict[str, Any]) -> None:
        dead: List[web.WebSocketResponse] = []
        for ws in self._ws_clients:
            try:
                await ws.send_json(_jsonable(payload))
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._ws_clients.discard(ws)

    async def attach(self, ws: web.WebSocketResponse) -> None:
        self._ws_clients.add(ws)
        await ws.send_json(_jsonable({"type": "status", "running": self._proc is not None, "dir": self._current_dir}))

    async def detach(self, ws: web.WebSocketResponse) -> None:
        self._ws_clients.discard(ws)

    async def start(self, max_workers: int = 2) -> str:
        async with self._lock:
            if self._proc is not None:
                return self._current_dir or ""
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir = os.path.realpath(os.path.join("/tmp", f"stmemory_locomo_demo_{stamp}"))
            cmd = [
                sys.executable,
                "-m",
                "benchmark.locomo.run",
                "--use-mock-llm",
                "--max-workers",
                str(int(max_workers)),
                "--output-dir",
                out_dir,
            ]
            self._current_dir = out_dir
            self._proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=_repo_root(),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=os.environ.copy(),
            )
            await self._broadcast({"type": "run_started", "dir": out_dir, "cmd": " ".join(cmd)})
            asyncio.create_task(self._pump())
            return out_dir

    async def stop(self) -> None:
        async with self._lock:
            if self._proc is None:
                return
            try:
                self._proc.terminate()
            except Exception:
                pass

    async def _pump(self) -> None:
        proc = self._proc
        out_dir = self._current_dir
        if proc is None:
            return
        try:
            assert proc.stdout is not None
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").rstrip("\n")
                await self._broadcast({"type": "log", "line": text})
            code = await proc.wait()
            payload: Dict[str, Any] = {"type": "run_finished", "exit_code": int(code), "dir": out_dir}
            if out_dir and os.path.exists(os.path.join(out_dir, "report.json")):
                try:
                    payload["run"] = _load_locomo_run(out_dir)
                except Exception:
                    payload["run"] = None
            await self._broadcast(payload)
        finally:
            self._proc = None


def create_app(demo: DemoEngine, locomo: LocomoRunner) -> web.Application:
    app = web.Application()

    async def index(_: web.Request) -> web.FileResponse:
        return web.FileResponse(os.path.join(_static_dir(), "index.html"))

    async def api_demo_state(_: web.Request) -> web.Response:
        return web.json_response(_jsonable(demo.state()))

    async def api_demo_reset(_: web.Request) -> web.Response:
        demo.reset()
        return web.json_response(_jsonable({"status": "ok", "state": demo.state()}))

    async def api_demo_step(_: web.Request) -> web.Response:
        return web.json_response(_jsonable(await demo.step()))

    async def api_chat(request: web.Request) -> web.Response:
        body = await request.json()
        query = str((body or {}).get("query") or "").strip()
        if not query:
            return web.json_response({"error": "query required"}, status=400)
        return web.json_response(_jsonable(await demo.chat(query)))

    async def api_vector_search(request: web.Request) -> web.Response:
        body = await request.json()
        query = str((body or {}).get("query") or "").strip()
        if not query:
            return web.json_response({"error": "query required"}, status=400)
        top_k = (body or {}).get("top_k", 10)
        distance = str((body or {}).get("distance") or "cosine")
        try:
            top_k_i = int(top_k)
        except Exception:
            top_k_i = 10
        res = await demo.plugin.vector_search(query, top_k=top_k_i, distance=distance)
        return web.json_response(_jsonable(res))

    async def api_vector_store_stats(_: web.Request) -> web.Response:
        return web.json_response(_jsonable(demo.plugin.get_vector_store_stats()))

    async def api_locomo_runs(_: web.Request) -> web.Response:
        return web.json_response(_jsonable({"runs": _safe_list_locomo_runs()}))

    async def api_locomo_run(request: web.Request) -> web.Response:
        d = str(request.query.get("dir") or "").strip()
        if not d:
            return web.json_response({"error": "dir required"}, status=400)
        return web.json_response(_jsonable(_load_locomo_run(d)))

    async def ws_locomo(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(heartbeat=20.0)
        await ws.prepare(request)
        await locomo.attach(ws)
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                    except Exception:
                        data = {}
                    action = str((data or {}).get("action") or "")
                    if action == "start":
                        mw = int((data or {}).get("max_workers") or 2)
                        out_dir = await locomo.start(max_workers=mw)
                        await ws.send_json(_jsonable({"type": "ack", "action": "start", "dir": out_dir}))
                    elif action == "stop":
                        await locomo.stop()
                        await ws.send_json(_jsonable({"type": "ack", "action": "stop"}))
                    elif action == "load":
                        d = str((data or {}).get("dir") or "")
                        if d:
                            try:
                                await ws.send_json(_jsonable({"type": "run_loaded", "run": _load_locomo_run(d)}))
                            except Exception as e:
                                await ws.send_json(_jsonable({"type": "error", "message": str(e)}))
                elif msg.type == WSMsgType.ERROR:
                    break
        finally:
            await locomo.detach(ws)
        return ws

    app.router.add_get("/", index)
    app.router.add_static("/static", _static_dir(), show_index=False)
    app.router.add_get("/api/demo/state", api_demo_state)
    app.router.add_post("/api/demo/reset", api_demo_reset)
    app.router.add_post("/api/demo/step", api_demo_step)
    app.router.add_post("/api/chat", api_chat)
    app.router.add_post("/api/vector_search", api_vector_search)
    app.router.add_get("/api/vector_store/stats", api_vector_store_stats)
    app.router.add_get("/api/locomo/runs", api_locomo_runs)
    app.router.add_get("/api/locomo/run", api_locomo_run)
    app.router.add_get("/ws/locomo", ws_locomo)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="STmemory visual demo (interactive UI)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--use-llm", action="store_true", default=False)
    parser.add_argument("--llm-base-url", default="")
    parser.add_argument("--llm-model", default="")
    parser.add_argument("--llm-api-key", default=os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or "")
    args = parser.parse_args()

    demo = DemoEngine(
        use_llm=args.use_llm,
        llm_base_url=str(args.llm_base_url or ""),
        llm_model=str(args.llm_model or ""),
        llm_api_key=str(args.llm_api_key or ""),
    )
    locomo = LocomoRunner()
    app = create_app(demo, locomo)
    web.run_app(app, host=str(args.host), port=int(args.port))


if __name__ == "__main__":
    main()
