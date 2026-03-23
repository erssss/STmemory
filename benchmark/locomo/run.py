import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from benchmark.locomo.reporting import (
    LocomoThresholds,
    check_thresholds,
    parse_latency_from_evaluation_txt,
    summarize_metrics,
    write_evaluation_csv,
)


def _http_json(method: str, url: str, body: Optional[Dict[str, Any]] = None, timeout_s: int = 10) -> Dict[str, Any]:
    data = None
    headers = {"Content-Type": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url=url, data=data, method=method, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read()
    if not raw:
        return {}
    return json.loads(raw.decode("utf-8"))


def _wait_healthz(base_url: str, timeout_s: int = 30) -> None:
    t0 = time.time()
    last_err = None
    while True:
        try:
            _http_json("GET", f"{base_url}/healthz", timeout_s=3)
            return
        except Exception as e:
            last_err = e
        if time.time() - t0 > timeout_s:
            raise RuntimeError(f"Server not ready: {last_err}")
        time.sleep(0.5)


def _wait_healthz_with_process(base_url: str, proc: subprocess.Popen, timeout_s: int = 30, log_sink: Optional[List[str]] = None) -> None:
    t0 = time.time()
    last_err: Optional[BaseException] = None
    while True:
        if proc.poll() is not None:
            out = ""
            if proc.stdout is not None:
                try:
                    out = proc.stdout.read() or ""
                except Exception:
                    out = ""
            if log_sink is not None and out:
                log_sink.append(out)
            msg = f"Server exited early (exit_code={proc.returncode})."
            if out.strip():
                msg = f"{msg}\n{out.strip()}"
            raise RuntimeError(msg)
        try:
            _http_json("GET", f"{base_url}/healthz", timeout_s=3)
            return
        except Exception as e:
            last_err = e
        if time.time() - t0 > timeout_s:
            raise RuntimeError(f"Server not ready: {last_err}")
        time.sleep(0.5)


def _locomo_dir(repo_root: str) -> str:
    locomo_dir = os.path.abspath(os.path.dirname(__file__))
    dataset_path = os.path.join(locomo_dir, "dataset", "locomo10.json")
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"Missing LOCOMO dataset: {dataset_path}")
    return locomo_dir


def _run_locomo_script(locomo_dir: str, args: List[str], env: Dict[str, str]) -> None:
    subprocess.run([sys.executable, *args], cwd=locomo_dir, env=env, check=True)


def _append_generate_scores(locomo_dir: str, results_dir: str, evaluation_txt_path: str, env: Dict[str, str]) -> None:
    p = subprocess.run(
        [sys.executable, "generate_scores.py", results_dir],
        cwd=locomo_dir,
        env=env,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    with open(evaluation_txt_path, "a", encoding="utf-8") as f:
        f.write("\n")
        f.write(p.stdout)


def _compute_latency_stats(values: List[float]) -> Dict[str, Optional[float]]:
    vs = [float(v) for v in values if v is not None]
    if not vs:
        return {"count": 0, "avg_s": None, "p95_s": None, "max_s": None}
    vs_sorted = sorted(vs)
    n = len(vs_sorted)
    p95_idx = max(0, min(n - 1, int(round(0.95 * (n - 1)))))
    return {
        "count": n,
        "avg_s": sum(vs_sorted) / n,
        "p95_s": vs_sorted[p95_idx],
        "max_s": vs_sorted[-1],
    }


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                rows.append(json.loads(s))
            except Exception:
                continue
    return rows


def _select_subset_indices(data: List[Any], subset_size: int, seed: int) -> List[int]:
    if subset_size <= 0 or not data:
        return []
    subset_size = min(int(subset_size), len(data))

    import random

    rng = random.Random(int(seed))
    groups: Dict[str, List[int]] = {}
    for idx, item in enumerate(data):
        qa = (item or {}).get("qa") or []
        cats = []
        for q in qa:
            try:
                c = int((q or {}).get("category", -1))
            except Exception:
                c = -1
            if c == 5:
                continue
            cats.append(c)
        sig = ",".join(str(x) for x in sorted(set(cats)))
        key = f"cats:{sig}|qa:{len(qa)}"
        groups.setdefault(key, []).append(idx)

    for k in list(groups.keys()):
        rng.shuffle(groups[k])

    keys = sorted(groups.keys(), key=lambda k: (len(groups[k]), k))
    chosen: List[int] = []
    cursor = 0
    while len(chosen) < subset_size and keys:
        k = keys[cursor % len(keys)]
        bucket = groups.get(k) or []
        if not bucket:
            keys = [x for x in keys if groups.get(x)]
            cursor += 1
            continue
        chosen.append(bucket.pop())
        cursor += 1

    chosen = sorted(set(chosen))
    if len(chosen) > subset_size:
        chosen = chosen[:subset_size]
    if len(chosen) < subset_size:
        remaining = [i for i in range(len(data)) if i not in set(chosen)]
        rng.shuffle(remaining)
        chosen.extend(sorted(remaining[: (subset_size - len(chosen))]))
        chosen = sorted(set(chosen))
    return chosen


def _estimate_item_cost(item: Any) -> Tuple[int, int, int]:
    qa = (item or {}).get("qa") or []
    qa_non_adv = 0
    for q in qa:
        try:
            c = int((q or {}).get("category", -1))
        except Exception:
            c = -1
        if c == 5:
            continue
        qa_non_adv += 1

    conv = (item or {}).get("conversation") or {}
    turns = 0
    chars = 0
    for k, v in (conv or {}).items():
        if not isinstance(k, str) or not k.startswith("session_") or k.endswith("_date_time"):
            continue
        if not isinstance(v, list):
            continue
        for chat in v:
            if not isinstance(chat, dict):
                continue
            t = str(chat.get("text") or "")
            turns += 1
            chars += len(t)

    return qa_non_adv, turns, chars


def _select_cheapest_index(data: List[Any]) -> Optional[int]:
    if not data:
        return None
    scored: List[Tuple[Tuple[int, int, int], int]] = []
    for idx, item in enumerate(data):
        cost = _estimate_item_cost(item)
        scored.append((cost, idx))
    candidates = [(c, i) for (c, i) in scored if c[0] > 0]
    if candidates:
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]
    scored.sort(key=lambda x: x[0])
    return scored[0][1] if scored else None


def _probe_report(output_dir: str) -> Dict[str, Any]:
    results_path = os.path.join(output_dir, "results.json")
    evaluation_metrics_path = os.path.join(output_dir, "evaluation_metrics.json")
    metrics_jsonl_path = os.path.join(output_dir, "minimax_api_metrics.jsonl")

    answer_latencies: List[float] = []
    answer_total = 0
    answer_ok = 0
    if os.path.exists(results_path):
        try:
            with open(results_path, "r", encoding="utf-8") as f:
                data = json.load(f) or {}
        except Exception:
            data = {}
        for _, items in (data or {}).items():
            for it in items or []:
                answer_total += 1
                rt = it.get("response_time")
                try:
                    rt_f = float(rt)
                except Exception:
                    rt_f = None
                if rt_f is not None and rt_f >= 0:
                    answer_latencies.append(rt_f)
                resp = str(it.get("response") or "")
                if resp and not resp.startswith("Unable to process"):
                    answer_ok += 1

    rows = _load_jsonl(metrics_jsonl_path)
    judge_latencies = [float(r.get("latency_s")) for r in rows if r.get("type") == "judge" and r.get("latency_s") is not None]
    judge_total = sum(1 for r in rows if r.get("type") == "judge")
    judge_ok = sum(1 for r in rows if r.get("type") == "judge" and bool(r.get("success")))

    quality_summary = summarize_metrics(evaluation_metrics_path) if os.path.exists(evaluation_metrics_path) else {}

    return {
        "answer": {
            "success_rate": (answer_ok / answer_total) if answer_total else None,
            "latency": _compute_latency_stats(answer_latencies),
            "total": answer_total,
            "success": answer_ok,
        },
        "judge": {
            "success_rate": (judge_ok / judge_total) if judge_total else None,
            "latency": _compute_latency_stats(judge_latencies),
            "total": judge_total,
            "success": judge_ok,
        },
        "quality": quality_summary,
        "metrics_path": metrics_jsonl_path if os.path.exists(metrics_jsonl_path) else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LOCOMO benchmark integrated with STmemory")
    parser.add_argument("--config", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dataset", default=None, help="LOCOMO dataset path")
    parser.add_argument("--model", default=os.getenv("MODEL") or "MiniMax-M2.7")
    parser.add_argument("--minimax-api-key", default=os.getenv("MINIMAX_API_KEY") or os.getenv("ANTHROPIC_API_KEY") or "")
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--filter-memories", action="store_true", default=False)
    parser.add_argument("--is-graph", action="store_true", default=False)
    parser.add_argument("--max-workers", type=int, default=10)
    parser.add_argument("--use-mock-llm", action="store_true", dest="use_mock_llm", default=False)
    parser.add_argument("--use-mock-openai", action="store_true", dest="use_mock_llm", default=False)
    parser.add_argument("--probe-size", type=int, default=1)
    parser.add_argument("--probe-seed", type=int, default=42)
    parser.add_argument("--subset-index", type=int, default=None)
    parser.add_argument("--max-qa", type=int, default=1)
    parser.add_argument("--skip-probe", action="store_true", default=False)
    parser.add_argument("--run-full", action="store_true", default=False)
    parser.add_argument("--max-add-p95-s", type=float, default=None)
    parser.add_argument("--max-search-p95-s", type=float, default=None)
    parser.add_argument("--fail-on-threshold", action="store_true", default=False)
    args = parser.parse_args()

    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        for key, value in cfg.items():
            if key == "thresholds" and isinstance(value, dict):
                if args.max_add_p95_s is None and value.get("max_add_p95_s") is not None:
                    args.max_add_p95_s = float(value["max_add_p95_s"])
                if args.max_search_p95_s is None and value.get("max_search_p95_s") is not None:
                    args.max_search_p95_s = float(value["max_search_p95_s"])
                continue
            if key == "openai_api_key" and not getattr(args, "minimax_api_key", "") and value:
                setattr(args, "minimax_api_key", str(value))
                continue
            if key == "use_mock_openai" and not bool(getattr(args, "use_mock_llm", False)) and value is not None:
                setattr(args, "use_mock_llm", bool(value))
                continue
            if key == "dataset_path" and getattr(args, "dataset", None) is None and value:
                setattr(args, "dataset", str(value))
                continue
            if getattr(args, key, None) in (None, False) and value is not None:
                setattr(args, key, value)

    if not bool(getattr(args, "use_mock_llm", False)) and not str(getattr(args, "minimax_api_key", "") or "").strip():
        args.use_mock_llm = True
        print("MINIMAX_API_KEY is empty; falling back to mock LLM (LOCOMO_USE_MOCK_LLM=1).", file=sys.stderr)

    args.run_full = False

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    locomo_dir = _locomo_dir(repo_root)
    api_base_url = f"http://{args.host}:{args.port}"
    dataset_path = args.dataset or os.getenv("LOCOMO_DATASET_PATH") or os.path.join(locomo_dir, "dataset", "locomo10.json")
    dataset_path = os.path.abspath(dataset_path)
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"Missing LOCOMO dataset: {dataset_path}")

    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset_data = json.load(f) or []
    except Exception as e:
        raise RuntimeError(f"Failed to read dataset: {dataset_path}") from e

    output_dir = args.output_dir
    if not output_dir:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(repo_root, "benchmark", "results", "locomo", stamp)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    subset_indices: List[int] = []
    effective_probe_size = int(args.probe_size)
    if args.subset_index is not None:
        subset_indices = [int(args.subset_index)]
    elif bool(args.skip_probe):
        subset_indices = []
    elif dataset_data and effective_probe_size > 0:
        if int(effective_probe_size) == 1:
            cheapest = _select_cheapest_index(dataset_data)
            if cheapest is not None:
                subset_indices = [int(cheapest)]
        else:
            subset_indices = _select_subset_indices(dataset_data, effective_probe_size, int(args.probe_seed))

    server_cmd = [
        sys.executable,
        "-m",
        "benchmark.locomo.server",
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]

    server_env = os.environ.copy()
    server_env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    try:
        if "STMEM_DEVICE" not in server_env:
            import torch

            if torch.cuda.is_available():
                server_env["STMEM_DEVICE"] = "cuda"
    except Exception:
        pass
    server_proc = subprocess.Popen(
        server_cmd,
        cwd=repo_root,
        env=server_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    server_log_buf: List[str] = []
    try:
        _wait_healthz_with_process(api_base_url, server_proc, timeout_s=30, log_sink=server_log_buf)

        env = os.environ.copy()
        shims_dir = os.path.join(repo_root, "benchmark", "shims")
        env["PYTHONPATH"] = os.pathsep.join([shims_dir, env.get("PYTHONPATH", "")]).strip(os.pathsep)

        metrics_path = os.path.join(output_dir, "minimax_api_metrics.jsonl")
        env.update(
            {
                "API_BASE_URL": api_base_url,
                "MODEL": args.model,
                "MINIMAX_API_KEY": args.minimax_api_key,
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "LOCOMO_DATASET_PATH": dataset_path,
                "LOCOMO_SEARCH_WORKERS": str(args.max_workers),
                "MINIMAX_METRICS_PATH": metrics_path,
            }
        )
        env.setdefault("LOCOMO_LLM_REQUEST_TIMEOUT_S", "60")
        env.setdefault("LOCOMO_EVAL_WORKERS", str(args.max_workers))
        env.setdefault("LOCOMO_EVAL_WORKERS_LIVE", "1")
        env.setdefault("LOCOMO_EVAL_GLOBAL_TIMEOUT_S", "3600")
        env["LOCOMO_MAX_QA"] = str(int(args.max_qa))
        env["LOCOMO_LLM_LOG_PATH"] = os.path.join(output_dir, "llm_outputs.jsonl")
        if bool(args.use_mock_llm):
            env["LOCOMO_USE_MOCK_LLM"] = "1"
        if subset_indices:
            env["LOCOMO_SUBSET_INDICES"] = ",".join(str(i) for i in subset_indices)

        _http_json("POST", f"{api_base_url}/reset_token_count")
        token1 = _http_json("GET", f"{api_base_url}/token_count")
        with open(os.path.join(output_dir, "token1.json"), "w", encoding="utf-8") as f:
            json.dump(token1, f, indent=2)

        run_args_add = ["run_experiments.py", "--method", "add", "--output_folder", output_dir, "--dataset-path", dataset_path]
        if args.is_graph:
            run_args_add.append("--is_graph")
        _run_locomo_script(locomo_dir, run_args_add, env=env)

        run_args_search = [
            "run_experiments.py",
            "--method",
            "search",
            "--output_folder",
            output_dir,
            "--top_k",
            str(args.top_k),
            "--dataset-path",
            dataset_path,
        ]
        if args.filter_memories:
            run_args_search.append("--filter_memories")
        if args.is_graph:
            run_args_search.append("--is_graph")
        _run_locomo_script(locomo_dir, run_args_search, env=env)

        token2 = _http_json("GET", f"{api_base_url}/token_count")
        with open(os.path.join(output_dir, "token2.json"), "w", encoding="utf-8") as f:
            json.dump(token2, f, indent=2)

        evaluation_metrics_path = os.path.join(output_dir, "evaluation_metrics.json")
        results_json_path = os.path.join(output_dir, "results.json")
        _run_locomo_script(
            locomo_dir,
            ["evals.py", "--input_file", results_json_path, "--output_file", evaluation_metrics_path, "--max_workers", str(args.max_workers)],
            env=env,
        )

        memory_evaluation_path = os.path.join(output_dir, "memory_evaluation.json")
        try:
            _run_locomo_script(
                locomo_dir,
                [
                    "evals_memory.py",
                    "--input_file",
                    results_json_path,
                    "--output_file",
                    memory_evaluation_path,
                    "--top_k",
                    str(args.top_k),
                    "--dataset",
                    dataset_path,
                ],
                env=env,
            )
        except Exception:
            memory_evaluation_path = ""

        evaluation_txt_path = os.path.join(output_dir, "evaluation.txt")
        if not os.path.exists(evaluation_txt_path):
            with open(evaluation_txt_path, "w", encoding="utf-8") as f:
                f.write("")
        _append_generate_scores(locomo_dir, output_dir, evaluation_txt_path, env=env)

        csv_path = os.path.join(output_dir, "evaluation_metrics.csv")
        write_evaluation_csv(evaluation_metrics_path, csv_path)

        latency = parse_latency_from_evaluation_txt(evaluation_txt_path)
        summary = summarize_metrics(evaluation_metrics_path)
        thresholds = LocomoThresholds(max_add_p95_s=args.max_add_p95_s, max_search_p95_s=args.max_search_p95_s)
        pass_thresholds, reasons = check_thresholds(latency, thresholds)

        report = {
            "output_dir": output_dir,
            "api_base_url": api_base_url,
            "dataset": dataset_path,
            "model": args.model,
            "probe": {"subset_indices": subset_indices if subset_indices else None, "probe_size": effective_probe_size, "probe_seed": int(args.probe_seed)},
            "debug": {"llm_log_path": os.path.join(output_dir, "llm_outputs.jsonl"), "max_qa": int(args.max_qa)},
            "thresholds": {"max_add_p95_s": args.max_add_p95_s, "max_search_p95_s": args.max_search_p95_s},
            "threshold_pass": pass_thresholds,
            "threshold_fail_reasons": reasons,
            "latency": latency,
            "metrics": summary,
            "memory_metrics_path": memory_evaluation_path or None,
        }
        with open(os.path.join(output_dir, "report.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        probe = _probe_report(output_dir)
        with open(os.path.join(output_dir, "probe_report.json"), "w", encoding="utf-8") as f:
            json.dump(probe, f, indent=2, ensure_ascii=False)

        if args.fail_on_threshold and not pass_thresholds:
            raise SystemExit(2)

        if bool(args.run_full) and subset_indices:
            full_dir = os.path.join(output_dir, "full")
            os.makedirs(full_dir, exist_ok=True)
            env_full = env.copy()
            env_full.pop("LOCOMO_SUBSET_INDICES", None)
            env_full["MINIMAX_METRICS_PATH"] = os.path.join(full_dir, "minimax_api_metrics.jsonl")

            _http_json("POST", f"{api_base_url}/reset_token_count")
            run_args_add_full = ["run_experiments.py", "--method", "add", "--output_folder", full_dir, "--dataset-path", dataset_path]
            if args.is_graph:
                run_args_add_full.append("--is_graph")
            _run_locomo_script(locomo_dir, run_args_add_full, env=env_full)

            run_args_search_full = [
                "run_experiments.py",
                "--method",
                "search",
                "--output_folder",
                full_dir,
                "--top_k",
                str(args.top_k),
                "--dataset-path",
                dataset_path,
            ]
            if args.filter_memories:
                run_args_search_full.append("--filter_memories")
            if args.is_graph:
                run_args_search_full.append("--is_graph")
            _run_locomo_script(locomo_dir, run_args_search_full, env=env_full)

            evaluation_metrics_path_full = os.path.join(full_dir, "evaluation_metrics.json")
            results_json_path_full = os.path.join(full_dir, "results.json")
            _run_locomo_script(
                locomo_dir,
                ["evals.py", "--input_file", results_json_path_full, "--output_file", evaluation_metrics_path_full, "--max_workers", str(args.max_workers)],
                env=env_full,
            )

            memory_evaluation_path_full = os.path.join(full_dir, "memory_evaluation.json")
            try:
                _run_locomo_script(
                    locomo_dir,
                    [
                        "evals_memory.py",
                        "--input_file",
                        results_json_path_full,
                        "--output_file",
                        memory_evaluation_path_full,
                        "--top_k",
                        str(args.top_k),
                        "--dataset",
                        dataset_path,
                    ],
                    env=env_full,
                )
            except Exception:
                memory_evaluation_path_full = ""

            evaluation_txt_path_full = os.path.join(full_dir, "evaluation.txt")
            if not os.path.exists(evaluation_txt_path_full):
                with open(evaluation_txt_path_full, "w", encoding="utf-8") as f:
                    f.write("")
            _append_generate_scores(locomo_dir, full_dir, evaluation_txt_path_full, env=env_full)

            csv_path_full = os.path.join(full_dir, "evaluation_metrics.csv")
            write_evaluation_csv(evaluation_metrics_path_full, csv_path_full)

            latency_full = parse_latency_from_evaluation_txt(evaluation_txt_path_full)
            summary_full = summarize_metrics(evaluation_metrics_path_full)
            pass_thresholds_full, reasons_full = check_thresholds(latency_full, thresholds)
            report_full = {
                "output_dir": full_dir,
                "api_base_url": api_base_url,
                "dataset": dataset_path,
                "model": args.model,
                "thresholds": {"max_add_p95_s": args.max_add_p95_s, "max_search_p95_s": args.max_search_p95_s},
                "threshold_pass": pass_thresholds_full,
                "threshold_fail_reasons": reasons_full,
                "latency": latency_full,
                "metrics": summary_full,
                "memory_metrics_path": memory_evaluation_path_full or None,
            }
            with open(os.path.join(full_dir, "report.json"), "w", encoding="utf-8") as f:
                json.dump(report_full, f, indent=2, ensure_ascii=False)
            probe_full = _probe_report(full_dir)
            with open(os.path.join(full_dir, "probe_report.json"), "w", encoding="utf-8") as f:
                json.dump(probe_full, f, indent=2, ensure_ascii=False)

    finally:
        if server_proc.poll() is None:
            server_proc.send_signal(signal.SIGTERM)
            try:
                server_proc.wait(timeout=10)
            except Exception:
                server_proc.kill()

        if server_proc.stdout is not None:
            try:
                out = server_proc.stdout.read()
            except Exception:
                out = ""
            if not out and server_log_buf:
                out = "".join(server_log_buf)
            if out:
                with open(os.path.join(output_dir, "server.log"), "w", encoding="utf-8") as f:
                    f.write(out)


if __name__ == "__main__":
    main()
