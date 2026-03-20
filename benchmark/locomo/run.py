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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LOCOMO benchmark integrated with STmemory")
    parser.add_argument("--config", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--model", default=os.getenv("MODEL") or "mock")
    parser.add_argument("--openai-api-key", default=os.getenv("OPENAI_API_KEY") or "mock")
    parser.add_argument("--openai-base-url", default=os.getenv("OPENAI_BASE_URL"))
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--filter-memories", action="store_true", default=False)
    parser.add_argument("--is-graph", action="store_true", default=False)
    parser.add_argument("--max-workers", type=int, default=10)
    parser.add_argument("--use-mock-openai", action="store_true", default=False)
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
            if getattr(args, key, None) in (None, False) and value is not None:
                setattr(args, key, value)

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    locomo_dir = _locomo_dir(repo_root)
    api_base_url = f"http://{args.host}:{args.port}"
    dataset_path = os.path.join(locomo_dir, "dataset", "locomo10.json")

    output_dir = args.output_dir
    if not output_dir:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(repo_root, "benchmark", "results", "locomo", stamp)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    enable_mock_openai = bool(args.use_mock_openai)
    effective_openai_base_url = args.openai_base_url
    if enable_mock_openai:
        effective_openai_base_url = f"{api_base_url}/v1"
    if not effective_openai_base_url:
        effective_openai_base_url = "https://api.openai.com/v1"

    server_cmd = [
        sys.executable,
        "-m",
        "benchmark.locomo.server",
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    if enable_mock_openai:
        server_cmd.extend(["--enable-mock-openai", "--dataset", dataset_path])

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
        env.update(
            {
                "API_BASE_URL": api_base_url,
                "OPENAI_API_KEY": args.openai_api_key,
                "OPENAI_BASE_URL": effective_openai_base_url,
                "MODEL": args.model,
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "LOCOMO_LOCAL_JUDGE": "1" if enable_mock_openai else "0",
                "LOCOMO_SEARCH_WORKERS": str(args.max_workers),
            }
        )

        _http_json("POST", f"{api_base_url}/reset_token_count")
        token1 = _http_json("GET", f"{api_base_url}/token_count")
        with open(os.path.join(output_dir, "token1.json"), "w", encoding="utf-8") as f:
            json.dump(token1, f, indent=2)

        run_args_add = ["run_experiments.py", "--method", "add", "--output_folder", output_dir]
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
            "openai_base_url": effective_openai_base_url,
            "model": args.model,
            "thresholds": {"max_add_p95_s": args.max_add_p95_s, "max_search_p95_s": args.max_search_p95_s},
            "threshold_pass": pass_thresholds,
            "threshold_fail_reasons": reasons,
            "latency": latency,
            "metrics": summary,
        }
        with open(os.path.join(output_dir, "report.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        if args.fail_on_threshold and not pass_thresholds:
            raise SystemExit(2)

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
