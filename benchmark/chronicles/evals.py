import argparse
import concurrent.futures
import json
import os
import threading
import time
from collections import defaultdict

from metrics.llm_judge import LocomoJudgeConfig, evaluate_llm_judge
from metrics.utils import calculate_bleu_scores, calculate_metrics
try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable, *args, **kwargs):
        return iterable


def process_item(item_data, judge_cfg: LocomoJudgeConfig):
    k, v = item_data
    local_results = defaultdict(list)

    for item in v:
        gt_answer = str(item["answer"])
        pred_answer = str(item["response"])
        category = str(item["category"])
        question = str(item["question"])

        if category == "5":
            continue

        metrics = calculate_metrics(pred_answer, gt_answer)
        bleu_scores = calculate_bleu_scores(pred_answer, gt_answer)
        try:
            llm_score = evaluate_llm_judge(question, gt_answer, pred_answer, config=judge_cfg)
        except Exception:
            llm_score = 0

        local_results[k].append(
            {
                "question": question,
                "answer": gt_answer,
                "response": pred_answer,
                "category": category,
                "bleu_score": bleu_scores["bleu1"],
                "f1_score": metrics["f1"],
                "llm_score": llm_score,
            }
        )

    return local_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG results")
    parser.add_argument("--input_file", type=str, default="results/rag_results_500_k1.json", help="Path to the input dataset file")
    parser.add_argument("--output_file", type=str, default="results/evaluation_metrics.json", help="Path to save the evaluation results")
    parser.add_argument("--max_workers", type=int, default=10, help="Maximum number of worker threads")
    parser.add_argument("--use-mock-llm", action="store_true", default=False, help="Use mock LLM judge")
    parser.add_argument("--eval-workers", type=int, default=None, help="Worker threads for evaluation")
    parser.add_argument("--eval-workers-live", type=int, default=1, help="Max workers when using live LLM")
    parser.add_argument("--global-timeout-s", type=float, default=3600.0, help="Global timeout for evaluation")
    parser.add_argument("--model", type=str, default="MiniMax-M2.7", help="LLM model for judging")
    parser.add_argument("--metrics-path", type=str, default=None, help="Path to append provider metrics jsonl")
    parser.add_argument("--llm-log-path", type=str, default=None, help="Path to append LLM prompt/response jsonl")
    parser.add_argument("--llm-request-timeout-s", type=float, default=60.0, help="LLM request timeout (seconds)")
    parser.add_argument("--judge-max-tokens", type=int, default=256, help="Max tokens for judge completion")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for judge completion")
    parser.add_argument("--system-prompt", type=str, default="You are a rigorous evaluator.", help="System prompt for judge")
    parser.add_argument("--llm-provider", type=str, default=os.getenv("LOCOMO_LLM_PROVIDER") or "", help="LLM provider: anthropic/minimax or openai_compat/ollama")
    parser.add_argument("--openai-base-url", type=str, default=os.getenv("OPENAI_BASE_URL") or "", help="OpenAI-compatible base URL (e.g. http://127.0.0.1:11434/v1)")
    parser.add_argument("--openai-api-key", type=str, default=os.getenv("OPENAI_API_KEY") or "", help="OpenAI-compatible API key (can be empty for local)")

    args = parser.parse_args()

    with open(args.input_file, "r") as f:
        data = json.load(f)

    results = defaultdict(list)
    results_lock = threading.Lock()

    effective_workers = int(args.eval_workers if args.eval_workers is not None else args.max_workers)
    if not bool(args.use_mock_llm):
        effective_workers = min(effective_workers, int(args.eval_workers_live))
    effective_workers = max(1, effective_workers)
    global_timeout_s = float(args.global_timeout_s)

    judge_cfg = LocomoJudgeConfig(
        use_mock_llm=bool(args.use_mock_llm),
        model=str(args.model or "MiniMax-M2.7"),
        metrics_path=(str(args.metrics_path) if args.metrics_path else None),
        llm_log_path=(str(args.llm_log_path) if args.llm_log_path else None),
        request_timeout_s=float(args.llm_request_timeout_s),
        max_tokens=int(args.judge_max_tokens),
        temperature=float(args.temperature),
        system_prompt=str(args.system_prompt),
        llm_provider=str(args.llm_provider),
        openai_base_url=str(args.openai_base_url),
        openai_api_key=str(args.openai_api_key),
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=effective_workers) as executor:
        futures = [executor.submit(process_item, item_data, judge_cfg) for item_data in data.items()]

        t0 = time.time()
        try:
            for future in tqdm(concurrent.futures.as_completed(futures, timeout=global_timeout_s), total=len(futures)):
                try:
                    local_results = future.result()
                except Exception:
                    continue
                with results_lock:
                    for k, items in local_results.items():
                        results[k].extend(items)
        except concurrent.futures.TimeoutError:
            for f in futures:
                f.cancel()
            raise RuntimeError(f"Evaluation timed out after {int(time.time() - t0)}s.")

    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
