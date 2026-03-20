import argparse
import concurrent.futures
import json
import os
import threading
import time
from collections import defaultdict

from metrics.llm_judge import evaluate_llm_judge
from metrics.utils import calculate_bleu_scores, calculate_metrics
try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable, *args, **kwargs):
        return iterable


def process_item(item_data):
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
            llm_score = evaluate_llm_judge(question, gt_answer, pred_answer)
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

    args = parser.parse_args()

    with open(args.input_file, "r") as f:
        data = json.load(f)

    results = defaultdict(list)
    results_lock = threading.Lock()

    effective_workers = int(os.getenv("LOCOMO_EVAL_WORKERS") or str(args.max_workers))
    if os.getenv("LOCOMO_USE_MOCK_LLM") != "1":
        effective_workers = min(effective_workers, int(os.getenv("LOCOMO_EVAL_WORKERS_LIVE") or "1"))
    effective_workers = max(1, effective_workers)
    global_timeout_s = float(os.getenv("LOCOMO_EVAL_GLOBAL_TIMEOUT_S") or "3600")

    with concurrent.futures.ThreadPoolExecutor(max_workers=effective_workers) as executor:
        futures = [executor.submit(process_item, item_data) for item_data in data.items()]

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
            raise RuntimeError(f"Evaluation timed out after {int(time.time() - t0)}s. Set LOCOMO_EVAL_GLOBAL_TIMEOUT_S/LOCOMO_EVAL_WORKERS_LIVE to tune.")

    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
