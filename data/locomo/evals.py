
import argparse
import concurrent.futures
import json
import os
import sys
import time
import threading
from collections import defaultdict

# 从 metrics.llm_judge 导入 LLM 评测函数
from metrics.llm_judge import evaluate_llm_judge
# 从 metrics.utils 导入 BLEU 分数和其他评测指标的计算函数
from metrics.utils import calculate_bleu_scores, calculate_metrics, simple_tokenize
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(
        iterable,
        total=None,
        desc=None,
        leave=True,
        file=None,
        mininterval=0.2,
        **kwargs,
    ):
        if file is None:
            file = sys.stdout
        if total is None:
            try:
                total = len(iterable)
            except Exception:
                total = None
        prefix = (str(desc).strip() + ": ") if desc else ""
        last_render = 0.0
        count = 0
        last_line = ""
        for item in iterable:
            yield item
            count += 1
            now = time.time()
            if (now - last_render) < float(mininterval) and (not total or count < total):
                continue
            if total:
                last_line = f"{prefix}{count}/{total}"
            else:
                last_line = f"{prefix}{count}"
            try:
                file.write("\r" + last_line)
                file.flush()
            except Exception:
                pass
            last_render = now
        if last_line:
            try:
                if leave:
                    file.write("\n")
                else:
                    file.write("\r" + (" " * len(last_line)) + "\r")
                file.flush()
            except Exception:
                pass



def process_single_item(k, item, skip_llm_judge, skip_bleu):
    gt_answer = str(item.get("answer", ""))
    pred_answer = str(item.get("response", ""))
    category = str(item.get("category", ""))
    question = str(item.get("question", ""))
    adv_answer = str(item.get("adversarial_answer", "") or "")

    if not gt_answer.strip() and category != "5":
        return None

    metrics = {}
    if gt_answer.strip():
        metrics = calculate_metrics(pred_answer, gt_answer)
    else:
        metrics = {"f1": 0.0}
    bleu1 = None
    if not skip_bleu:
        if gt_answer.strip():
            bleu_scores = calculate_bleu_scores(pred_answer, gt_answer)
            bleu1 = bleu_scores.get("bleu1")

    llm_score = None
    if category == "5":
        pt = set(simple_tokenize(pred_answer))
        at = set(simple_tokenize(adv_answer))
        overlap = 0.0
        if at:
            overlap = len(pt & at) / max(1, len(at))
        adversarial_hit = int(
            pred_answer.strip().lower() == adv_answer.strip().lower() or overlap >= 0.5
        )
        judge_gold = adv_answer if adv_answer.strip() else gt_answer
        if judge_gold.strip():
            llm_score_raw = evaluate_llm_judge(question, judge_gold, pred_answer)
            llm_score = 1 - llm_score_raw if adv_answer.strip() else llm_score_raw
    else:
        if not skip_llm_judge:
            llm_score = evaluate_llm_judge(question, gt_answer, pred_answer)

    result = {
        "question": question,
        "answer": gt_answer,
        "response": pred_answer,
        "category": category,
        "f1_score": metrics.get("f1", 0.0),
    }
    if "sbert_similarity" in metrics:
        result["sbert_similarity"] = metrics["sbert_similarity"]
    if bleu1 is not None:
        result["bleu_score"] = bleu1
    if llm_score is not None:
        result["llm_score"] = llm_score
    if category == "5":
        result["adversarial_answer"] = adv_answer
        result["adversarial_overlap"] = overlap
        result["adversarial_hit"] = adversarial_hit
    return k, result



def main():
    """
    主函数：解析参数，读取数据，进行多线程评测，并保存结果。
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_results_dir = os.path.join(base_dir, "results")
    default_input_file = os.path.join(default_results_dir, "results.json")
    if not os.path.exists(default_input_file):
        default_input_file = os.path.join(default_results_dir, "rag_results_500_k1.json")
    default_output_file = os.path.join(default_results_dir, "evaluation_metrics.json")

    parser = argparse.ArgumentParser(description="评估RAG结果")
    parser.add_argument(
        "--input_file", type=str, default=default_input_file, help="输入数据集文件路径"
    )
    parser.add_argument(
        "--output_file", type=str, default=default_output_file, help="评测结果保存路径"
    )
    parser.add_argument("--max_workers", type=int, default=10, help="最大线程数")
    parser.add_argument("--skip_llm_judge", action="store_true", help="跳过 LLM judge 评测")
    parser.add_argument("--skip_bleu", action="store_true", help="跳过 BLEU 评测")
    parser.add_argument("--max_items", type=int, default=0, help="最多评测多少条样本（0 表示全部）")

    args = parser.parse_args()

    # 读取输入数据
    with open(args.input_file, "r") as f:
        data = json.load(f)

    results = defaultdict(list)  # 汇总所有评测结果
    results_lock = threading.Lock()  # 线程锁，保证多线程写入安全
    records = []
    for k, v in data.items():
        if not isinstance(v, list):
            continue
        for item in v:
            if isinstance(item, dict):
                records.append((k, item))
    if args.max_items and args.max_items > 0:
        records = records[: args.max_items]

    # 使用 ThreadPoolExecutor 进行多线程处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [
            executor.submit(
                process_single_item,
                k,
                item,
                args.skip_llm_judge,
                args.skip_bleu,
            )
            for (k, item) in records
        ]

        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Evaluating",
            file=sys.stdout,
        ):
            out = future.result()
            if out is None:
                continue
            k, result = out
            with results_lock:
                results[k].append(result)

    # 保存评测结果到 JSON 文件
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"评测结果已保存到 {args.output_file}")



# 程序入口
if __name__ == "__main__":
    main()
