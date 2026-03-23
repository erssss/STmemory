import json
import os.path
import sys

try:
    import pandas as pd
except Exception:
    pd = None

results_dir = sys.argv[1] if len(sys.argv) > 1 else "results"


def load_token_usage():
    try:
        with open(os.path.join(results_dir, "token1.json"), "r") as f:
            token1_data = json.load(f)
        with open(os.path.join(results_dir, "token2.json"), "r") as f:
            token2_data = json.load(f)

        token1_count = token1_data["token_count"]
        token2_count = token2_data["token_count"]

        token_usage = {
            "prompt_tokens": token2_count["prompt_tokens"] - token1_count["prompt_tokens"],
            "completion_tokens": token2_count["completion_tokens"] - token1_count["completion_tokens"],
            "total_tokens": token2_count["total_tokens"] - token1_count["total_tokens"],
            "cached_tokens": token2_count["cached_tokens"] - token1_count["cached_tokens"],
        }

        return token_usage
    except FileNotFoundError as e:
        print(f"Warning: Could not load token files: {e}")
        return None
    except Exception as e:
        print(f"Warning: Error processing token data: {e}")
        return None


with open(os.path.join(results_dir, "evaluation_metrics.json"), "r") as f:
    data = json.load(f)

all_items = []
for key in data:
    all_items.extend(data[key])

if pd is not None:
    df = pd.DataFrame(all_items)
    df["category"] = pd.to_numeric(df["category"])

    result = df.groupby("category").agg({"bleu_score": "mean", "f1_score": "mean", "llm_score": "mean"}).round(4)
    result["count"] = df.groupby("category").size()

    print("Mean Scores Per Category:")
    print(result)

    overall_means = df.agg({"bleu_score": "mean", "f1_score": "mean", "llm_score": "mean"}).round(4)

    print("\nOverall Mean Scores:")
    print(overall_means)
else:
    def _to_float(v):
        try:
            return float(v)
        except Exception:
            return 0.0

    def _to_int(v):
        try:
            return int(v)
        except Exception:
            try:
                return int(float(v))
            except Exception:
                return 0

    groups = {}
    for item in all_items:
        cat = _to_int(item.get("category"))
        g = groups.get(cat)
        if g is None:
            g = {"bleu_score_sum": 0.0, "f1_score_sum": 0.0, "llm_score_sum": 0.0, "count": 0}
            groups[cat] = g
        g["bleu_score_sum"] += _to_float(item.get("bleu_score"))
        g["f1_score_sum"] += _to_float(item.get("f1_score"))
        g["llm_score_sum"] += _to_float(item.get("llm_score"))
        g["count"] += 1

    overall = {"bleu_score_sum": 0.0, "f1_score_sum": 0.0, "llm_score_sum": 0.0, "count": 0}
    for g in groups.values():
        overall["bleu_score_sum"] += g["bleu_score_sum"]
        overall["f1_score_sum"] += g["f1_score_sum"]
        overall["llm_score_sum"] += g["llm_score_sum"]
        overall["count"] += g["count"]

    def _mean(sum_v, count):
        return (sum_v / count) if count else 0.0

    print("Mean Scores Per Category:")
    header = f"{'category':>8}  {'bleu_score':>10}  {'f1_score':>8}  {'llm_score':>9}  {'count':>5}"
    print(header)
    for cat in sorted(groups.keys()):
        g = groups[cat]
        print(
            f"{cat:>8}  "
            f"{_mean(g['bleu_score_sum'], g['count']):>10.4f}  "
            f"{_mean(g['f1_score_sum'], g['count']):>8.4f}  "
            f"{_mean(g['llm_score_sum'], g['count']):>9.4f}  "
            f"{g['count']:>5}"
        )

    print("\nOverall Mean Scores:")
    print(
        f"bleu_score    {_mean(overall['bleu_score_sum'], overall['count']):.4f}\n"
        f"f1_score      {_mean(overall['f1_score_sum'], overall['count']):.4f}\n"
        f"llm_score     {_mean(overall['llm_score_sum'], overall['count']):.4f}"
    )

token_usage = load_token_usage()
if token_usage:
    print("\nToken Usage During Evaluation:")
    print(f"Prompt tokens used: {token_usage['prompt_tokens']:,}")
    print(f"Completion tokens used: {token_usage['completion_tokens']:,}")
    print(f"Total tokens used: {token_usage['total_tokens']:,}")
    print(f"Cached tokens used: {token_usage['cached_tokens']:,}")
else:
    print("\nToken usage information not available.")
