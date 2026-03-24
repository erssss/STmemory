import json
import os.path
import sys

import pandas as pd

# 从命令行参数获取结果目录，如果没有则默认使用"results"
results_dir = sys.argv[1] if len(sys.argv) > 1 else "results"


# 加载token使用数据
def load_token_usage():
    def _safe_int(x):
        try:
            return int(x)
        except Exception:
            return 0

    def _read_json_if_exists(path: str):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return None
        except Exception:
            return None

    def _token_usage_from_results_json():
        results_path = os.path.join(results_dir, "results.json")
        results_data = _read_json_if_exists(results_path)
        if not isinstance(results_data, dict):
            return None

        prompt_tokens = 0
        completion_tokens = 0
        cached_tokens = 0
        saw_any = False

        for _, items in results_data.items():
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue

                if "ollama_prompt_eval_count" in item or "ollama_eval_count" in item:
                    prompt_tokens += _safe_int(item.get("ollama_prompt_eval_count"))
                    completion_tokens += _safe_int(item.get("ollama_eval_count"))
                    saw_any = True
                    continue

                token_usage = item.get("token_usage")
                if isinstance(token_usage, dict):
                    prompt_tokens += _safe_int(token_usage.get("prompt_tokens"))
                    completion_tokens += _safe_int(token_usage.get("completion_tokens"))
                    cached_tokens += _safe_int(token_usage.get("cached_tokens"))
                    saw_any = True
                    continue

                usage = item.get("usage")
                if isinstance(usage, dict):
                    prompt_tokens += _safe_int(usage.get("prompt_tokens"))
                    completion_tokens += _safe_int(usage.get("completion_tokens"))
                    cached_tokens += _safe_int(usage.get("cached_tokens"))
                    saw_any = True

        if not saw_any:
            return None

        total_tokens = prompt_tokens + completion_tokens
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cached_tokens": cached_tokens,
        }

    token1_path = os.path.join(results_dir, "token1.json")
    token2_path = os.path.join(results_dir, "token2.json")
    token1_data = _read_json_if_exists(token1_path)
    token2_data = _read_json_if_exists(token2_path)

    if token1_data is None or token2_data is None:
        return _token_usage_from_results_json()

    try:
        token1_count = token1_data.get("token_count", token1_data)
        token2_count = token2_data.get("token_count", token2_data)

        prompt_tokens = _safe_int(token2_count.get("prompt_tokens")) - _safe_int(
            token1_count.get("prompt_tokens")
        )
        completion_tokens = _safe_int(token2_count.get("completion_tokens")) - _safe_int(
            token1_count.get("completion_tokens")
        )
        total_tokens = _safe_int(token2_count.get("total_tokens")) - _safe_int(
            token1_count.get("total_tokens")
        )
        cached_tokens = _safe_int(token2_count.get("cached_tokens")) - _safe_int(
            token1_count.get("cached_tokens")
        )

        if total_tokens == 0 and (prompt_tokens != 0 or completion_tokens != 0):
            total_tokens = prompt_tokens + completion_tokens

        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cached_tokens": cached_tokens,
        }
    except Exception:
        return _token_usage_from_results_json()


# 加载评估指标数据
with open(os.path.join(results_dir, "evaluation_metrics.json"), "r") as f:
    data = json.load(f)

# 将数据扁平化为问题项列表
all_items = []
for key in data:
    all_items.extend(data[key])

# 转换为DataFrame以便分析
df = pd.DataFrame(all_items)

# 将分类转换为数值类型
df["category"] = pd.to_numeric(df["category"])

# 按分类计算平均得分
result = df.groupby("category").agg({"bleu_score": "mean", "f1_score": "mean", "llm_score": "mean"}).round(4)
# result = df.groupby("category").agg({"bleu_score": "mean", "f1_score": "mean"}).round(4)

# 添加每个分类的问题数量
result["count"] = df.groupby("category").size()

# 打印按分类统计的结果
print("按分类平均得分:")
print(result)

# 计算整体平均得分
overall_means = df.agg({"bleu_score": "mean", "f1_score": "mean", "llm_score": "mean"}).round(4)
# overall_means = df.agg({"bleu_score": "mean", "f1_score": "mean"}).round(4)

print("\n整体平均得分:")
print(overall_means)

# 加载并显示token使用信息
token_usage = load_token_usage()
if token_usage:
    print("\n评估过程中的Token使用情况:")
    print(f"使用的提示词tokens: {token_usage['prompt_tokens']:,}")
    print(f"使用的补全tokens: {token_usage['completion_tokens']:,}")
    print(f"使用的总tokens: {token_usage['total_tokens']:,}")
    print(f"使用的缓存tokens: {token_usage['cached_tokens']:,}")
else:
    print("\ntoken使用信息不可用。")
