#!/bin/bash
set -ex

# --- 配置区 ---
# 默认 Ollama 地址和模型名称
OLLAMA_URL=${OLLAMA_URL:-"http://localhost:11434"}
MODEL_NAME=${MODEL_NAME:-"qwen3.5:9b"} # 你可以根据实际下载的模型修改，如 qwen2, mistral 等

output_folder=${1:-"results"}
dataset_path=${2:-"dataset/locomo10.json"}

if [ ! -d "$output_folder" ]; then
    mkdir -p "$output_folder"
fi

# --- Token 计数说明 ---
# 注意：Ollama 原生并不直接提供全局的 "/token_count" 路由。
# 通常 token 信息包含在每个 /api/generate 的返回 JSON 中（prompt_eval_count 和 eval_count）。
# 如果你的 Python 脚本没有内部处理 Ollama 的 token 统计，这两行 curl 可能会失效。
# 这里我们假设你可能通过某种方式模拟或忽略它，或者你的 Python 脚本会自行记录。

echo "Using Ollama at $OLLAMA_URL with model $MODEL_NAME"

export OLLAMA_KEEP_ALIVE=${OLLAMA_KEEP_ALIVE:-"-1"}

# 如果你之前有重置 token 的逻辑，在 Ollama 中通常不需要，或者需要通过 Python 逻辑实现
# curl -s -X POST "$OLLAMA_URL/reset_token_count" || echo "Warning: Token reset not available"

# --- 运行实验 ---
# 提示：确保你的 run_experiments.py 已经修改为使用 Ollama Python 库或指向 Ollama 的 API 端口
python3 run_experiments.py \
    --method add \
    --output_folder "$output_folder" \
    --dataset_path "$dataset_path" \
    --model "$MODEL_NAME" \
    --base_url "$OLLAMA_URL"

python3 run_experiments.py \
    --method search \
    --output_folder "$output_folder" \
    --top_k 30 \
    --dataset_path "$dataset_path" \
    --model "$MODEL_NAME" \
    --base_url "$OLLAMA_URL" \
    --ollama_keep_alive "$OLLAMA_KEEP_ALIVE"

# --- 评估与评分 ---
python3 evals.py \
    --input_file "./$output_folder/results.json" \
    --output_file "./$output_folder/evaluation_metrics.json"

echo "Ollama Experiment Report" > "./$output_folder/evaluation.txt"
echo "Model: $MODEL_NAME" >> "./$output_folder/evaluation.txt"
echo "--------------------------" >> "./$output_folder/evaluation.txt"

python3 generate_scores.py "$output_folder" >> "./$output_folder/evaluation.txt"

cat "./$output_folder/evaluation.txt"
