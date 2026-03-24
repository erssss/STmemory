import argparse
import os
import numpy as np

from methods.add import MemoryADD
from methods.search import MemorySearch


def _flatten_ollama_token_counts(results_by_conversation):
    prompt_counts = []
    completion_counts = []
    if not results_by_conversation:
        return prompt_counts, completion_counts
    for _, items in results_by_conversation.items():
        if not items:
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            prompt_counts.append(int(item.get("ollama_prompt_eval_count") or 0))
            completion_counts.append(int(item.get("ollama_eval_count") or 0))
    return prompt_counts, completion_counts


def main():
    parser = argparse.ArgumentParser(
        description="Run memory experiments with Ollama",
    )
    parser.add_argument(
        "--method",
        choices=["add", "search"],
        default="add",
        help="Method to use",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=1000,
        help="Chunk size for processing",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="results/",
        help="Output path for results",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=30,
        help="Number of top memories to retrieve",
    )
    parser.add_argument(
        "--filter_memories",
        action="store_true",
        default=False,
        help="Whether to filter memories",
    )
    parser.add_argument(
        "--is_graph",
        action="store_true",
        default=False,
        help="Whether to use graph-based search",
    )
    parser.add_argument(
        "--num_chunks",
        type=int,
        default=1,
        help="Number of chunks to process",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="dataset/locomo10.json",
        help="Path to the dataset JSON",
    )

    # --- 新增 Ollama 相关参数 ---
    parser.add_argument(
        "--model",
        type=str,
        default="qwen3.5:9b",
        help="Ollama model name",
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default=None,
        help="(Deprecated) Ollama base URL. Use --ollama_base_url instead.",
    )
    parser.add_argument(
        "--ollama_base_url",
        type=str,
        default="http://localhost:11434",
        help="Ollama API base URL (native, e.g. http://localhost:11434)",
    )
    parser.add_argument(
        "--local_db_path",
        type=str,
        default=os.getenv("LOCAL_DB_PATH"),
        help="Local SQLite path for storing memories",
    )
    parser.add_argument(
        "--ollama_embedding_model",
        type=str,
        default=os.getenv("OLLAMA_EMBEDDING_MODEL") or "nomic-embed-text",
        help="Ollama embedding model name (for local SDK mode)",
    )
    parser.add_argument(
        "--ollama_chat_timeout",
        type=int,
        default=600,
        help="Timeout (seconds) for Ollama /api/chat requests",
    )
    parser.add_argument(
        "--ollama_num_predict",
        type=int,
        default=128,
        help="Max tokens to generate for Ollama /api/chat (num_predict)",
    )
    parser.add_argument(
        "--context_k",
        type=int,
        default=10,
        help="How many retrieved memories per speaker to include in the answer prompt",
    )
    parser.add_argument(
        "--max_conversations",
        type=int,
        default=0,
        help="Limit number of conversations to process (0 means all)",
    )
    parser.add_argument(
        "--max_questions_per_conversation",
        type=int,
        default=0,
        help="Limit number of questions per conversation (0 means all)",
    )
    parser.add_argument(
        "--ollama_keep_alive",
        type=str,
        default= "-1",
        help="Ollama keep_alive value (e.g. 30m, 5m, -1)",
    )
    parser.add_argument(
        "--full_context",
        action="store_true",
        default=False,
        help="Run without memory retrieval and use full conversation context",
    )
    parser.add_argument(
        "--anti_pollution",
        action="store_true",
        default=True,
        help="Enable anti-pollution memory features in local SDK mode",
    )
    parser.add_argument(
        "--isolation_mode",
        type=str,
        default="strict",
        help="Isolation mode for local SDK mode (strict or compat)",
    )

    args = parser.parse_args()

    if args.base_url:
        args.ollama_base_url = args.base_url

    dataset_path = args.dataset_path
    os.makedirs(args.output_folder, exist_ok=True)
    if not args.local_db_path:
        args.local_db_path = os.path.join(args.output_folder, "stmem_locomo.db")
    args.memory_base_url = "local"

    print(f"Running experiments with method: {args.method}")
    print(f"Using Ollama Model: {args.model} at {args.ollama_base_url}")
    print(f"Using Memory (local SDK): {args.local_db_path}")
    print(f"Using dataset: {dataset_path}")

    # 这里的初始化需要确保 MemoryADD 和 MemorySearch 的 __init__ 方法能够接收 model 和 base_url
    if args.method == "add":
        memory_manager = MemoryADD(
            data_path=dataset_path,
            is_graph=args.is_graph,
            model=args.model,
            api_base_url=args.memory_base_url,
            local_db_path=args.local_db_path,
            ollama_base_url=args.ollama_base_url,
            ollama_embedding_model=args.ollama_embedding_model,
            anti_pollution=args.anti_pollution,
            isolation_mode=args.isolation_mode,
        )
        print("process_all_conversations")
        memory_manager.process_all_conversations()
        print("server_execution_time")
        server_time = memory_manager.server_execution_time
        request_count = memory_manager.request_count
        request_times = memory_manager.request_times

    elif args.method == "search":
        output_file_path = os.path.join(
            args.output_folder,
            "results.json",
        )
        memory_searcher = MemorySearch(
            output_file_path,
            args.top_k,
            args.filter_memories,
            args.is_graph,
            model=args.model,
            api_base_url=args.memory_base_url,
            openai_base_url=f"{args.ollama_base_url.rstrip('/')}/v1",
            local_db_path=args.local_db_path,
            ollama_base_url=args.ollama_base_url,
            ollama_embedding_model=args.ollama_embedding_model,
            ollama_chat_timeout=args.ollama_chat_timeout,
            ollama_num_predict=args.ollama_num_predict,
            ollama_keep_alive=args.ollama_keep_alive,
            context_k=args.context_k,
            max_conversations=args.max_conversations,
            max_questions_per_conversation=args.max_questions_per_conversation,
            use_full_context=args.full_context,
            anti_pollution=args.anti_pollution,
            isolation_mode=args.isolation_mode,
        )
        memory_searcher.process_data_file(dataset_path)
        server_time = memory_searcher.server_execution_time
        request_count = memory_searcher.request_count
        request_times = memory_searcher.request_times
        ollama_chat_times = getattr(memory_searcher, "ollama_request_times", [])
        ollama_prompt_eval_counts, ollama_eval_counts = _flatten_ollama_token_counts(
            getattr(memory_searcher, "results", None)
        )

    # 计算 P95 延迟（增加容错处理，防止 request_times 为空）
    if request_times:
        p95_latency = np.percentile(request_times, 95)
        avg_time = server_time / request_count
    else:
        p95_latency = 0
        avg_time = 0

    # 写入评估结果
    os.makedirs(args.output_folder, exist_ok=True)
    with open(os.path.join(args.output_folder, "evaluation.txt"), "a") as f:
        f.write(f"\n--- Experiment: {args.method} (Model: {args.model}) ---\n")
        f.write(f"  Total server execution time: {server_time:.4f} seconds\n")
        f.write(f"  Total requests: {request_count}\n")
        f.write(f"  Average request time: {avg_time:.4f} seconds\n")
        f.write(f"  P95 request time: {p95_latency:.4f} seconds\n")
        if args.method == "search":
            f.write(f"  Answer prompt context_k: {args.context_k}\n")
            f.write(f"  Ollama chat timeout: {args.ollama_chat_timeout} seconds\n")
            f.write(f"  Ollama num_predict: {args.ollama_num_predict}\n")
            f.write(f"  Ollama keep_alive: {args.ollama_keep_alive}\n")
            if ollama_chat_times:
                f.write(
                    f"  Ollama chat avg time: {float(np.mean(ollama_chat_times)):.4f} seconds\n"
                )
                f.write(
                    f"  Ollama chat P95 time: {float(np.percentile(ollama_chat_times, 95)):.4f} seconds\n"
                )
            prompt_tokens = np.array(ollama_prompt_eval_counts or [], dtype=float)
            completion_tokens = np.array(ollama_eval_counts or [], dtype=float)
            if prompt_tokens.size and completion_tokens.size:
                total_tokens = prompt_tokens + completion_tokens
                f.write(f"  Ollama chat questions: {int(total_tokens.size)}\n")
                f.write(
                    f"  Ollama chat total prompt tokens: {int(prompt_tokens.sum())}\n"
                )
                f.write(
                    f"  Ollama chat total completion tokens: {int(completion_tokens.sum())}\n"
                )
                f.write(f"  Ollama chat total tokens: {int(total_tokens.sum())}\n")
                f.write(
                    f"  Ollama chat avg total tokens: {float(np.mean(total_tokens)):.2f}\n"
                )
                f.write(
                    f"  Ollama chat P95 total tokens: {float(np.percentile(total_tokens, 95)):.2f}\n"
                )
            else:
                f.write("  Ollama chat questions: 0\n")
                f.write("  Ollama chat total prompt tokens: 0\n")
                f.write("  Ollama chat total completion tokens: 0\n")
                f.write("  Ollama chat total tokens: 0\n")


if __name__ == "__main__":
    main()
