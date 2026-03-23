import argparse
import os
import numpy as np

from methods.add import MemoryADD
from methods.search import MemorySearch


def main():
    parser = argparse.ArgumentParser(description="Run memory experiments")
    parser.add_argument("--method", choices=["add", "search"], default="add", help="Method to use")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Chunk size for processing")
    parser.add_argument("--output_folder", type=str, default="results/", help="Output path for results")
    parser.add_argument("--top_k", type=int, default=30, help="Number of top memories to retrieve")
    parser.add_argument("--filter_memories", action="store_true", default=False, help="Whether to filter memories")
    parser.add_argument("--is_graph", action="store_true", default=False, help="Whether to use graph-based search")
    parser.add_argument("--num_chunks", type=int, default=1, help="Number of chunks to process")
    parser.add_argument("--dataset-path", type=str, default=None, help="Path to LOCOMO dataset json")
    parser.add_argument("--api-base-url", type=str, default="http://127.0.0.1:8000", help="STmemory LOCOMO server base URL")
    parser.add_argument("--model", type=str, default="MiniMax-M2.7", help="LLM model used for answering/judging")
    parser.add_argument("--minimax-api-key", type=str, default=os.getenv("MINIMAX_API_KEY") or os.getenv("ANTHROPIC_API_KEY") or "", help="MiniMax/Anthropic API key")
    parser.add_argument("--llm-provider", type=str, default=os.getenv("LOCOMO_LLM_PROVIDER") or "", help="LLM provider: anthropic/minimax or openai_compat/ollama")
    parser.add_argument("--openai-base-url", type=str, default=os.getenv("OPENAI_BASE_URL") or "", help="OpenAI-compatible base URL (e.g. http://127.0.0.1:11434/v1)")
    parser.add_argument("--openai-api-key", type=str, default=os.getenv("OPENAI_API_KEY") or "", help="OpenAI-compatible API key (can be empty for local)")
    parser.add_argument("--use-mock-llm", action="store_true", default=False, help="Use mock LLM instead of calling provider")
    parser.add_argument("--workers", type=int, default=1, help="Worker threads for add/search")
    parser.add_argument("--subset-indices", type=str, default="", help="Comma-separated subset indices")
    parser.add_argument("--max-conversations", type=int, default=0, help="Max conversations to process (0 means no limit)")
    parser.add_argument("--max-qa", type=int, default=0, help="Max QA per conversation (0 means no limit)")
    parser.add_argument("--metrics-path", type=str, default=None, help="Path to append provider metrics jsonl")
    parser.add_argument("--llm-log-path", type=str, default=None, help="Path to append LLM prompt/response jsonl")
    parser.add_argument("--llm-request-timeout-s", type=float, default=60.0, help="LLM request timeout (seconds)")
    parser.add_argument("--minimax-max-tokens", type=int, default=512, help="Max tokens for answer generation")
    parser.add_argument("--minimax-temperature", type=float, default=0.0, help="Temperature for answer generation")
    parser.add_argument("--minimax-system-prompt", type=str, default="You are a helpful assistant.", help="System prompt for answer generation")
    parser.add_argument("--add-future-timeout-s", type=float, default=300.0, help="Timeout for add worker futures (seconds)")

    args = parser.parse_args()

    dataset_path = args.dataset_path or "dataset/chronicles10.json"

    print(f"Running experiments with chunk size: {args.chunk_size}")
    print(f"Using dataset: {dataset_path}")

    if args.method == "add":
        memory_manager = MemoryADD(
            api_base_url=str(args.api_base_url),
            data_path=dataset_path,
            is_graph=bool(args.is_graph),
            add_workers=int(args.workers),
            subset_indices=str(args.subset_indices),
            max_conversations=int(args.max_conversations),
            add_future_timeout_s=float(args.add_future_timeout_s),
        )
        memory_manager.process_all_conversations()
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
            api_base_url=str(args.api_base_url),
            model=str(args.model),
            minimax_api_key=str(args.minimax_api_key),
            llm_provider=str(args.llm_provider),
            openai_base_url=str(args.openai_base_url),
            openai_api_key=str(args.openai_api_key),
            use_mock_llm=bool(args.use_mock_llm),
            metrics_path=(str(args.metrics_path) if args.metrics_path else None),
            llm_log_path=(str(args.llm_log_path) if args.llm_log_path else None),
            llm_request_timeout_s=float(args.llm_request_timeout_s),
            minimax_max_tokens=int(args.minimax_max_tokens),
            minimax_temperature=float(args.minimax_temperature),
            minimax_system_prompt=str(args.minimax_system_prompt),
            max_qa=int(args.max_qa),
            search_workers=int(args.workers),
            subset_indices=str(args.subset_indices),
            max_conversations=int(args.max_conversations),
        )
        memory_searcher.process_data_file(dataset_path)
        server_time = memory_searcher.server_execution_time
        request_count = memory_searcher.request_count
        request_times = memory_searcher.request_times

    p95_latency = np.percentile(request_times, 95)

    with open(os.path.join(args.output_folder, "evaluation.txt"), "a") as f:
        f.write(f"action: {args.method}\n")
        f.write(f"  Total server execution time: {server_time:.4f} seconds\n")
        f.write(f"  Total requests: {request_count}\n")
        f.write(f"  Average request time: {(server_time / request_count):.4f} seconds\n")
        f.write(f"  P95 request time: {p95_latency:.4f} seconds\n")


if __name__ == "__main__":
    main()
