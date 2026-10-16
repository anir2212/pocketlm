"""Record CPU llama-bench throughput and server latency with optional grammar."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
from pathlib import Path

from transformers import AutoTokenizer

from eval.tool_eval import BASE, REVISION
from quant.runtime import GGUFServer
from train.formatting import render
from train.model_files import cached_snapshot


def benchmark(model: Path, bench_binary: Path, server_binary: Path,
              output: Path, grammar_path: Path | None = None,
              request_repetitions: int = 20) -> dict:
    if request_repetitions < 2:
        raise ValueError("at least two HTTP requests are needed for a latency distribution")
    command = [str(bench_binary), "--model", str(model), "--n-prompt", "512",
               "--n-gen", "128", "--threads", "4", "--n-gpu-layers", "0",
               "--repetitions", "3", "--output", "json"]
    bench = subprocess.run(command, text=True, capture_output=True, timeout=900, check=True)
    raw = json.loads(bench.stdout)
    rows = [json.loads(line) for line in Path("data/processed/test.jsonl").read_text().splitlines()]
    tokenizer = AutoTokenizer.from_pretrained(cached_snapshot(BASE, REVISION))
    prompt = render(rows[0], tokenizer, include_answer=False)
    grammar = grammar_path.read_text() if grammar_path else None
    with GGUFServer(server_binary, model) as server:
        warmup = server.complete(prompt, max_tokens=16)
        completions = [server.complete(prompt, max_tokens=128, grammar=grammar)
                       for _ in range(request_repetitions)]
    if any(item["ttft_seconds"] is None for item in completions):
        raise RuntimeError("an HTTP benchmark request emitted no content token")
    ttfts = [item["ttft_seconds"] for item in completions]
    elapsed = [item["elapsed_seconds"] for item in completions]
    generation_rates = [item["timings"].get("predicted_per_second")
                        for item in completions]
    rates = [rate for rate in generation_rates if rate is not None]
    request_samples = [{"ttft_seconds": item["ttft_seconds"],
                        "elapsed_seconds": item["elapsed_seconds"],
                        "peak_server_rss_bytes_sampled":
                        item["peak_server_rss_bytes_sampled"],
                        "native_timings": item["timings"]}
                       for item in completions]
    report = {
        "model": str(model), "file_size_bytes": model.stat().st_size,
        "grammar": str(grammar_path) if grammar_path else None,
        "hardware": {"device": "local CPU", "threads": 4, "gpu_offload_layers": 0,
                     "platform": platform.platform(), "machine": platform.machine(),
                     "processor": platform.processor(), "logical_cpus": os.cpu_count()},
        "llama_bench_command": command, "llama_bench": raw,
        "warmup_elapsed_seconds": warmup["elapsed_seconds"],
        "request_repetitions": request_repetitions, "request_samples": request_samples,
        "time_to_first_token_seconds": statistics.median(ttfts),
        "time_to_first_token_p95_seconds": sorted(ttfts)[
            (95 * len(ttfts) + 99) // 100 - 1],
        "end_to_end_seconds": statistics.median(elapsed),
        "end_to_end_p95_seconds": sorted(elapsed)[
            (95 * len(elapsed) + 99) // 100 - 1],
        "request_generation_tokens_per_second_median": (
            statistics.median(rates) if rates else None),
        "peak_server_rss_bytes_sampled": max(
            item["peak_server_rss_bytes_sampled"] for item in completions),
        "native_timings": completions[0]["timings"],
        "sample_output": completions[0]["content"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--bench-binary", type=Path, required=True)
    parser.add_argument("--server-binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grammar", type=Path)
    parser.add_argument("--request-repetitions", type=int, default=20)
    args = parser.parse_args()
    result = benchmark(args.model, args.bench_binary, args.server_binary,
                       args.output, args.grammar, args.request_repetitions)
    print(json.dumps({key: value for key, value in result.items()
                      if key not in {"sample_output", "request_samples"}}, indent=2))
