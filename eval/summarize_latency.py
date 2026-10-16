"""Create a compact, source-hashed summary of llama.cpp benchmark JSON files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def summarize(inputs: list[Path], output: Path) -> dict:
    rows = []
    for path in inputs:
        raw = path.read_bytes()
        report = json.loads(raw)
        prompt = next(run for run in report["llama_bench"]
                      if run["n_prompt"] == 512 and run["n_gen"] == 0)
        generation = next(run for run in report["llama_bench"]
                          if run["n_prompt"] == 0 and run["n_gen"] == 128)
        rows.append({
            "source": str(path), "source_sha256": hashlib.sha256(raw).hexdigest(),
            "model": report["model"], "grammar": report["grammar"],
            "hardware": report["hardware"], "file_size_bytes": report["file_size_bytes"],
            "prompt_tokens_per_second": prompt["avg_ts"],
            "generation_tokens_per_second": generation["avg_ts"],
            "bench_repetitions": len(generation["samples_ts"]),
            "time_to_first_token_seconds_median_or_one_request": report[
                "time_to_first_token_seconds"],
            "end_to_end_seconds_median_or_one_request": report["end_to_end_seconds"],
            "generated_tokens_first_request": report["native_timings"].get("predicted_n"),
            "request_repetitions": report.get("request_repetitions", 1),
            "time_to_first_token_p95_seconds": report.get(
                "time_to_first_token_p95_seconds"),
            "end_to_end_p95_seconds": report.get("end_to_end_p95_seconds"),
            "request_generation_tokens_per_second_median": report.get(
                "request_generation_tokens_per_second_median"),
            "peak_server_rss_bytes_sampled": report["peak_server_rss_bytes_sampled"],
        })
    result = {"rows": rows,
              "note": "llama-bench throughput has fixed lengths and does not apply "
                      "the grammar. HTTP values describe the sample count in each row; "
                      "legacy rows contain one request."}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(args.inputs, args.output), indent=2))
