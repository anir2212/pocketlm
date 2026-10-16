"""Build the measured GGUF quality, speed, and memory table and Pareto plot."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path


def read_row(name: str, grammar: str, directory: Path) -> dict:
    label = f"{name}_{grammar}"
    quality = json.loads((directory / f"gguf_eval_{label}.json").read_text())
    latency = json.loads((directory / f"latency_{label}.json").read_text())
    bench = latency["llama_bench"]
    prompt = next((run["avg_ts"] for run in bench if run["n_prompt"] == 512
                   and run["n_gen"] == 0), None)
    generation = next((run["avg_ts"] for run in bench if run["n_gen"] == 128
                       and run["n_prompt"] == 0), None)
    return {
        "format": name, "grammar": grammar,
        "examples": quality["metrics"]["examples"],
        "json_validity": quality["metrics"]["json_validity"],
        "full_call_exact_match": quality["metrics"]["full_call_exact_match"],
        "accuracy_ci95": quality["confidence_intervals_95pct"]["full_call_exact_match"],
        "prompt_tokens_per_second": prompt,
        "llama_bench_generation_tokens_per_second": generation,
        "request_generation_tokens_per_second": latency[
            "request_generation_tokens_per_second_median"],
        "time_to_first_token_seconds": latency["time_to_first_token_seconds"],
        "time_to_first_token_p95_seconds": latency["time_to_first_token_p95_seconds"],
        "request_repetitions": latency["request_repetitions"],
        "peak_server_rss_mib": round(latency["peak_server_rss_bytes_sampled"] / 2**20, 1),
        "file_mib": round(latency["file_size_bytes"] / 2**20, 1),
    }


def build(directory: Path, output: Path, chart: Path,
          formats: list[str]) -> dict:
    cache = Path(tempfile.gettempdir()) / "pocketlm_matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("Install plot dependencies with uv sync --extra plot") from error

    rows = [read_row(name, grammar, directory)
            for name in formats for grammar in ("plain", "grammar")]
    fig, axes = plt.subplots(figsize=(9, 6))
    for row in rows:
        axes.scatter(row["request_generation_tokens_per_second"],
                     100 * row["full_call_exact_match"],
                     s=row["file_mib"] / 2, alpha=0.55,
                     marker="o" if row["grammar"] == "grammar" else "s")
        axes.annotate(f"{row['format']} {row['grammar']}",
                      (row["request_generation_tokens_per_second"],
                       100 * row["full_call_exact_match"]), xytext=(5, 5),
                      textcoords="offset points", fontsize=8)
    axes.set_xlabel("Median generation tokens/s (20 warmed CPU requests, grammar aware)")
    axes.set_ylabel("Full-call exact match on held-out calls (%)")
    axes.set_title("PocketLM GGUF quality–latency tradeoff (bubble size = file size)")
    axes.grid(alpha=0.25)
    fig.tight_layout()
    chart.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(chart, dpi=160)
    plt.close(fig)
    report = {"rows": rows, "chart": str(chart),
              "note": "All quality runs use the same generated holdout. Accuracy CI is "
                      "bootstrap over examples. Request speed is a median of 20 "
                      "repeated requests to one warm server, not a general device SLA. "
                      "The fixed-length llama-bench throughput omits grammar."}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, default=Path("runs"))
    parser.add_argument("--output", type=Path, default=Path("bench/quant_report.json"))
    parser.add_argument("--chart", type=Path, default=Path("bench/quant_pareto.png"))
    parser.add_argument("--formats", nargs="+", default=["F16", "Q8_0", "Q5_K_M", "Q4_K_M"])
    args = parser.parse_args()
    print(json.dumps(build(args.directory, args.output, args.chart, args.formats), indent=2))
