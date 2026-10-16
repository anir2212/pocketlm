"""Resume the full GGUF quality and latency matrix without repeating valid runs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

FORMATS = ("F16", "Q8_0", "Q5_K_M", "Q4_K_M")
MODES = ("plain", "grammar")


def complete_quality(path: Path, model: Path, grammar: Path | None,
                     expected_examples: int,
                     expected_rows: list[dict] | None = None) -> bool:
    try:
        report = json.loads(path.read_text())
        return (Path(report["model"]).resolve() == model.resolve()
                and report["grammar"] == (str(grammar) if grammar else None)
                and report["metrics"]["examples"] == expected_examples
                and len(report["predictions"]) == expected_examples
                and path.stat().st_mtime_ns >= model.stat().st_mtime_ns
                and (grammar is None or path.stat().st_mtime_ns >= grammar.stat().st_mtime_ns)
                and (expected_rows is None or all(
                    prediction["id"] == row["id"]
                    and prediction["gold"] == row["target"]
                    for prediction, row in zip(report["predictions"], expected_rows,
                                               strict=True))))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def complete_latency(path: Path, model: Path, grammar: Path | None) -> bool:
    try:
        report = json.loads(path.read_text())
        bench = report["llama_bench"]
        return (Path(report["model"]).resolve() == model.resolve()
                and report["grammar"] == (str(grammar) if grammar else None)
                and report["file_size_bytes"] == model.stat().st_size
                and path.stat().st_mtime_ns >= model.stat().st_mtime_ns
                and (grammar is None or path.stat().st_mtime_ns >= grammar.stat().st_mtime_ns)
                and any(row["n_prompt"] == 512 and row["n_gen"] == 0 for row in bench)
                and any(row["n_prompt"] == 0 and row["n_gen"] == 128 for row in bench)
                and report["request_repetitions"] >= 20
                and len(report["request_samples"]) == report["request_repetitions"]
                and report["time_to_first_token_seconds"] > 0)
    except (OSError, ValueError, KeyError, TypeError):
        return False


def run(directory: Path, llama_cpp: Path, phase: str = "all") -> None:
    if phase not in {"all", "quality", "latency"}:
        raise ValueError(f"unsupported phase: {phase}")
    server = llama_cpp / "build/bin/llama-server"
    bench = llama_cpp / "build/bin/llama-bench"
    test_rows = [json.loads(line) for line in
                 (directory / "data/processed/test.jsonl").read_text().splitlines()]
    held_out = [row for row in test_rows if row["source"] == "synthetic_device"]
    expected_examples = len(held_out)
    if not expected_examples:
        raise ValueError("the held-out device set is empty")
    for executable in (server, bench):
        if not executable.is_file():
            raise FileNotFoundError(executable)
    for format_name in FORMATS:
        model = directory / "artifacts" / f"model-{format_name}.gguf"
        if not model.is_file():
            raise FileNotFoundError(model)
        for mode in MODES:
            label = f"{format_name}_{mode}"
            grammar_path = Path("quant/grammar.gbnf") if mode == "grammar" else None
            grammar = (["--grammar", str(grammar_path)] if grammar_path else [])
            quality = directory / "runs" / f"gguf_eval_{label}.json"
            latency = directory / "runs" / f"latency_{label}.json"
            if phase in {"all", "quality"}:
                if complete_quality(quality, model, grammar_path, expected_examples,
                                    held_out):
                    print(f"Quality {label}: already complete", flush=True)
                else:
                    print(f"Quality {label}: running {expected_examples} examples", flush=True)
                    subprocess.run([sys.executable, "-m", "eval.gguf_eval", "--model",
                                    str(model), "--server-binary", str(server),
                                    "--max-examples", str(expected_examples), *grammar,
                                    "--output", str(quality)], cwd=directory, check=True)
            if phase in {"all", "latency"}:
                if not complete_quality(quality, model, grammar_path, expected_examples,
                                        held_out):
                    raise RuntimeError(f"full quality result is missing for {label}")
                if complete_latency(latency, model, grammar_path):
                    print(f"Latency {label}: already complete", flush=True)
                else:
                    print(f"Latency {label}: running", flush=True)
                    subprocess.run([sys.executable, "-m", "eval.latency_bench", "--model",
                                    str(model), "--bench-binary", str(bench),
                                    "--server-binary", str(server), *grammar,
                                    "--output", str(latency)], cwd=directory, check=True)
    if phase != "quality":
        subprocess.run([sys.executable, "-m", "eval.report"], cwd=directory, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama-cpp-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("all", "quality", "latency"), default="all")
    args = parser.parse_args()
    run(Path.cwd(), args.llama_cpp_dir.resolve(), args.phase)
