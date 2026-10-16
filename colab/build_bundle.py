"""Create a small private Colab upload containing source, configs, and processed data."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = (
    "pyproject.toml", "uv.lock", "Makefile", "README.md", "RESULTS.md",
    "configs", "tools_api", "train", "eval", "quant", "demo", "scratch",
    "colab/checkpoint_bundle.py",
    "data/build_mix.py", "data/build_synthetic.py", "data/clean.py",
    "data/dedup.py", "data/stats.py", "data/SOURCES.md", "data/STATS.md",
    "data/processed/train.jsonl", "data/processed/test.jsonl",
    "data/processed/build_report.json",
    # The completed local rank-8 baseline prevents an unnecessary GPU retrain.
    "runs/sft_r8_lr1e4_2k/adapter/adapter_config.json",
    "runs/sft_r8_lr1e4_2k/adapter/adapter_model.safetensors",
    "runs/sft_r8_lr1e4_2k/metrics.json",
    "runs/sft_r8_lr1e4_2k/run_config.yaml",
    "runs/sft_r8_lr1e4_2k/source_config.yaml",
)


def build(output: Path) -> tuple[int, int]:
    files = []
    missing = []
    for name in INCLUDE:
        path = ROOT / name
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(candidate for candidate in path.rglob("*") if candidate.is_file() and
                         "__pycache__" not in candidate.parts)
        else:
            missing.append(name)
    if missing:
        raise FileNotFoundError("Colab bundle inputs missing: " + ", ".join(missing))
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(files)):
            archive.write(path, Path("pocketlm") / path.relative_to(ROOT))
    return len(files), output.stat().st_size


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("colab/pocketlm_colab_bundle.zip"))
    args = parser.parse_args()
    count, size = build(args.output)
    print(f"wrote {count} files, {size:,} bytes to {args.output}")
