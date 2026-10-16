"""Bounded ARC Easy and HellaSwag checks for base and adapter forgetting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from eval.tool_eval import BASE, REVISION
from train.model_files import cached_snapshot


def evaluate(output: Path, adapter: Path | None, limit: int = 100) -> dict:
    if limit <= 0:
        raise ValueError("limit must be positive")
    try:
        import lm_eval
        from lm_eval.utils import handle_non_serializable
    except ImportError as error:
        raise RuntimeError(
            "Install the general-eval extra: uv sync --extra general-eval"
        ) from error

    device = "cuda:0" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    model_args = {"pretrained": cached_snapshot(BASE, REVISION),
                  "dtype": "float16" if device != "cpu" else "float32"}
    if adapter:
        model_args["peft"] = str(adapter.resolve())
    result = lm_eval.simple_evaluate(
        model="hf", model_args=model_args, tasks=["arc_easy", "hellaswag"],
        num_fewshot=0, batch_size=1, device=device, limit=limit, log_samples=False,
        apply_chat_template=True,
    )
    if result is None:
        raise RuntimeError("lm-eval returned no results")
    report = {
        "model": BASE, "revision": REVISION,
        "adapter": str(adapter) if adapter else None,
        "device": device, "limit_per_task": limit,
        "num_fewshot": 0, "results": result["results"],
        "versions": result.get("versions"), "n_samples": result.get("n-samples"),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, default=handle_non_serializable, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("runs/base_general_eval.json"))
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.output, args.adapter, args.limit), indent=2))
