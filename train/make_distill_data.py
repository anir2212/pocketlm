"""Generate schema-valid sequence-level distillation targets from the tuned teacher."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from eval.tool_eval import generate, load_model, parse_output
from tools_api.executor import validate_target
from train.sft import load_rows


@torch.no_grad()
def build(adapter: Path, output: Path, max_prompts: int = 1000, seed: int = 19) -> dict:
    rows = [row for row in load_rows(Path("data/processed/train.jsonl"))
            if row["source"] == "synthetic_device"]
    random.Random(seed).shuffle(rows)
    rows = rows[:max_prompts]
    model, tokenizer, device = load_model(adapter)
    kept = []
    exact = 0
    for index, row in enumerate(rows):
        raw = generate(model, tokenizer, device, row)
        parsed = parse_output(raw)
        if parsed is not None and validate_target(parsed):
            exact += parsed == row["target"]
            kept.append({"id": row["id"], "row": row, "teacher_output": parsed})
        if (index + 1) % 100 == 0:
            print(f"teacher generated {index + 1}/{len(rows)}; schema-valid {len(kept)}",
                  flush=True)
    if len(kept) < 2:
        raise RuntimeError("too few schema-valid teacher sequences to distil")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(item) + "\n" for item in kept))
    report = {"teacher_adapter": str(adapter), "prompts_sampled": len(rows),
              "schema_valid_kept": len(kept), "exact_gold_agreement": exact,
              "seed": seed, "source": "training split only"}
    output.with_suffix(".report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/distill_teacher.jsonl"))
    parser.add_argument("--max-prompts", type=int, default=1000)
    args = parser.parse_args()
    print(json.dumps(build(args.adapter, args.output, args.max_prompts), indent=2))
