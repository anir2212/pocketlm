"""Sample SFT outputs on training prompts and keep incorrect generations as DPO negatives."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import torch
import yaml

from eval.tool_eval import load_model, parse_output
from tools_api.executor import SCHEMAS, validate_target
from train.formatting import render
from train.sft import load_rows


def rejection_reason(gold: dict, output: str) -> str | None:
    predicted = parse_output(output)
    if predicted == gold:
        return None
    if predicted is None:
        return "invalid_json"
    if "refusal" in gold and "refusal" not in predicted:
        return "missing_refusal"
    if "clarification" in gold and "clarification" not in predicted:
        return "missing_clarification"
    calls = predicted.get("calls", [])
    if isinstance(calls, list) and any(
        isinstance(call, dict) and (
            not isinstance(call.get("tool"), str) or call["tool"] not in SCHEMAS
        ) for call in calls
    ):
        return "hallucinated_tool"
    if not validate_target(predicted):
        return "invalid_arguments"
    return "wrong_valid_output"


@torch.no_grad()
def build(config_path: Path, adapter: Path, output: Path,
          max_prompts: int | None = None) -> dict:
    config = yaml.safe_load(config_path.read_text())
    rng = random.Random(config["seed"])
    torch.manual_seed(config["seed"])
    rows = [row for row in load_rows(Path("data/processed/train.jsonl"))
            if row["source"] == "synthetic_device"]
    rng.shuffle(rows)
    rows = rows[:max_prompts or config["sample_prompts"]]
    model, tokenizer, device = load_model(adapter)
    pairs = []
    reasons = Counter()
    for index, row in enumerate(rows):
        prompt = render(row, tokenizer, include_answer=False)
        encoded = tokenizer(prompt, return_tensors="pt")
        ids = encoded["input_ids"].to(device)
        generated = model.generate(
            ids, attention_mask=encoded["attention_mask"].to(device),
            max_new_tokens=config["max_new_tokens"], do_sample=True,
            temperature=config["temperature"], top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
        rejected = tokenizer.decode(generated[0, ids.shape[1]:],
                                    skip_special_tokens=True).strip()
        reason = rejection_reason(row["target"], rejected)
        if reason and rejected:
            reasons[reason] += 1
            pairs.append({"id": row["id"], "prompt": prompt,
                          "chosen": json.dumps(row["target"], separators=(",", ":"),
                                               sort_keys=True),
                          "rejected": rejected, "reason": reason})
        if (index + 1) % 100 == 0:
            print(f"sampled {index + 1}/{len(rows)}, kept {len(pairs)}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(pair) + "\n" for pair in pairs))
    report = {"sft_adapter": str(adapter), "prompts_sampled": len(rows),
              "pairs_kept": len(pairs), "reasons": dict(reasons),
              "seed": config["seed"], "temperature": config["temperature"],
              "model_revision": config["model_revision"]}
    output.with_suffix(".report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/dpo.yaml"))
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/preference_pairs.jsonl"))
    parser.add_argument("--max-prompts", type=int)
    args = parser.parse_args()
    print(json.dumps(build(args.config, args.adapter, args.output, args.max_prompts), indent=2))
