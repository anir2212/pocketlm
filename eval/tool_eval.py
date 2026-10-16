"""Generate on held-out device prompts and bootstrap structured-output metrics."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from tools_api.executor import SCHEMAS, validate_target
from train.formatting import render
from train.model_files import cached_snapshot

BASE = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"


def parse_output(raw: str) -> dict | None:
    try:
        parsed = json.loads(raw.strip())
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def atoms(target: dict) -> Counter:
    result = Counter()
    for call in target.get("calls", []):
        for key, value in call.get("arguments", {}).items():
            result[call.get("tool"), key, json.dumps(value, sort_keys=True)] += 1
    return result


def item_metrics(row: dict, raw: str) -> dict:
    gold = row["target"]
    predicted = parse_output(raw)
    has_calls = "calls" in gold
    kinds = {
        "json_valid": predicted is not None,
        "target_valid": predicted is not None and validate_target(predicted),
        "full_exact": predicted == gold,
        "tool_exact": (has_calls and predicted is not None and isinstance(
            predicted.get("calls"), list) and
            all(isinstance(call, dict) for call in predicted["calls"]) and
            [call.get("tool") for call in predicted["calls"]] ==
            [call["tool"] for call in gold["calls"]]),
        "hallucinated_tool": False,
        "correct_refusal": "refusal" in gold and predicted == gold,
        "correct_clarification": "clarification" in gold and predicted == gold,
        "has_calls": has_calls,
        "is_refusal": "refusal" in gold,
        "is_clarification": "clarification" in gold,
        "argument_tp": 0, "argument_pred": 0, "argument_gold": sum(atoms(gold).values()),
    }
    if predicted is not None and isinstance(predicted.get("calls"), list):
        calls = predicted["calls"]
        kinds["hallucinated_tool"] = any(
            isinstance(call, dict) and (
                not isinstance(call.get("tool"), str) or call["tool"] not in SCHEMAS
            ) for call in calls
        )
        try:
            prediction_atoms = atoms(predicted)
            gold_atoms = atoms(gold)
            kinds["argument_tp"] = sum((prediction_atoms & gold_atoms).values())
            kinds["argument_pred"] = sum(prediction_atoms.values())
        except (TypeError, AttributeError):
            pass
    return kinds


def summarize(items: list[dict]) -> dict:
    count = len(items)
    if not count:
        raise ValueError("no evaluation items")
    def rate(field: str, denominator: str | None = None):
        relevant = [item for item in items if denominator is None or item[denominator]]
        return sum(bool(item[field]) for item in relevant) / len(relevant) if relevant else None

    tp = sum(item["argument_tp"] for item in items)
    predicted = sum(item["argument_pred"] for item in items)
    gold = sum(item["argument_gold"] for item in items)
    precision = tp / predicted if predicted else 0.0
    recall = tp / gold if gold else 0.0
    return {
        "examples": count,
        "json_validity": rate("json_valid"),
        "target_schema_validity": rate("target_valid"),
        "full_call_exact_match": rate("full_exact", "has_calls"),
        "overall_target_exact_match": rate("full_exact"),
        "tool_name_exact_match_on_calls": rate("tool_exact", "has_calls"),
        "argument_f1": 2 * precision * recall / (precision + recall)
        if precision + recall else 0.0,
        "hallucinated_tool_rate": rate("hallucinated_tool"),
        "correct_refusal_rate": rate("correct_refusal", "is_refusal"),
        "correct_clarification_rate": rate("correct_clarification", "is_clarification"),
    }


def bootstrap(items: list[dict], repetitions: int = 1000, seed: int = 17) -> dict:
    rng = random.Random(seed)
    keys = [key for key in summarize(items) if key != "examples"]
    distributions = {key: [] for key in keys}
    for _ in range(repetitions):
        sample = [rng.choice(items) for _ in items]
        scores = summarize(sample)
        for key in keys:
            if scores[key] is not None:
                distributions[key].append(scores[key])
    result = {}
    for key, values in distributions.items():
        values.sort()
        result[key] = [values[int(0.025 * len(values))],
                       values[min(len(values) - 1, int(0.975 * len(values)))]] if values else None
    return result


def load_model(adapter: Path | None, model_id: str = BASE, revision: str = REVISION):
    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    snapshot = cached_snapshot(model_id, revision)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        snapshot, local_files_only=True,
        dtype=torch.float16 if device != "cpu" else torch.float32, low_cpu_mem_usage=True,
    )
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    return model.to(device).eval(), tokenizer, device


@torch.no_grad()
def generate(model, tokenizer, device: str, row: dict, max_new_tokens: int = 160) -> str:
    encoded = tokenizer(render(row, tokenizer, include_answer=False), return_tensors="pt")
    ids = encoded["input_ids"].to(device)
    mask = encoded["attention_mask"].to(device)
    output = model.generate(ids, attention_mask=mask, max_new_tokens=max_new_tokens,
                            do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output[0, ids.shape[1]:], skip_special_tokens=True).strip()


def evaluate(input_path: Path, output_path: Path, adapter: Path | None,
             max_examples: int | None = None,
             model_id: str = BASE, revision: str = REVISION) -> dict:
    rows = [json.loads(line) for line in input_path.read_text().splitlines() if line]
    rows = [row for row in rows if row["source"] == "synthetic_device"]
    if max_examples:
        rows = rows[:max_examples]
    model, tokenizer, device = load_model(adapter, model_id, revision)
    predictions = []
    metrics = []
    for row in rows:
        raw = generate(model, tokenizer, device, row)
        predictions.append({"id": row["id"], "gold": row["target"], "output": raw})
        metrics.append(item_metrics(row, raw))
    result = {"model": model_id, "revision": revision,
              "adapter": str(adapter) if adapter else None, "device": device,
              "metrics": summarize(metrics), "confidence_intervals_95pct": bootstrap(metrics),
              "predictions": predictions}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/processed/test.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("runs/base_tool_eval.json"))
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--model-id", default=BASE)
    parser.add_argument("--revision", default=REVISION)
    args = parser.parse_args()
    result = evaluate(args.input, args.output, args.adapter, args.max_examples,
                      args.model_id, args.revision)
    summary = {key: value for key, value in result.items() if key != "predictions"}
    print(json.dumps(summary, indent=2))
