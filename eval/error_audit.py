"""Audit the same held-out requests across every saved GGUF quality run."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from eval.tool_eval import parse_output
from tools_api.executor import validate_target

FORMATS = ("F16", "Q8_0", "Q5_K_M", "Q4_K_M")
MODES = ("plain", "grammar")


def _kind(target: dict) -> str:
    return next(key for key in ("calls", "clarification", "refusal") if key in target)


def _stratum(row: dict) -> str:
    target = row["target"]
    if "calls" not in target:
        return _kind(target)
    calls = target["calls"]
    return calls[0]["tool"] if len(calls) == 1 else "multi_call"


def classify(gold: dict, raw: str) -> str:
    """Return one exclusive failure cause in order of earliest contract violation."""
    predicted = parse_output(raw)
    if predicted is None:
        return "invalid_json_object"
    if not validate_target(predicted):
        return "invalid_target_schema"
    if predicted == gold:
        return "exact"
    if _kind(predicted) != _kind(gold):
        return "wrong_response_kind"
    if "calls" in gold and [call["tool"] for call in predicted["calls"]] != [
        call["tool"] for call in gold["calls"]
    ]:
        return "wrong_tool_sequence"
    return "wrong_arguments_or_payload"


def audit(input_path: Path, runs_dir: Path, output_path: Path,
          formats: tuple[str, ...] = FORMATS) -> dict:
    rows = [json.loads(line) for line in input_path.read_text().splitlines() if line]
    rows = [row for row in rows if row["source"] == "synthetic_device"]
    ids = [row["id"] for row in rows]
    if not rows or len(set(ids)) != len(ids):
        raise ValueError("holdout must have nonempty, unique device IDs")

    result = {
        "input": {"path": str(input_path), "sha256": hashlib.sha256(
            input_path.read_bytes()).hexdigest(), "examples": len(rows)},
        "category_order": ["invalid_json_object", "invalid_target_schema",
                           "wrong_response_kind", "wrong_tool_sequence",
                           "wrong_arguments_or_payload", "exact"],
        "modes": {},
    }
    for name in formats:
        for mode in MODES:
            label = f"{name}_{mode}"
            path = runs_dir / f"gguf_eval_{label}.json"
            raw = path.read_bytes()
            report = json.loads(raw)
            predictions = report["predictions"]
            if len(predictions) != len(rows) or [p["id"] for p in predictions] != ids:
                raise ValueError(f"{label}: prediction IDs differ from holdout")
            if any(p["gold"] != row["target"] for p, row in zip(
                predictions, rows, strict=True
            )):
                raise ValueError(f"{label}: gold targets differ from holdout")
            if Path(report["model"]).name != f"model-{name}.gguf" or bool(
                report["grammar"]
            ) != (mode == "grammar"):
                raise ValueError(f"{label}: model or grammar metadata differs")
            counts: Counter[str] = Counter()
            strata: dict[str, Counter[str]] = defaultdict(Counter)
            example_ids: dict[str, list[str]] = defaultdict(list)
            for row, prediction in zip(rows, predictions, strict=True):
                category = classify(row["target"], prediction["content"])
                counts[category] += 1
                strata[_stratum(row)][category] += 1
                if category != "exact" and len(example_ids[category]) < 5:
                    example_ids[category].append(row["id"])
            result["modes"][label] = {
                "path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
                "counts": dict(counts),
                "strata": {key: {"examples": sum(values.values()),
                                  "exact": values["exact"],
                                  "errors": sum(values.values()) - values["exact"]}
                           for key, values in sorted(strata.items())},
                "example_ids": dict(example_ids),
            }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/processed/test.jsonl"))
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--output", type=Path, default=Path("bench/quality_error_audit.json"))
    arguments = parser.parse_args()
    report = audit(arguments.input, arguments.runs_dir, arguments.output)
    for label, item in report["modes"].items():
        print(label, item["counts"])
