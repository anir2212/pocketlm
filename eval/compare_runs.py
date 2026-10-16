"""Compute paired structured-output differences from saved per-request predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from eval.tool_eval import item_metrics


def _load(path: Path) -> tuple[dict, list[dict]]:
    report = json.loads(path.read_text())
    rows = report["predictions"]
    return report, rows


def compare(left_path: Path, right_path: Path, output: Path,
            seed: int = 17, repetitions: int = 1000) -> dict:
    left_report, left = _load(left_path)
    right_report, right = _load(right_path)
    if not left or len(left) != len(right) or [row["id"] for row in left] != [
        row["id"] for row in right
    ] or any(a["gold"] != b["gold"] for a, b in zip(left, right, strict=True)):
        raise ValueError("runs must contain the same nonempty ordered IDs and gold targets")
    pairs = []
    for a, b in zip(left, right, strict=True):
        first = item_metrics({"target": a["gold"]}, a.get("output", a.get("content", "")))
        second = item_metrics({"target": b["gold"]}, b.get("output", b.get("content", "")))
        pairs.append({metric: (int(first[metric]), int(second[metric]))
                      for metric in ("full_exact", "target_valid", "json_valid")})
    comparisons = {}
    for metric in ("full_exact", "target_valid", "json_valid"):
        values = [pair[metric] for pair in pairs]
        n = len(values)
        deltas = []
        rng = random.Random(seed)
        for _ in range(repetitions):
            sample = [rng.choice(values) for _ in values]
            deltas.append(sum(right - left for left, right in sample) / n)
        deltas.sort()
        comparisons[metric] = {
            "left_correct": sum(left for left, _ in values),
            "right_correct": sum(right for _, right in values),
            "both_correct": sum(left and right for left, right in values),
            "left_only": sum(left and not right for left, right in values),
            "right_only": sum(right and not left for left, right in values),
            "right_minus_left": sum(right - left for left, right in values) / n,
            "paired_bootstrap_95pct": [deltas[int(0.025 * repetitions)],
                                       deltas[min(repetitions - 1, int(0.975 * repetitions))]],
        }
    result = {
        "examples": len(pairs), "seed": seed, "bootstrap_repetitions": repetitions,
        "left": {"path": str(left_path),
                 "sha256": hashlib.sha256(left_path.read_bytes()).hexdigest(),
                 "model": left_report.get("model"), "grammar": left_report.get("grammar")},
        "right": {"path": str(right_path),
                  "sha256": hashlib.sha256(right_path.read_bytes()).hexdigest(),
                  "model": right_report.get("model"), "grammar": right_report.get("grammar")},
        "comparisons": comparisons,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compare(args.left, args.right, args.output), indent=2))
