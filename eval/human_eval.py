"""Predict, score and pair-compare model runs on a validated human-written request set.

Every step re-validates the set (provenance, schemas, overlap) and checks that a
prediction file was produced from exactly the same frozen rows. Template rows can
pass through ``predict --allow-examples`` as a plumbing check, but are never scored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

from eval.compare_runs import paired_summary
from eval.error_audit import classify, stratum
from eval.human_set import default_references, kind, load_set
from eval.tool_eval import BASE, REVISION, bootstrap, item_metrics, parse_output
from tools_api.executor import validate_target

SELECTED_ADAPTER = Path("runs/sft_r8_lr1e4_2k/adapter")
PAIRED_METRICS = ("full_exact", "target_valid", "json_valid")


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _set_identity(loaded: dict) -> dict:
    return {"set_version": loaded["set_version"], "manifest_sha256": loaded["manifest"]["sha256"],
            "rows_sha256": loaded["rows"]["sha256"],
            "tool_schema_sha256": loaded["tool_schema_sha256"], "example": loaded["example"],
            "examples": loaded["row_count"]}


def _model_row(row: dict) -> dict:
    # The same system prompt and tool catalog as the generated holdout (train.formatting).
    return {"source": "synthetic_device", "prompt": row["prompt"]}


def _write(path: Path, report: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def _prediction_report(loaded: dict, backend: dict, outputs: list[dict]) -> dict:
    return {"set": _set_identity(loaded), "backend": backend,
            "contains_example_rows": loaded["example"],
            "quality_result": False if loaded["example"] else None,
            "predictions": outputs}


def predict_hf(manifest: Path, output: Path, adapter: Path | None, references: list[Path],
               allow_examples: bool = False, model_id: str = BASE,
               revision: str = REVISION) -> dict:
    from eval.tool_eval import generate, load_model

    loaded = load_set(manifest, references, allow_examples)
    model, tokenizer, device = load_model(adapter, model_id, revision)
    outputs = [{"id": row["id"], "targets": row["targets"],
                "output": generate(model, tokenizer, device, _model_row(row))}
               for row in loaded["rows_list"]]
    weights = adapter / "adapter_model.safetensors" if adapter else None
    backend = {"kind": "hf", "model": model_id, "revision": revision,
               "adapter": str(adapter) if adapter else None,
               "adapter_sha256": _file_sha(weights) if weights and weights.is_file() else None,
               "device": device, "decoding": "greedy", "max_new_tokens": 160}
    return _write(output, _prediction_report(loaded, backend, outputs))


def _git_commit(path: Path) -> str | None:
    for parent in path.resolve().parents:
        if (parent / ".git").exists():
            result = subprocess.run(["git", "-C", str(parent), "rev-parse", "HEAD"],
                                    capture_output=True, text=True, check=False)
            return result.stdout.strip() or None
    return None


def predict_gguf(manifest: Path, output: Path, model: Path, server_binary: Path,
                 references: list[Path], grammar_path: Path | None = None,
                 allow_examples: bool = False) -> dict:
    from transformers import AutoTokenizer

    from quant.runtime import GGUFServer
    from train.formatting import render
    from train.model_files import cached_snapshot

    loaded = load_set(manifest, references, allow_examples)
    tokenizer = AutoTokenizer.from_pretrained(cached_snapshot(BASE, REVISION))
    grammar = grammar_path.read_text() if grammar_path else None
    outputs = []
    with GGUFServer(server_binary, model) as server:
        for row in loaded["rows_list"]:
            completion = server.complete(
                render(_model_row(row), tokenizer, include_answer=False), grammar=grammar)
            outputs.append({"id": row["id"], "targets": row["targets"],
                            "output": completion["content"]})
    backend = {"kind": "gguf", "model": str(model), "model_sha256": _file_sha(model),
               "grammar": str(grammar_path) if grammar_path else None,
               "grammar_sha256": _file_sha(grammar_path) if grammar_path else None,
               "server_binary": str(server_binary), "llama_cpp_commit": _git_commit(server_binary),
               "decoding": "temperature 0, seed 7", "max_tokens": 160}
    return _write(output, _prediction_report(loaded, backend, outputs))


def score_item(targets: list[dict], raw: str) -> dict:
    """Score one output; an exact match to any listed alternative counts as exact.

    Denominators use the first target's kind; the validator requires alternatives to
    share it, so the call/refusal/clarification denominators never depend on the output.
    """
    predicted = parse_output(raw)
    index = next((i for i, target in enumerate(targets) if predicted == target), 0)
    item = item_metrics({"target": targets[index]}, raw)
    item["matched_alternative"] = item["full_exact"] and index > 0
    item["kind_correct"] = (predicted is not None and validate_target(predicted)
                            and kind(predicted) == kind(targets[0]))
    item["failure"] = classify(targets[index], raw)
    return item


def _checked_predictions(loaded: dict, path: Path) -> tuple[dict, list[dict]]:
    report = json.loads(path.read_text())
    if report.get("contains_example_rows") or any(
        row.get("example") for row in loaded["rows_list"]
    ):
        raise ValueError(f"{path}: template/example rows are never scored")
    identity = _set_identity(loaded)
    recorded = report.get("set", {})
    for key in ("set_version", "rows_sha256", "tool_schema_sha256"):
        if recorded.get(key) != identity[key]:
            raise ValueError(f"{path}: {key} differs from the validated set")
    predictions = report.get("predictions", [])
    rows = loaded["rows_list"]
    if [p.get("id") for p in predictions] != [row["id"] for row in rows] or any(
        p.get("targets") != row["targets"] for p, row in zip(predictions, rows, strict=True)
    ):
        raise ValueError(f"{path}: prediction IDs or targets differ from the set")
    return report, predictions


def _fraction(items: list[dict], field: str, denominator: str | None = None) -> dict:
    relevant = [item for item in items if denominator is None or item[denominator]]
    correct = sum(bool(item[field]) for item in relevant)
    return {"correct": correct, "denominator": len(relevant),
            "rate": correct / len(relevant) if relevant else None}


def counted_summary(items: list[dict]) -> dict:
    """Every metric with its numerator and denominator."""
    if not items:
        raise ValueError("no evaluation items")
    tp = sum(item["argument_tp"] for item in items)
    predicted = sum(item["argument_pred"] for item in items)
    gold = sum(item["argument_gold"] for item in items)
    precision = tp / predicted if predicted else 0.0
    recall = tp / gold if gold else 0.0
    return {
        "examples": len(items),
        "json_validity": _fraction(items, "json_valid"),
        "target_schema_validity": _fraction(items, "target_valid"),
        "overall_target_exact_match": _fraction(items, "full_exact"),
        "full_call_exact_match": _fraction(items, "full_exact", "has_calls"),
        "tool_name_exact_match_on_calls": _fraction(items, "tool_exact", "has_calls"),
        "hallucinated_tool": _fraction(items, "hallucinated_tool"),
        "correct_refusal": _fraction(items, "correct_refusal", "is_refusal"),
        "correct_clarification": _fraction(items, "correct_clarification", "is_clarification"),
        "response_kind_correct": _fraction(items, "kind_correct"),
        "exact_via_alternative_target": _fraction(items, "matched_alternative"),
        "argument_micro": {"true_positive": tp, "predicted": predicted, "gold": gold,
                           "precision": precision, "recall": recall,
                           "f1": 2 * precision * recall / (precision + recall)
                           if precision + recall else 0.0},
    }


def score(manifest: Path, predictions_path: Path, output: Path, references: list[Path],
          seed: int = 17, repetitions: int = 1000) -> dict:
    loaded = load_set(manifest, references)
    report, predictions = _checked_predictions(loaded, predictions_path)
    rows = loaded["rows_list"]
    items = [score_item(row["targets"], prediction["output"])
             for row, prediction in zip(rows, predictions, strict=True)]
    failures = Counter(item["failure"] for item in items)
    strata: dict[str, Counter] = defaultdict(Counter)
    for row, item in zip(rows, items, strict=True):
        strata[stratum({"target": row["targets"][0]})][item["failure"]] += 1
    near = loaded["near_duplicates"]
    novel = [item for row, item in zip(rows, items, strict=True) if row["id"] not in near]
    result = {
        "set": _set_identity(loaded),
        "predictions": {"path": str(predictions_path), "sha256": _file_sha(predictions_path),
                        "backend": report.get("backend")},
        "references": loaded["references"],
        "metrics": counted_summary(items),
        "metrics_excluding_near_duplicates": counted_summary(novel) if novel else None,
        "confidence_intervals_95pct": bootstrap(items, repetitions, seed),
        "bootstrap": {"seed": seed, "repetitions": repetitions},
        "failure_order": ["invalid_json_object", "invalid_target_schema",
                          "wrong_response_kind", "wrong_tool_sequence",
                          "wrong_arguments_or_payload", "exact"],
        "failures": dict(failures),
        "strata": {key: {"examples": sum(values.values()), "exact": values["exact"],
                         "failures": {k: v for k, v in values.items() if k != "exact"}}
                   for key, values in sorted(strata.items())},
        "items": [{"id": row["id"], "failure": item["failure"],
                   "exact": bool(item["full_exact"])}
                  for row, item in zip(rows, items, strict=True)],
    }
    return _write(output, result)


def compare(manifest: Path, left_path: Path, right_path: Path, output: Path,
            references: list[Path], seed: int = 17, repetitions: int = 1000) -> dict:
    loaded = load_set(manifest, references)
    left_report, left = _checked_predictions(loaded, left_path)
    right_report, right = _checked_predictions(loaded, right_path)
    rows = loaded["rows_list"]
    pairs = []
    for row, a, b in zip(rows, left, right, strict=True):
        first = score_item(row["targets"], a["output"])
        second = score_item(row["targets"], b["output"])
        pairs.append({metric: (int(first[metric]), int(second[metric]))
                      for metric in PAIRED_METRICS})
    result = {
        "set": _set_identity(loaded), "seed": seed, "bootstrap_repetitions": repetitions,
        "left": {"path": str(left_path), "sha256": _file_sha(left_path),
                 "backend": left_report.get("backend")},
        "right": {"path": str(right_path), "sha256": _file_sha(right_path),
                  "backend": right_report.get("backend")},
        "comparisons": {metric: paired_summary([pair[metric] for pair in pairs],
                                               seed, repetitions)
                        for metric in PAIRED_METRICS},
    }
    return _write(output, result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reference", type=Path, action="append",
                        help="override the training/holdout prompt files to check against")
    commands = parser.add_subparsers(dest="command", required=True)
    hf = commands.add_parser("predict-hf")
    hf.add_argument("--adapter", type=Path, default=SELECTED_ADAPTER)
    hf.add_argument("--no-adapter", action="store_true", help="evaluate the base model")
    gguf = commands.add_parser("predict-gguf")
    gguf.add_argument("--model", type=Path, required=True)
    gguf.add_argument("--server-binary", type=Path, required=True)
    gguf.add_argument("--grammar", type=Path)
    for command in (hf, gguf):
        command.add_argument("--manifest", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--allow-examples", action="store_true",
                             help="plumbing check on template rows; never a quality result")
    scorer = commands.add_parser("score")
    scorer.add_argument("--manifest", type=Path, required=True)
    scorer.add_argument("--predictions", type=Path, required=True)
    scorer.add_argument("--output", type=Path, required=True)
    pair = commands.add_parser("compare")
    pair.add_argument("--manifest", type=Path, required=True)
    pair.add_argument("--left", type=Path, required=True)
    pair.add_argument("--right", type=Path, required=True)
    pair.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    references = args.reference or default_references()
    try:
        if args.command == "predict-hf":
            result = predict_hf(args.manifest, args.output,
                                None if args.no_adapter else args.adapter,
                                references, args.allow_examples)
            result = {key: value for key, value in result.items() if key != "predictions"}
        elif args.command == "predict-gguf":
            result = predict_gguf(args.manifest, args.output, args.model, args.server_binary,
                                  references, args.grammar, args.allow_examples)
            result = {key: value for key, value in result.items() if key != "predictions"}
        elif args.command == "score":
            result = score(args.manifest, args.predictions, args.output, references)
            result = {key: value for key, value in result.items() if key != "items"}
        else:
            result = compare(args.manifest, args.left, args.right, args.output, references)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
