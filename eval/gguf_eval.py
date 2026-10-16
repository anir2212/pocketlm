"""Score a GGUF on the same held-out tool requests as the HF model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from eval.tool_eval import BASE, REVISION, bootstrap, item_metrics, summarize
from quant.runtime import GGUFServer
from train.formatting import render
from train.model_files import cached_snapshot


def evaluate(model: Path, output: Path, server_binary: Path,
             grammar_path: Path | None = None, max_examples: int | None = None) -> dict:
    rows = [json.loads(line) for line in Path("data/processed/test.jsonl").read_text().splitlines()]
    rows = [row for row in rows if row["source"] == "synthetic_device"]
    if max_examples:
        rows = rows[:max_examples]
    if not rows:
        raise ValueError("no held-out rows")
    tokenizer = AutoTokenizer.from_pretrained(cached_snapshot(BASE, REVISION))
    grammar = grammar_path.read_text() if grammar_path else None
    records = []
    with GGUFServer(server_binary, model) as server:
        for row in rows:
            completion = server.complete(render(row, tokenizer, include_answer=False),
                                         grammar=grammar)
            records.append({"id": row["id"], "gold": row["target"], **completion})
    scores = [item_metrics(row, record["content"])
              for row, record in zip(rows, records, strict=True)]
    report = {"model": str(model), "grammar": str(grammar_path) if grammar_path else None,
              "metrics": summarize(scores), "confidence_intervals_95pct": bootstrap(scores),
              "predictions": records}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--server-binary", type=Path, required=True)
    parser.add_argument("--grammar", type=Path)
    parser.add_argument("--max-examples", type=int)
    args = parser.parse_args()
    result = evaluate(args.model, args.output, args.server_binary,
                      args.grammar, args.max_examples)
    print(json.dumps({key: value for key, value in result.items() if key != "predictions"},
                     indent=2))
