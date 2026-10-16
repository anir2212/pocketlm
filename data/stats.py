"""Report exact Qwen chat-template token counts for the built corpus."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from transformers import AutoTokenizer

from train.formatting import render


def report(data_dir: Path, output: Path, model: str, revision: str | None = None) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision, trust_remote_code=False)
    counts = Counter()
    tokens = defaultdict(list)
    kinds = Counter()
    for split in ("train", "test"):
        for line in (data_dir / f"{split}.jsonl").read_text().splitlines():
            row = json.loads(line)
            key = (split, row["source"])
            counts[key] += 1
            tokens[key].append(len(tokenizer(render(row, tokenizer))["input_ids"]))
            if row["source"] == "synthetic_device":
                target = row["target"]
                kinds[split, "refusal"] += int("refusal" in target)
                kinds[split, "clarification"] += int("clarification" in target)
                kinds[split, "multi_call"] += int(len(target.get("calls", [])) > 1)
    build = json.loads((data_dir / "build_report.json").read_text())
    lines = ["# PocketLM corpus statistics", "",
             f"Tokenizer: `{model}` at `{revision or 'unversioned'}` chat template. "
             "Counts include system catalog, user, "
             "and assistant output.",
             "", "| Split | Source | Rows | Total tokens | Median tokens | Max tokens |",
             "|---|---|---:|---:|---:|---:|"]
    for (split, source), count in sorted(counts.items()):
        values = sorted(tokens[split, source])
        lines.append(f"| {split} | {source} | {count:,} | {sum(values):,} | "
                     f"{values[len(values)//2]:,} | {values[-1]:,} |")
    lines.extend(["", "## Domain target types", "",
                  "| Split | Refusals | Clarifications | Multiple calls |",
                  "|---|---:|---:|---:|"])
    for split in ("train", "test"):
        lines.append(f"| {split} | {kinds[split, 'refusal']} | "
                     f"{kinds[split, 'clarification']} | {kinds[split, 'multi_call']} |")
    lines.extend(["", "## Filtering and split", "",
                  f"Raw synthetic rows: {build['raw_domain']:,}. Domain exact duplicate "
                  "prompts removed: "
                  f"{build['domain_dedup']['exact_removed']:,}; approximate MinHash near "
                  "duplicates "
                  f"removed: {build['domain_dedup']['near_removed']:,}.", "",
                  "The held-out device test uses unseen prompt templates and excludes "
                  "`close_app` and `set_sleep_timer` from device training rows. All accepted "
                  "function calls were checked against their schema. MinHash LSH is an "
                  "approximate 5-gram Jaccard filter at a configured 0.8 threshold, not a "
                  "mathematical guarantee that every near duplicate is removed.", "",
                  "The public function and chat samples carry their original licenses; see "
                  "[SOURCES.md](SOURCES.md). The generated JSONL files are local artifacts, "
                  "not committed to Git.", ""])
    output.write_text("\n".join(lines))
    return {"rows": sum(counts.values()), "tokens": sum(sum(group) for group in tokens.values())}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, default=Path("data/STATS.md"))
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--revision", default="7ae557604adf67be50417f59c2c2f167def9a775")
    args = parser.parse_args()
    print(json.dumps(report(args.data, args.output, args.model, args.revision)))
