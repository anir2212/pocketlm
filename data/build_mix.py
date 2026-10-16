"""Build a schema-filtered, split-safe domain/general-function/chat mixture."""

from __future__ import annotations

import argparse
import ast
import json
import random
import re
from collections import Counter
from pathlib import Path

import yaml

from data.build_synthetic import generate
from data.clean import clean_row
from data.dedup import deduplicate


def extract_tool(system: str, name: str) -> dict | None:
    decoder = json.JSONDecoder()
    for index, character in enumerate(system):
        if character != "{":
            continue
        try:
            candidate, _ = decoder.raw_decode(system[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and candidate.get("name") == name and isinstance(
                candidate.get("parameters"), dict):
            return candidate
    return None


def convert_glaive(raw: dict, index: int) -> dict | None:
    chat = raw.get("chat", "")
    match = re.search(r"USER:\s*(.*?)\s*ASSISTANT:\s*<functioncall>\s*(.*?)\s*<\|endoftext\|>",
                      chat, re.S)
    if not match:
        return None
    try:
        literal = match.group(2).strip()
        call = (json.loads(literal) if literal.startswith('{"') and "'" not in literal
                else ast.literal_eval(literal))
        name = call["name"]
        arguments = call["arguments"]
        arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
        tool = extract_tool(raw.get("system", ""), name)
        if tool is None:
            return None
        return {"id": f"glaive_{index:06d}", "source": "general_function", "split": "train",
                "prompt": match.group(1).strip(), "tools": [tool],
                "target": {"calls": [{"tool": name, "arguments": arguments}]}}
    except (KeyError, TypeError, ValueError, SyntaxError, json.JSONDecodeError):
        return None


def general_functions(limit: int, heldout_tools: set[str],
                      revision: str) -> tuple[list[dict], dict]:
    from datasets import load_dataset

    rows = []
    rejected = Counter()
    for index, raw in enumerate(load_dataset("glaiveai/glaive-function-calling-v2",
                                               split="train", streaming=True, revision=revision)):
        row = convert_glaive(raw, index)
        if row is None:
            rejected["conversion"] += 1
            continue
        if row["target"]["calls"][0]["tool"] in heldout_tools:
            rejected["heldout_tool"] += 1
            continue
        row, reason = clean_row(row)
        if reason:
            rejected[reason] += 1
            continue
        rows.append(row)
        if len(rows) >= limit:
            break
    if len(rows) < limit:
        raise RuntimeError(f"needed {limit} general-function rows, got {len(rows)}")
    return rows, dict(rejected)


def general_chat(limit: int, revision: str) -> tuple[list[dict], dict]:
    from datasets import load_dataset

    rows = []
    rejected = Counter()
    for index, raw in enumerate(load_dataset("databricks/databricks-dolly-15k",
                                               split="train", streaming=True, revision=revision)):
        prompt = raw.get("instruction", "")
        if raw.get("context"):
            prompt += "\nContext: " + raw["context"]
        row = {"id": f"dolly_{index:05d}", "source": "general_chat", "split": "train",
               "prompt": prompt, "target": {"text": raw.get("response", "")}}
        row, reason = clean_row(row)
        if reason:
            rejected[reason] += 1
            continue
        rows.append(row)
        if len(rows) >= limit:
            break
    if len(rows) < limit:
        raise RuntimeError(f"needed {limit} general-chat rows, got {len(rows)}")
    return rows, dict(rejected)


def build(config_path: Path, output: Path, domain_only: bool = False) -> dict:
    config = yaml.safe_load(config_path.read_text())
    domain = generate(config)
    cleaned = []
    rejected = Counter()
    for row in domain:
        row, reason = clean_row(row)
        if reason:
            rejected[reason] += 1
        else:
            cleaned.append(row)
    domain, dedup_counts = deduplicate(cleaned)
    train = [row for row in domain if row["split"] == "train"]
    test = [row for row in domain if row["split"] == "test"]
    if not domain_only:
        ratios = config["mix"]
        desired_general = round(len(train) * ratios["general_function"] / ratios["domain"])
        desired_chat = min(config["general_chat_limit"],
                           round(len(train) * ratios["general_chat"] / ratios["domain"]))
        general, general_rejected = general_functions(min(config["general_function_limit"],
                                                      desired_general * 3),
                                                       set(config["heldout_tools"]),
                                                       config["function_dataset_revision"])
        chat, chat_rejected = general_chat(desired_chat, config["chat_dataset_revision"])
        # General sources may share prompts; keep domain test examples first so
        # no identical or near-identical public prompt contaminates the holdout.
        mixed, mixed_dedup = deduplicate(test + train + general + chat)
        public = [row for row in mixed if row["source"] == "general_function"]
        if len(public) < desired_general:
            raise RuntimeError(
                f"only {len(public)} unique function examples; need {desired_general}"
            )
        selected = {row["id"] for row in random.Random(config["seed"]).sample(
            public, desired_general)}
        mixed = [row for row in mixed if row["source"] != "general_function" or
                 row["id"] in selected]
    else:
        general_rejected, chat_rejected, mixed_dedup = {}, {}, {}
        mixed = domain
    random.Random(config["seed"]).shuffle(mixed)
    output.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        rows = [row for row in mixed if row["split"] == split]
        (output / f"{split}.jsonl").write_text("".join(
            json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    counts = Counter((row["split"], row["source"]) for row in mixed)
    report = {"raw_domain": len(cleaned) + sum(rejected.values()),
              "domain_clean_rejected": dict(rejected), "domain_dedup": dedup_counts,
              "public_function_rejected": general_rejected, "public_chat_rejected": chat_rejected,
              "mixed_dedup": mixed_dedup,
              "counts": {f"{split}/{source}": count
                         for (split, source), count in sorted(counts.items())}}
    (output / "build_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/corpus.yaml"))
    parser.add_argument("--output", type=Path, default=Path("data/processed"))
    parser.add_argument("--domain-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build(args.config, args.output, args.domain_only), indent=2))
