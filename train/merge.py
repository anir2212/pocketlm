"""Merge a PEFT adapter into pinned full-precision base weights for GGUF conversion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from train.model_files import cached_snapshot

BASE = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"


def merge(adapter: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    snapshot = cached_snapshot(BASE, REVISION)
    base = AutoModelForCausalLM.from_pretrained(
        snapshot, local_files_only=True, dtype=torch.float16, low_cpu_mem_usage=True
    )
    model = PeftModel.from_pretrained(base, adapter).merge_and_unload()
    model.save_pretrained(output, safe_serialization=True)
    AutoTokenizer.from_pretrained(snapshot, local_files_only=True).save_pretrained(output)
    report = {"base": BASE, "revision": REVISION, "adapter": str(adapter),
              "parameters": sum(parameter.numel() for parameter in model.parameters()),
              "dtype": str(model.dtype), "output": str(output)}
    (output / "merge_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/merged_sft"))
    args = parser.parse_args()
    print(json.dumps(merge(args.adapter, args.output), indent=2))
