"""Continue an SFT adapter with DPO; TRL copies it into a frozen reference adapter."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import time
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

from train.model_files import cached_snapshot


def train(config_path: Path, adapter: Path, pairs_path: Path, output: Path,
          beta: float | None = None, max_steps: int | None = None) -> dict:
    config = yaml.safe_load(config_path.read_text())
    if beta is not None:
        config["beta"] = beta
    if max_steps is not None:
        config["max_steps_override"] = max_steps
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output / "source_config.yaml")
    (output / "run_config.yaml").write_text(yaml.safe_dump(config, sort_keys=True))
    torch.manual_seed(config["seed"])
    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    snapshot = cached_snapshot(config["model_id"], config["model_revision"])
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        snapshot, local_files_only=True,
        dtype=torch.float16 if device != "cpu" else torch.float32,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(model, adapter, is_trainable=True)
    rows = [json.loads(line) for line in pairs_path.read_text().splitlines() if line]
    rows = [row for row in rows if len(tokenizer(row["prompt"] + row["chosen"])[
        "input_ids"]) <= config["max_length"] and len(tokenizer(row["prompt"] + row[
            "rejected"])["input_ids"]) <= config["max_length"]]
    if len(rows) < 2:
        raise RuntimeError("need at least two on-policy preference pairs")
    rng = random.Random(config["seed"])
    rng.shuffle(rows)
    cutoff = max(1, round(0.05 * len(rows)))
    validation = Dataset.from_list([{key: row[key] for key in ("prompt", "chosen", "rejected")}
                                    for row in rows[:cutoff]])
    training = Dataset.from_list([{key: row[key] for key in ("prompt", "chosen", "rejected")}
                                  for row in rows[cutoff:]])
    options = DPOConfig(
        output_dir=str(output / "trainer"), seed=config["seed"], beta=config["beta"],
        learning_rate=config["learning_rate"], num_train_epochs=config["epochs"],
        max_steps=max_steps or -1, max_length=config["max_length"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        gradient_checkpointing=True, eval_strategy="epoch", save_strategy="no",
        logging_steps=20, logging_first_step=True, report_to="none",
        bf16=False, fp16=False,
    )
    trainer = DPOTrainer(model=model, ref_model=None, args=options,
                         train_dataset=training, eval_dataset=validation,
                         processing_class=tokenizer)
    started = time.perf_counter()
    outcome = trainer.train()
    duration = time.perf_counter() - started
    trainer.model.save_pretrained(output / "adapter")
    tokenizer.save_pretrained(output / "adapter")
    with (output / "log.csv").open("w", newline="") as file:
        keys = sorted({key for row in trainer.state.log_history for key in row})
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(trainer.state.log_history)
    metrics = {"sft_adapter": str(adapter), "beta": config["beta"],
               "device": device, "pairs_train": len(training),
               "pairs_validation": len(validation),
               "steps_completed": trainer.state.global_step,
               "duration_seconds": duration, "train_loss": outcome.training_loss,
               "last_eval": next((row for row in reversed(trainer.state.log_history)
                                  if "eval_loss" in row), None)}
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/dpo.yaml"))
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, default=Path("runs/preference_pairs.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("runs/dpo_beta01"))
    parser.add_argument("--beta", type=float)
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()
    print(json.dumps(train(args.config, args.adapter, args.pairs, args.output,
                           args.beta, args.max_steps), indent=2))
