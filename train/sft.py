"""TRL completion-only LoRA/QLoRA training with exact run configs and local CSV logs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import time
from pathlib import Path

import psutil
import torch
import yaml
from datasets import Dataset
from huggingface_hub import snapshot_download
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
from trl import SFTConfig, SFTTrainer

from train.formatting import messages
from train.model_files import cached_snapshot

TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def split_train_validation(rows: list[dict], fraction: float, seed: int):
    """Stratify by source, never use the held-out test file for model selection."""
    by_source: dict[str, list[dict]] = {}
    for row in rows:
        by_source.setdefault(row["source"], []).append(row)
    rng = random.Random(seed)
    train, validation = [], []
    for source_rows in by_source.values():
        rng.shuffle(source_rows)
        boundary = max(1, round(len(source_rows) * fraction))
        validation.extend(source_rows[:boundary])
        train.extend(source_rows[boundary:])
    rng.shuffle(train)
    rng.shuffle(validation)
    return train, validation


def dataset(rows: list[dict], tokenizer, max_length: int) -> tuple[Dataset, int]:
    records = []
    excluded = 0
    for row in rows:
        turns = messages(row, include_answer=True)
        if len(tokenizer.apply_chat_template(turns)) > max_length:
            excluded += 1
            continue
        records.append({"prompt": turns[:-1], "completion": turns[-1:]})
    if not records:
        raise RuntimeError("all rows exceeded the context length")
    return Dataset.from_list(records), excluded


class ResourceCallback(TrainerCallback):
    def __init__(self, device: str) -> None:
        self.device = device
        self.peak_mps_bytes = 0
        self.peak_rss_bytes = 0

    def on_step_end(self, args, state, control, **kwargs):
        self.peak_rss_bytes = max(self.peak_rss_bytes, psutil.Process().memory_info().rss)
        if self.device == "mps":
            self.peak_mps_bytes = max(self.peak_mps_bytes,
                                      torch.mps.driver_allocated_memory())
        return control


def train(config_path: Path, output: Path, max_steps: int | None = None,
          max_rows: int | None = None, rank: int | None = None,
          learning_rate: float | None = None,
          resume_from_checkpoint: Path | None = None) -> dict:
    config = yaml.safe_load(config_path.read_text())
    if rank is not None:
        config["rank"], config["alpha"] = rank, 2 * rank
    if learning_rate is not None:
        config["learning_rate"] = learning_rate
    if max_steps is not None:
        config["max_steps_override"] = max_steps
    if max_rows is not None:
        config["max_rows_override"] = max_rows
    if resume_from_checkpoint is not None and config["quantization"] != "nf4":
        raise ValueError("checkpoint resume is enabled only for the NF4 QLoRA run")
    effective_config = yaml.safe_dump(config, sort_keys=True)
    if resume_from_checkpoint is not None:
        previous = output / "run_config.yaml"
        if not previous.is_file() or previous.read_text() != effective_config:
            raise ValueError("checkpoint run configuration differs from this invocation")
        recorded_data = output / "train_data.sha256"
        current_data_hash = hashlib.sha256(
            Path("data/processed/train.jsonl").read_bytes()).hexdigest()
        if not recorded_data.is_file() or recorded_data.read_text().strip() != current_data_hash:
            raise ValueError("checkpoint training data differs from this invocation")
        if resume_from_checkpoint.resolve().parent != (output / "trainer").resolve():
            raise ValueError("checkpoint must be inside this run's trainer directory")
    output.mkdir(parents=True, exist_ok=True)
    is_primary = int(os.getenv("RANK", "0")) == 0
    if is_primary:
        if resume_from_checkpoint is None:
            shutil.copy2(config_path, output / "source_config.yaml")
            (output / "run_config.yaml").write_text(effective_config)
            (output / "train_data.sha256").write_text(
                hashlib.sha256(Path("data/processed/train.jsonl").read_bytes()).hexdigest()
                + "\n")
    torch.manual_seed(config["seed"])
    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    if config["quantization"] == "nf4" and device != "cuda":
        raise RuntimeError("NF4 QLoRA requires CUDA here; use the provided Colab/T4 run")
    use_flash_attention = device == "cuda" and os.getenv("POCKETLM_FLASH_ATTENTION") == "1"
    effective_packing = bool(config["packing"] and use_flash_attention)
    if config["packing"] and not effective_packing:
        print("Disabled packing: this backend cannot guarantee isolation between packed samples.",
              flush=True)
    try:
        snapshot = cached_snapshot(config["model_id"], config["model_revision"])
    except FileNotFoundError:
        snapshot = snapshot_download(repo_id=config["model_id"],
                                     revision=config["model_revision"])
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    rows = load_rows(Path("data/processed/train.jsonl"))
    if max_rows:
        rows = rows[:max_rows]
    train_rows, validation_rows = split_train_validation(
        rows, config["validation_fraction"], config["seed"]
    )
    training, train_excluded = dataset(train_rows, tokenizer, config["max_length"])
    validation, validation_excluded = dataset(validation_rows, tokenizer, config["max_length"])
    training_tokens_per_epoch = sum(
        len(tokenizer.apply_chat_template(messages(row, include_answer=True)))
        for row in train_rows
        if len(tokenizer.apply_chat_template(messages(row, include_answer=True))) <=
        config["max_length"]
    )
    quantization = (BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                       bnb_4bit_compute_dtype=torch.float16,
                                       bnb_4bit_use_double_quant=True)
                    if config["quantization"] == "nf4" else None)
    model = AutoModelForCausalLM.from_pretrained(
        snapshot, local_files_only=True,
        dtype=torch.float16 if device != "cpu" else torch.float32,
        quantization_config=quantization,
        device_map="auto" if quantization is not None else None,
        attn_implementation="flash_attention_2" if use_flash_attention else "sdpa",
        low_cpu_mem_usage=True,
    )
    if quantization is not None:
        model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(r=config["rank"], lora_alpha=config["alpha"], lora_dropout=0.05,
                      target_modules=list(TARGET_MODULES), bias="none", task_type="CAUSAL_LM")
    checkpoint_steps = config.get("checkpoint_steps") if config["quantization"] == "nf4" else None
    options = SFTConfig(
        output_dir=str(output / "trainer"), seed=config["seed"],
        num_train_epochs=config["epochs"], max_steps=max_steps or -1,
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"], max_length=config["max_length"],
        packing=effective_packing, eval_packing=False, completion_only_loss=True,
        gradient_checkpointing=True, eval_strategy="epoch",
        save_strategy="steps" if checkpoint_steps else "no",
        save_steps=checkpoint_steps or 500, save_total_limit=1 if checkpoint_steps else None,
        save_only_model=False,
        logging_steps=20, logging_first_step=True, report_to="none", fp16=False,
        bf16=False, dataset_num_proc=1, dataloader_num_workers=0,
    )
    resources = ResourceCallback(device)
    trainer = SFTTrainer(model=model, args=options, train_dataset=training,
                         eval_dataset=validation, processing_class=tokenizer,
                         peft_config=lora, callbacks=[resources])
    started = time.perf_counter()
    outcome = trainer.train(resume_from_checkpoint=str(resume_from_checkpoint)
                            if resume_from_checkpoint else None)
    duration = time.perf_counter() - started
    if trainer.is_world_process_zero():
        trainer.model.save_pretrained(output / "adapter")
        tokenizer.save_pretrained(output / "adapter")
        with (output / "log.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["step", "epoch", "loss", "eval_loss",
                                                       "learning_rate", "grad_norm"])
            writer.writeheader()
            for item in trainer.state.log_history:
                writer.writerow({key: item.get(key) for key in writer.fieldnames})
    metrics = {
        "model_id": config["model_id"], "model_revision": config["model_revision"],
        "device": device, "quantization": config["quantization"],
        "rank": config["rank"], "alpha": config["alpha"],
        "checkpoint_interval_steps": checkpoint_steps,
        "resumed_from_checkpoint": str(resume_from_checkpoint)
        if resume_from_checkpoint else None,
        "distributed_world_size": int(os.getenv("WORLD_SIZE", "1")),
        "packing_requested": config["packing"], "packing_effective": effective_packing,
        "learning_rate": config["learning_rate"], "epochs_configured": config["epochs"],
        "steps_completed": trainer.state.global_step, "duration_seconds": duration,
        "train_examples": len(training), "validation_examples": len(validation),
        "train_excluded_too_long": train_excluded,
        "validation_excluded_too_long": validation_excluded,
        "train_loss": outcome.training_loss,
        "estimated_input_tokens_per_second": (
            training_tokens_per_epoch * (trainer.state.epoch or 0) / duration
            if duration else None
        ),
        "input_tokens_per_epoch": training_tokens_per_epoch,
        "eval_loss": next((item["eval_loss"] for item in reversed(trainer.state.log_history)
                           if "eval_loss" in item), None),
        "peak_mps_driver_bytes_sampled": resources.peak_mps_bytes if device == "mps" else None,
        "peak_cuda_allocated_bytes": (torch.cuda.max_memory_allocated() if device == "cuda"
                                      else None),
        "peak_process_rss_bytes_sampled": resources.peak_rss_bytes,
    }
    if trainer.is_world_process_zero():
        (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/sft_lora.yaml"))
    parser.add_argument("--output", type=Path, default=Path("runs/sft_lora"))
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    args = parser.parse_args()
    print(json.dumps(train(args.config, args.output, args.max_steps, args.max_rows,
                           args.rank, args.learning_rate, args.resume_from_checkpoint), indent=2))
