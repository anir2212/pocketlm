"""Train and evaluate the scratch decoder on streamed TinyStories."""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import torch
import yaml

from scratch.tiny_gpt import GPTConfig, TinyGPT, decode, encode


def load_texts(dataset: str, split: str, limit: int, local_file: str | None,
               revision: str | None = None) -> list[str]:
    if local_file:
        texts = [line.strip() for line in Path(local_file).read_text().splitlines()]
        return [text for text in texts if text][:limit]
    from datasets import load_dataset

    stream = load_dataset(dataset, split=split, streaming=True, revision=revision)
    texts = []
    for row in stream:
        text = row.get("text", "")
        if isinstance(text, str) and text.strip():
            texts.append(text)
        if len(texts) >= limit:
            break
    if len(texts) < limit:
        raise RuntimeError(f"requested {limit} {split} stories, found {len(texts)}")
    return texts


def token_tensor(texts: list[str]) -> torch.Tensor:
    return torch.tensor(encode("\n\n".join(texts)), dtype=torch.long)


def batch(data: torch.Tensor, size: int, block: int, generator: torch.Generator, device: str):
    if len(data) < block + 2:
        raise ValueError("corpus is shorter than one training window")
    starts = torch.randint(0, len(data) - block - 1, (size,), generator=generator)
    x = torch.stack([data[start:start + block] for start in starts])
    y = torch.stack([data[start + 1:start + block + 1] for start in starts])
    return x.to(device), y.to(device)


@torch.no_grad()
def evaluate(model: TinyGPT, data: torch.Tensor, batches: int, size: int,
             block: int, generator: torch.Generator, device: str) -> float:
    model.eval()
    losses = []
    for _ in range(batches):
        x, y = batch(data, size, block, generator, device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def train(config_path: Path, output: Path, train_file: str | None = None,
          validation_file: str | None = None, steps_override: int | None = None) -> dict:
    config = yaml.safe_load(config_path.read_text())
    seed = int(config["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    train_texts = load_texts(config["dataset"], "train", config["max_train_stories"],
                             train_file, config.get("dataset_revision"))
    validation_texts = load_texts(config["dataset"], "validation",
                                  config["max_validation_stories"], validation_file,
                                  config.get("dataset_revision"))
    train_tokens = token_tensor(train_texts)
    validation_tokens = token_tensor(validation_texts)
    model = TinyGPT(GPTConfig(block_size=config["block_size"], **config["model"])).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    generator = torch.Generator().manual_seed(seed)
    steps = steps_override or config["steps"]
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output / "config.yaml")
    losses: list[dict] = []
    started = time.perf_counter()
    model.train()
    for step in range(1, steps + 1):
        x, y = batch(train_tokens, config["batch_size"], config["block_size"], generator, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % config["eval_interval"] == 0 or step == steps:
            validation_loss = evaluate(model, validation_tokens, 5, config["batch_size"],
                                       config["block_size"], generator, device)
            row = {"step": step, "train_loss": loss.item(),
                   "validation_loss": validation_loss,
                   "validation_perplexity": math.exp(validation_loss)}
            losses.append(row)
            print(json.dumps(row), flush=True)
    prompt = torch.tensor([encode("Once upon a time")], device=device)
    sample = decode(model.generate(prompt, 120)[0].tolist())
    torch.save({"model": model.state_dict(), "config": asdict(model.config)},
               output / "checkpoint.pt")
    report = {"dataset": config["dataset"] if not train_file else "local_fixture",
              "dataset_revision": config.get("dataset_revision"),
              "train_stories": len(train_texts), "validation_stories": len(validation_texts),
              "train_bytes": len(train_tokens), "validation_bytes": len(validation_tokens),
              "parameters": parameter_count, "device": device, "seed": seed,
              "steps": steps, "duration_seconds": time.perf_counter() - started,
              "curve": losses, "sample": sample}
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/tiny_gpt.yaml"))
    parser.add_argument("--output", type=Path, default=Path("runs/tiny_gpt"))
    parser.add_argument("--train-file")
    parser.add_argument("--validation-file")
    parser.add_argument("--steps", type=int)
    args = parser.parse_args()
    train(args.config, args.output, args.train_file, args.validation_file, args.steps)
