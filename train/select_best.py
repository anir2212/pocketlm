"""Select the lowest validation-loss LoRA run without touching the held-out test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def select(candidates: list[Path], output: Path) -> Path:
    scored = []
    for candidate in candidates:
        metrics = json.loads((candidate / "metrics.json").read_text())
        loss = metrics.get("eval_loss")
        if loss is None:
            raise ValueError(f"missing evaluation loss: {candidate}")
        scored.append((float(loss), candidate))
    if not scored:
        raise ValueError("no candidate runs")
    best = min(scored, key=lambda item: item[0])[1] / "adapter"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(str(best) + "\n")
    return best


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("runs/best_sft_adapter.txt"))
    parser.add_argument("candidates", nargs="+", type=Path)
    args = parser.parse_args()
    print(select(args.candidates, args.output))
