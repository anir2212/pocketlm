# PocketLM

PocketLM asks whether a small model can turn everyday requests into device-tool calls that software can check, while remaining practical to run locally. Its device is a mock, so the demo cannot change a real laptop. A separate from-scratch transformer makes the model mechanics inspectable.

## Why this project

A device assistant has to turn vague requests into a small set of safe, structured actions, often with limited memory or network access. This project asks how far a small local model can get, and measures where grammar constraints help more than another training pass.

## Milestones

| Plan milestone | Where it stands |
|---|---|
| M1–M2: scratch model and corpus | A 12.8M-parameter byte-level decoder completed 1,000 steps; the schema-checked corpus has 8,697 total rows, including 728 held-out synthetic device requests. |
| M3: LoRA / QLoRA | Rank-8 LoRA got 637/728 exact targets; the returned NF4-trained rank-16 adapter got 604/728 under local CPU evaluation. These runs changed several training factors together. |
| M4: preference optimization | DPO training code is present; a completed DPO comparison is not measured. |
| M5–M6: evaluate and quantize | All eight GGUF format/grammar quality cells and 20-request-per-cell CPU latency checks are recorded in [RESULTS.md](RESULTS.md). These are synthetic requests and one repeated latency prompt. |
| M7–M8: stretch and demo | Multi-GPU and student distillation are unmeasured. The local demo uses mock device state. |

The next quality check is an untouched set of human-written requests, especially paraphrases and multi-call instructions. No real-device reliability is claimed.

## Run locally

You need Python 3.12, uv; optional llama.cpp for GGUF and CUDA for QLoRA.

```sh
uv sync --python python3.12 --extra test
make tiny
make data
uv run --no-sync pytest -q tests
```

The scratch-model run and corpus build are reproducible locally. LoRA and GGUF comparisons have measured results; the DPO path remains unmeasured. The optional [Colab notebook](colab/Run_PocketLM.ipynb) runs one GPU experiment at a time. Build its upload archive with `uv run python -m colab.build_bundle`.

A completed NF4 QLoRA run has been returned from Colab and checked against the local corpus and training config. Its adapter was evaluated on the same 728 synthetic held-out requests as the local rank-8 LoRA adapter, using an unquantized CPU base for both. QLoRA got 604 exact targets and rank 8 got 637; the results include paired counts and limitations. The weights and Colab ZIP stay outside Git.

The local CPU GGUF comparison now covers four weight formats, each with and without a JSON grammar. The [measurement table](bench/quant_report.json) and [chart](bench/quant_pareto.png) include held-out exact-call accuracy, file size, and repeated-request timing. The timing sample repeats one prompt 20 times per mode, so it describes that workload rather than a general latency guarantee. A separate, bounded check compares the pinned base model with the rank-8 adapter on 100 cached ARC Easy and 100 HellaSwag questions per task ([base](bench/base_general_eval_100.json), [adapter](bench/sft_r8_general_eval_100.json)). See [results](RESULTS.md) for the numbers and their limits.

## Repository map

`scratch/` trains the small GPT; `data/` builds and documents the tool-call set; `bench/` holds evaluation evidence; `quant/` and `demo/` cover local inference.

## Scope

The data mixes generated examples and public function calls. It is a research fixture, not a real device assistant. Model checkpoints and the Colab upload archive are generated locally and are not in Git.
