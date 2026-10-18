# PocketLM

Small language models, tool calls, and local evaluation.

A small-model lab for device commands. It trains a byte-level GPT from scratch, builds a schema-checked tool-call corpus, and compares adapter tuning and GGUF inference on held-out examples.

## Why this project

A device assistant has to turn vague requests into a small set of safe, structured actions, often with limited memory or network access. This project asks how far a small local model can get, and measures where grammar constraints help more than another training pass.

## Run locally

You need Python 3.12, uv; optional llama.cpp for GGUF and CUDA for QLoRA.

```sh
uv sync --python python3.12 --extra test
make tiny
make data
uv run --no-sync pytest -q tests
```

The scratch-model run and corpus build are reproducible locally. LoRA/DPO and GGUF evaluation code is present; see [results](RESULTS.md) for completed comparisons. The optional [Colab notebook](colab/Run_PocketLM.ipynb) runs one GPU experiment at a time. Build its upload archive with `uv run python -m colab.build_bundle`.

A completed NF4 QLoRA run has been returned from Colab and checked against the local corpus and training config. Its adapter was evaluated on the same 728 synthetic held-out requests as the local rank-8 LoRA adapter, using an unquantized CPU base for both. QLoRA got 604 exact targets and rank 8 got 637; the results include paired counts and limitations. The weights and Colab ZIP stay outside Git.

The local CPU GGUF comparison now covers four weight formats, each with and without a JSON grammar. The [measurement table](bench/quant_report.json) and [chart](bench/quant_pareto.png) include held-out exact-call accuracy, file size, and repeated-request timing. The timing sample repeats one prompt 20 times per mode, so it describes that workload rather than a general latency guarantee. A separate, bounded check compares the pinned base model with the rank-8 adapter on 100 cached ARC Easy and 100 HellaSwag questions per task ([base](bench/base_general_eval_100.json), [adapter](bench/sft_r8_general_eval_100.json)). See [results](RESULTS.md) for the numbers and their limits.

## Repository map

`scratch/` trains the small GPT; `data/` builds and documents the tool-call set; `bench/` holds evaluation evidence; `quant/` and `demo/` cover local inference.

## Scope

The data mixes generated examples and public function calls. It is a research fixture, not a real device assistant. Model checkpoints and the Colab upload archive are generated locally and are not in Git.
