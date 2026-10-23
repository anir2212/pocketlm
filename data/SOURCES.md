# Data and model sources

| Source | Published license | Intended use | Notes |
|---|---|---|---|
| [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) | CDLA-Sharing-1.0 | Scratch decoder training/validation | Revision `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`; streamed 5,000 train and 500 validation stories. Generated stories have limited complexity. |
| [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) | Apache-2.0 | Primary LoRA/DPO base | Pinned revision `7ae557604adf67be50417f59c2c2f167def9a775`; weights cached locally. A rank-8 LoRA (local MPS) and a rank-16 NF4 QLoRA (Colab) were trained on it; DPO code exists but no DPO run has been measured (see `RESULTS.md`). |
| [SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) | Apache-2.0 | Comparison base | Pinned revision `a10cc1512eabd3dde888204e902eca88bddb4951`; evaluation pending. |
| [SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) | Apache-2.0 | Distillation student | Pinned revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`; training pending. |
| [Glaive function calling v2](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2) | Apache-2.0 label on dataset card | General tool-call mix | Revision `e7f4b6456019f5d8bcb991ef0dd67d8ff23221ac`; only schema-valid parsed function calls are retained. Dataset provenance should be reviewed before any redistribution. |
| [Databricks Dolly 15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) | CC BY-SA 3.0 | General chat retention mix | Revision `bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a`; attribution and share-alike obligations apply. |
| Local synthetic device corpus | Project-authored | Domain tool calls | Generated from `tools_api/schema.json` and deterministic templates; no real users or device APIs. |

Published license labels are recorded for reproducibility; any redistribution of a mixed dataset or trained weights needs a separate license review. No third-party dataset is committed in this repository.
