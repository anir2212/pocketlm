#!/usr/bin/env bash
set -euo pipefail

if [[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')" -lt 2 ]]; then
  echo "This comparison needs two visible CUDA GPUs." >&2
  exit 1
fi
.venv/bin/accelerate launch --multi_gpu --num_processes 2 -m train.sft \
  --config configs/sft_lora.yaml --output runs/sft_ddp_r16_2k --max-rows 2000 \
  --rank 16 --learning-rate 0.0002
