#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: quant/quantise.sh INPUT_F16_GGUF OUTPUT_DIR" >&2
  exit 2
fi
: "${LLAMA_CPP_DIR:?Set LLAMA_CPP_DIR to a checkout of pinned llama.cpp}"
expected_revision=23b0202a189c44a54625aadcb37a946dd1d6278d
actual_revision="$(git -C "${LLAMA_CPP_DIR}" rev-parse HEAD)"
if [[ "$actual_revision" != "$expected_revision" ]]; then
  echo "llama.cpp checkout is $actual_revision; expected $expected_revision" >&2
  exit 1
fi
mkdir -p "$2"
for kind in Q8_0 Q5_K_M Q4_K_M; do
  "${LLAMA_CPP_DIR}/build/bin/llama-quantize" "$1" "$2/model-${kind}.gguf" "$kind"
done
