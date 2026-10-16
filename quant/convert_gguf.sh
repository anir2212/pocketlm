#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: quant/convert_gguf.sh MERGED_HF_DIR OUTPUT_F16_GGUF" >&2
  exit 2
fi
: "${LLAMA_CPP_DIR:?Set LLAMA_CPP_DIR to a checkout of pinned llama.cpp}"
expected_revision=23b0202a189c44a54625aadcb37a946dd1d6278d
actual_revision="$(git -C "${LLAMA_CPP_DIR}" rev-parse HEAD)"
if [[ "$actual_revision" != "$expected_revision" ]]; then
  echo "llama.cpp checkout is $actual_revision; expected $expected_revision" >&2
  exit 1
fi
"${POCKETLM_PYTHON:-.venv/bin/python}" "${LLAMA_CPP_DIR}/convert_hf_to_gguf.py" "$1" --outtype f16 --outfile "$2"
