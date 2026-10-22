.PHONY: tiny test data sft sft-cuda dpo quant eval eval-general eval-gguf eval-gguf-resume audit-errors \
	human-validate human-eval

PYTHON := .venv/bin/python
GGUF_PHASE ?= all
SFT_RUNS := runs/sft_r8_lr1e4_2k runs/sft_r16_lr2e4_2k runs/sft_r32_lr2e4_2k
HUMAN_TEMPLATE := data/human_eval/TEMPLATE_v0.manifest.json
HUMAN_SET ?=
HUMAN_TAG ?= human_v1
HUMAN_ADAPTER ?= runs/sft_r8_lr1e4_2k/adapter
HUMAN_GGUF ?= artifacts/model-Q4_K_M.gguf
HUMAN_GRAMMAR ?= quant/grammar.gbnf

tiny:
	.venv/bin/python -m scratch.train_tiny --config configs/tiny_gpt.yaml --output runs/tiny_gpt

test:
	.venv/bin/pytest -q tests
	.venv/bin/ruff check scratch data tools_api train eval quant demo colab tests

data:
	.venv/bin/python -m data.build_mix --config configs/corpus.yaml --output data/processed
	.venv/bin/python -m data.stats --data data/processed --output data/STATS.md

sft:
	$(PYTHON) -m train.sft --output runs/sft_r8_lr1e4_2k --max-rows 2000 --rank 8 --learning-rate 0.0001
	$(PYTHON) -m train.sft --output runs/sft_r16_lr2e4_2k --max-rows 2000 --rank 16 --learning-rate 0.0002
	$(PYTHON) -m train.sft --output runs/sft_r32_lr2e4_2k --max-rows 2000 --rank 32 --learning-rate 0.0002
	$(PYTHON) -m train.select_best $(SFT_RUNS)

sft-cuda:
	$(PYTHON) -m train.sft --config configs/sft_qlora.yaml --output runs/sft_qlora_r16_2k --max-rows 2000

dpo:
	$(PYTHON) -m train.make_preferences --adapter "$$(cat runs/best_sft_adapter.txt)"
	$(PYTHON) -m train.dpo --adapter "$$(cat runs/best_sft_adapter.txt)" --beta 0.1 --output runs/dpo_beta01
	$(PYTHON) -m train.dpo --adapter "$$(cat runs/best_sft_adapter.txt)" --beta 0.3 --output runs/dpo_beta03

quant:
	$(PYTHON) -m train.merge --adapter "$$(cat runs/best_sft_adapter.txt)" --output artifacts/merged_sft
	LLAMA_CPP_DIR="$${LLAMA_CPP_DIR:?Set LLAMA_CPP_DIR to pinned llama.cpp checkout}" quant/convert_gguf.sh artifacts/merged_sft artifacts/model-F16.gguf
	LLAMA_CPP_DIR="$${LLAMA_CPP_DIR:?Set LLAMA_CPP_DIR to pinned llama.cpp checkout}" quant/quantise.sh artifacts/model-F16.gguf artifacts

eval:
	$(PYTHON) -m eval.tool_eval --output runs/base_tool_eval.json
	$(PYTHON) -m eval.tool_eval --adapter "$$(cat runs/best_sft_adapter.txt)" --output runs/sft_tool_eval.json
	$(PYTHON) -m eval.tool_eval --adapter runs/dpo_beta01/adapter --output runs/dpo_beta01_tool_eval.json
	$(PYTHON) -m eval.tool_eval --adapter runs/dpo_beta03/adapter --output runs/dpo_beta03_tool_eval.json

eval-general:
	$(PYTHON) -m eval.general_eval --output runs/base_general_eval.json
	$(PYTHON) -m eval.general_eval --adapter "$$(cat runs/best_sft_adapter.txt)" --output runs/sft_general_eval.json

eval-gguf:
	@set -eu; \
	: "$${LLAMA_CPP_DIR:?Set LLAMA_CPP_DIR to pinned llama.cpp checkout}"; \
	for format in F16 Q8_0 Q5_K_M Q4_K_M; do \
	  for mode in plain grammar; do \
	    grammar_arg=""; \
	    if [ "$$mode" = grammar ]; then grammar_arg="--grammar quant/grammar.gbnf"; fi; \
	    $(PYTHON) -m eval.gguf_eval --model "artifacts/model-$$format.gguf" \
	      --server-binary "$$LLAMA_CPP_DIR/build/bin/llama-server" \
	      --max-examples "$${GGUF_EVAL_LIMIT:-728}" $$grammar_arg \
	      --output "runs/gguf_eval_$${format}_$${mode}.json"; \
	    $(PYTHON) -m eval.latency_bench --model "artifacts/model-$$format.gguf" \
	      --bench-binary "$$LLAMA_CPP_DIR/build/bin/llama-bench" \
	      --server-binary "$$LLAMA_CPP_DIR/build/bin/llama-server" $$grammar_arg \
	      --output "runs/latency_$${format}_$${mode}.json"; \
	  done; \
	done
	$(PYTHON) -m eval.report

eval-gguf-resume:
	$(PYTHON) -m eval.run_gguf_matrix --llama-cpp-dir "$${LLAMA_CPP_DIR:?Set LLAMA_CPP_DIR to pinned llama.cpp checkout}" --phase $(GGUF_PHASE)

audit-errors:
	$(PYTHON) -m eval.error_audit

human-validate:
	$(PYTHON) -m eval.human_set --manifest "$(or $(HUMAN_SET),$(HUMAN_TEMPLATE))"

human-eval:
	@test -n "$(HUMAN_SET)" || { echo "Set HUMAN_SET to a real human-written manifest; the template is never scored." >&2; exit 2; }
	@: "$${LLAMA_CPP_DIR:?Set LLAMA_CPP_DIR to pinned llama.cpp checkout}"
	$(PYTHON) -m eval.human_set --manifest "$(HUMAN_SET)" --output bench/human_eval/$(HUMAN_TAG)_validation.json
	$(PYTHON) -m eval.human_eval predict-hf --manifest "$(HUMAN_SET)" --adapter "$(HUMAN_ADAPTER)" \
	  --output runs/human_eval/$(HUMAN_TAG)_adapter.json
	$(PYTHON) -m eval.human_eval predict-gguf --manifest "$(HUMAN_SET)" --model "$(HUMAN_GGUF)" \
	  --server-binary "$$LLAMA_CPP_DIR/build/bin/llama-server" --grammar "$(HUMAN_GRAMMAR)" \
	  --output runs/human_eval/$(HUMAN_TAG)_gguf.json
	$(PYTHON) -m eval.human_eval score --manifest "$(HUMAN_SET)" \
	  --predictions runs/human_eval/$(HUMAN_TAG)_adapter.json --output bench/human_eval/$(HUMAN_TAG)_adapter_score.json
	$(PYTHON) -m eval.human_eval score --manifest "$(HUMAN_SET)" \
	  --predictions runs/human_eval/$(HUMAN_TAG)_gguf.json --output bench/human_eval/$(HUMAN_TAG)_gguf_score.json
	$(PYTHON) -m eval.human_eval compare --manifest "$(HUMAN_SET)" \
	  --left runs/human_eval/$(HUMAN_TAG)_adapter.json --right runs/human_eval/$(HUMAN_TAG)_gguf.json \
	  --output bench/human_eval/$(HUMAN_TAG)_adapter_vs_gguf.json
