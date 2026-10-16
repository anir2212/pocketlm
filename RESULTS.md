# PocketLM measured results

## M1: scratch decoder on TinyStories

The run used the committed `configs/tiny_gpt.yaml`, seed 7, pinned TinyStories revision `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`, 5,000 streamed training stories (4,197,328 UTF-8 bytes), and 500 streamed validation stories (406,229 bytes). The model has **12,807,680 trainable parameters**, four decoder blocks, eight attention heads, 512-wide embeddings, learned positions, and an explicit future-token mask. It trained for 1,000 steps on an Apple M3 Pro MPS device in **64.267 seconds**. Training windows were sampled from concatenated stories, and validation used five random batches at each checkpoint. These small sampled validation estimates are useful for a learning curve but are not a full-corpus benchmark.

| Step | Training loss | Validation loss | Validation perplexity per byte token |
|---:|---:|---:|---:|
| 1 | 5.6791 | 4.3112 | 74.528 |
| 100 | 2.3701 | 2.3985 | 11.006 |
| 300 | 2.3076 | 2.2263 | 9.265 |
| 500 | 1.9952 | 1.8531 | 6.380 |
| 700 | 1.7587 | 1.6519 | 5.217 |
| 1000 | 1.7140 | 1.4634 | **4.321** |

The full curve and generated sample are in [`bench/tiny_gpt_results.json`](bench/tiny_gpt_results.json). The sample has broken words and weak coherence; this run demonstrates implementation and training, not a useful assistant. Byte-level perplexity must not be compared with differently tokenized models. A causal-mask test changes future tokens and confirms earlier logits remain equal within tolerance.

## M2: corpus engineering

`make data` generated **8,000 raw synthetic device examples** with 13 validated tool schemas, then filtered exact duplicate prompts (**2,415**) and approximate MinHash 5-gram near duplicates (**75**). It kept **4,782 domain training** rows and **728 held-out domain test** rows. The training set also contains **2,391 schema-valid public function calls** and **796 general-chat** rows, for **7,969 training rows** and **8,697 total rows**. Source/token breakdown is in [`data/STATS.md`](data/STATS.md), which counted **4,104,035 Qwen chat-template tokens including the held-out test**. These are corpus counts, not model accuracy.

The domain training rows include **825 refusals**, **282 clarification requests**, and **809 multi-call requests**; the held-out set includes **106**, **71**, and **64** respectively. The final 825/4,782 refusal share is **17.3%**; deduplication changed it from the generator's 15% target. Public Glaive rows were parsed only when a declared function schema and call could be validated, and PII-like strings were filtered. Glaive and Dolly revisions are pinned in [`configs/corpus.yaml`](configs/corpus.yaml). MinHash LSH is approximate and cannot prove every similar prompt was removed. The test set holds out alternate in-scope phrasings and two device tools from device training; it is still generated from templates, not human-written requests.

## First LoRA training run and pipeline verification

The first 2,000-row rank-8 LoRA run completed on this Mac's MPS device in **3,362.9 seconds** over **950 optimizer steps** and two epochs. It used 1,898 training examples and 101 internal validation examples after excluding one overlength row; packing was disabled on this backend to preserve sample isolation. Its final training loss was **0.3078** and internal validation loss was **0.2331**. Peak sampled MPS driver allocation was **9.14 GB**. These are optimization and resource measurements, not held-out task accuracy; the adapter is at `runs/sft_r8_lr1e4_2k/adapter`.

A local rank-16 comparison attempt on 13 October 2026 reached **22 of 950** planned optimizer steps before it was interrupted: host free disk fell from roughly 7.4 GiB to as little as 2.1 GiB while macOS swap exceeded 18 GiB. Other project checks were also running, so this observation does not isolate the rank-16 model's memory use. The attempt saved no adapter or complete-run metrics and must not be compared with the finished rank-8 run. The rank-16 and rank-32 comparisons remain pending until adequate disk headroom and a quieter memory window are available.

## Held-out structured-output evaluation

The trained rank-8 adapter was evaluated once on all **728 synthetic held-out device requests** with greedy generation on CPU. The raw predictions and seeded 1,000-resample bootstrap intervals are in `runs/sft_tool_eval_full.json`. The split contains 551 tool-call targets, 106 refusals, and 71 clarification targets. These metrics use exact parsed JSON object equality, so formatting and object key order do not affect exact match.

| Metric | Result | 95% bootstrap interval |
|---|---:|---:|
| JSON validity (all 728) | 99.59% | 99.04–100.00% |
| Target-schema validity (all 728) | 96.29% | 94.92–97.66% |
| Full-call exact match (551 call targets) | 83.48% | 80.37–86.57% |
| Overall target exact match (all 728) | 87.50% | 85.03–89.97% |
| Tool-name exact match (551 call targets) | 88.02% | 85.33–90.61% |
| Argument micro-F1 | 90.43% | 87.57–93.10% |
| Hallucinated tool name (all 728) | 1.51% | 0.69–2.34% |
| Correct refusal (106 targets) | 100.00% | 100.00–100.00% |
| Correct clarification (71 targets) | 100.00% | 100.00–100.00% |

The confidence intervals reflect resampling this generated test set; they do not measure transfer to human-written requests or real devices. The refusal and clarification categories are template generated, so their perfect scores should not be read as broad safety performance. The base-model five-request smoke is too small to support a before/after accuracy claim.

A separate two-step isolated LoRA smoke completed with evaluation loss 1.1668, but it is only an API/runtime check and is not comparable to the trained adapter. A five-request base-model structured-output smoke generated zero JSON-valid replies; five prompts are too few to estimate task accuracy. A two-request **base F16 GGUF plus grammar** smoke parsed the grammar in pinned llama.cpp, produced two schema-valid JSON replies, and exercised held-out metric scoring. Those results are also too small to estimate task accuracy and do not measure the tuned model.

The pinned llama.cpp server streamed a valid grammar-constrained refusal and returned timing fields; `llama-bench` produced separate 512-token prompt and 128-token generation rows. The ARC Easy/HellaSwag harness completed two-question smoke runs with and without a chat template; those scores are diagnostic only. The Pareto report rendered from temporary smoke fixtures, proving the reporting code can read the native benchmark format. Its temporary chart is not an experiment result.

The first rank-8 adapter has been merged and converted to F16, Q8_0, Q5_K_M, and Q4_K_M GGUF files. Their measured file sizes are **948.10, 506.47, 400.62, and 379.38 MiB** respectively; Q4_K_M uses 40.0% of the F16 file size. This is a disk-size comparison, not a memory or speed result. The Q4_K_M model was evaluated through the CPU llama.cpp server with the generated GBNF grammar on **all 728** held-out requests. The raw outputs and timing fields are in `runs/gguf_eval_Q4_K_M_grammar.json`.

| Metric | Q4_K_M + grammar | 95% bootstrap interval |
|---|---:|---:|
| JSON and target-schema validity | 100.00% | 100.00–100.00% |
| Full-call exact match (551 call targets) | 82.03% | 78.69–85.47% |
| Overall target exact match (all 728) | 86.40% | 83.79–89.15% |
| Tool-name exact match (551 call targets) | 86.57% | 83.61–89.42% |
| Argument micro-F1 | 90.54% | 87.85–92.98% |
| Hallucinated tool name | 0.00% | 0.00–0.00% |

Compared request by request with the Hugging Face adapter's greedy run, Q4_K_M plus grammar got 629 exact targets versus 637, with 619 correct in both, 18 correct only in Hugging Face, and 10 correct only in GGUF. The paired overall exact-match difference is **−1.10 percentage points**, with a seeded 1,000-resample paired bootstrap interval of **−2.61 to +0.41 points**. This comparison changes both weight precision and decoding constraint, so it does not isolate quantization's effect. The 100% grammar validity is a property of this grammar and sample, not evidence that the selected valid call is always right. The other two SFT comparison runs, NF4 QLoRA, full DPO, general-capability evaluation, and remaining GGUF modes/latency matrix remain **pending measured runs**. No speedup or real-world quality claim is made yet.

The matching **Q4_K_M without grammar** run on the same 728 requests is in `runs/gguf_eval_Q4_K_M_plain.json`: JSON validity 98.76%, target-schema validity 95.33%, full-call exact match 79.67%, overall target exact match 84.62%, and argument micro-F1 85.15%. Comparing Q4 with and without grammar request by request, grammar improved overall exact targets from **616 to 629** (**+1.79 percentage points**, paired 95% bootstrap interval **+0.96 to +2.75 points**) and schema-valid targets from **694 to 728** (**+4.67 points**, interval **+3.30 to +6.32 points**). Those intervals describe this synthetic holdout and decoding setup; they are not a general guarantee about grammar-constrained generation.

A post-hoc exclusive error breakdown illustrates the remaining failure mode. Among the 112 plain Q4 misses, 9 were invalid JSON, 25 were valid JSON outside the target schema, 54 had wrong tool names/order, and 24 had wrong arguments or another exact-target difference. Among the 99 grammar Q4 misses, syntax/schema failures were zero, but 74 had wrong tool names/order and 25 had wrong arguments or another exact-target difference. The grammar forces valid output forms; it cannot guarantee that the model selects the correct tool.

The matching **F16 with grammar** run on the same 728 requests is in `runs/gguf_eval_F16_grammar.json`. It reached 100% JSON/schema validity, **85.12%** full-call exact match on the 551 call targets (95% bootstrap interval 82.17–88.08%), **88.74%** overall target exact match (86.54–91.07%), 90.20% tool-name exact match on calls, and 93.29% argument micro-F1. F16 got 646 exact targets; Q4 got 629. In a paired comparison with the same llama.cpp runtime and grammar, Q4 was **−2.34 percentage points** lower on overall exact targets, with a 95% paired bootstrap interval of **−3.85 to −0.96 points**; 624 targets were correct in both, 22 only in F16, and 5 only in Q4. The saved `bench/f16_grammar_vs_q4_grammar.json` records the paired counts and hashes of both prediction files. This measures an accuracy cost of Q4 on this synthetic set and setup; it does not imply the same gap on other prompts or hardware.

The **Q5_K_M with grammar** run completed the same 728 requests in `runs/gguf_eval_Q5_K_M_grammar.json`: 100% JSON/schema validity, **87.11%** full-call exact match on the 551 calls, **90.11%** overall target exact match, 91.47% tool-name exact match, and 92.78% argument micro-F1. Q5 got 656 exact targets versus F16's 646. The paired Q5-minus-F16 overall difference was **+1.37 percentage points** (1,000-resample 95% bootstrap interval **+0.14 to +2.61 points**); 641 requests were right in both, 15 only in Q5, and 5 only in F16. The source-hashed counts are in `bench/f16_grammar_vs_q5_grammar.json`. A quantized model can score above its F16 baseline on a finite test set because rounding changes token choices; this is a measured outcome on generated prompts, not evidence that Q5 is generally more accurate. Q5 latency and the other matrix modes are still unmeasured.

The **F16 without grammar** run has now completed the same 728 requests in `runs/gguf_eval_F16_plain.json`. It reached 99.45% JSON validity, 97.80% target-schema validity, 84.21% full-call exact match on call targets, and 88.05% overall exact match (641/728). Comparing F16 with grammar on the same requests, grammar lifted exact targets by **0.69 percentage points** (paired 95% bootstrap interval **+0.14 to +1.37 points**) and target-schema validity by **2.20 points** (interval **+1.24 to +3.30 points**). All 641 plain-mode exact targets remained exact with grammar; five additional targets became exact. `bench/f16_plain_vs_grammar.json` records the paired counts and source hashes. This result depends on the generated holdout and the specific GBNF grammar.

The **Q8_0 without grammar** run completed all 728 requests in `runs/gguf_eval_Q8_0_plain.json`: 724/728 JSON-valid, 713/728 target-schema-valid, and **643/728 exact targets** (88.32%). On the same requests F16 plain had 641 exact targets. The paired Q8-minus-F16 difference was **+0.27 percentage points**, with a seeded 95% bootstrap interval of **−0.41 to +0.96 points**; 639 requests were correct in both, two only in F16, and four only in Q8. The source-hashed comparison is in `bench/f16_plain_vs_q8_plain.json`. This small, uncertain difference does not establish that Q8 improves accuracy.

On 15 October, **Q8_0 with grammar** completed the same 728 synthetic held-out requests in `runs/gguf_eval_Q8_0_grammar.json`: **728/728 JSON/schema-valid**, **647/728 exact targets** (88.87%), **85.30%** full-call exact match on 551 call targets, 90.56% tool-name exact match on calls, and 93.73% argument micro-F1. Against Q8 plain, grammar made four additional targets exact (**+0.55 percentage points**, paired 95% bootstrap interval **+0.14 to +1.10**) and fifteen additional targets schema-valid (**+2.06 points**, interval **+1.10 to +3.16**). All 643 plain exact targets stayed exact under grammar. Against F16 grammar, Q8 got 647 versus 646 exact targets; two were correct only for F16 and three only for Q8, giving a **+0.14-point** difference with a paired interval **−0.41 to +0.69 points**. This does not establish a general Q8 quality improvement. Source-hashed comparisons are in `bench/q8_plain_vs_grammar.json` and `bench/f16_grammar_vs_q8_grammar.json`. The Q8 run occurred while another pre-existing Ollama server was using substantial CPU, so its request timings are not a clean latency comparison. The repeated HTTP latency matrix remains unmeasured.

On 15 October, **Q5_K_M without grammar** completed all **728** held-out requests in `runs/gguf_eval_Q5_K_M_plain.json`: **718/728 JSON-valid**, **702/728 target-schema-valid**, and **644/728 exact targets** (88.46%). Full-call exact match was **84.94%** on the 551 call targets. On the same prompts, Q5 grammar got 656 exact targets and 728 schema-valid outputs. Grammar added **12 exact targets** (**+1.65 percentage points**, paired 95% bootstrap interval **+0.82 to +2.61**) and **26 schema-valid outputs** (**+3.57 points**, interval **+2.34 to +4.95**); no plain-mode exact target was lost. Compared with F16 plain, Q5 plain got three more exact targets (644 versus 641; **+0.41 points**, interval **−1.10 to +1.65**), while schema validity was **1.37 points lower** (interval **−2.61 to −0.27**). The exact-match interval crosses zero, so this does not establish a Q5 accuracy gain over F16 plain. The paired counts and prediction-file hashes are in `bench/q5_plain_vs_grammar.json` and `bench/f16_plain_vs_q5_plain.json`. All **eight** format/grammar quality cells now have full held-out runs; the repeated-request latency matrix is still pending. Earlier statements above describe their original checkpoints.

### Reproducible failure audit

`make audit-errors` reads the saved predictions for all eight modes, checks ordered IDs and gold targets against the same 728-row holdout, and writes source SHA-256 hashes plus exclusive error counts to [`bench/quality_error_audit.json`](bench/quality_error_audit.json). Each miss is assigned to its earliest contract failure: invalid JSON object, invalid target schema, wrong response kind, wrong tool sequence, or wrong arguments/payload. These are diagnostic labels, not independent failure probabilities; a schema-invalid output could also have selected the wrong tool.

| Mode | Exact | Invalid JSON | Invalid schema | Wrong kind | Wrong tool sequence | Wrong arguments/payload |
|---|---:|---:|---:|---:|---:|---:|
| F16 plain | 641 | 4 | 12 | 27 | 17 | 27 |
| F16 grammar | 646 | 0 | 0 | 27 | 27 | 28 |
| Q8_0 plain | 643 | 4 | 11 | 24 | 18 | 28 |
| Q8_0 grammar | 647 | 0 | 0 | 24 | 28 | 29 |
| Q5_K_M plain | 644 | 10 | 16 | 19 | 16 | 23 |
| Q5_K_M grammar | 656 | 0 | 0 | 20 | 28 | 24 |
| Q4_K_M plain | 616 | 9 | 25 | 24 | 30 | 24 |
| Q4_K_M grammar | 629 | 0 | 0 | 24 | 50 | 25 |

The largest concentrated gap is the held-out battery paraphrase: F16 and Q4 grammar missed **21/21** single `get_battery` targets, while Q5 grammar missed **17/21**. One repeated prompt pattern is “how much charge remains?”; the model often returned a valid but irrelevant clarification about volume or brightness. The generator used “show battery level” in training and “how much charge remains?” in the test split. This is evidence of poor generalization across this particular paraphrase, not a population estimate for battery queries. F16 grammar also missed **12/23** `get_device_status` and **21/64** multi-call targets. Q5 grammar reduced those to **8/23** and **20/64**; Q4 grammar missed **20/23** and **27/64**. The nearly perfect `close_app` and `set_sleep_timer` scores are on generated prompts with their tool schemas present in the system message, so they do not establish robust unseen-tool use. The audit points to paraphrase coverage and multi-call ordering as the next training-data priorities; changing this holdout after seeing these errors would invalidate direct comparisons to the existing results.

## Focused CPU size and latency comparison

The pinned llama.cpp binaries ran on this Apple M3 Pro Mac (macOS 15.3.1, arm64), with four CPU threads and zero GPU layers. `llama-bench` measured a **512-token prompt** and **128-token generation**, each over three repetitions. A separate warmed HTTP request used a 614-token prompt and ended after 15 generated tokens. The compact, source-hashed measurements are in `bench/latency_partial.json`; full raw outputs are in `runs/latency_*.json`.

| Model/mode | File MiB | Prompt tokens/s | Generation tokens/s | One-request first token | One-request end-to-end | Sampled server RSS MiB |
|---|---:|---:|---:|---:|---:|---:|
| F16 + grammar | 948.10 | 1174.11 | 101.14 | 0.662 s | 0.815 s | 1150.5 |
| Q4_K_M + grammar | 379.38 | 1074.30 | 178.62 | 0.652 s | 0.746 s | 739.2 |
| Q4_K_M plain | 379.38 | 1081.03 | 184.32 | 0.757 s | 0.862 s | 718.3 |

Q4_K_M generation throughput was **1.77× F16** in these fixed-length CPU runs, while its model file was 40.0% as large and its grammar-run sampled RSS was 35.8% lower. F16 processed the fixed prompt faster. The benchmark throughput rows do **not** apply grammar, even when paired with a grammar HTTP row; the Q4 plain/grammar throughput difference is run variation, not a measured grammar overhead. The HTTP timings are one short request each and cannot establish p50/p95 latency or a reliable grammar latency difference. RSS was sampled and can miss a brief peak. Q8_0/Q5_K_M and broader latency distributions remain unmeasured.
