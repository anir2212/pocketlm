# PocketLM corpus statistics

Tokenizer: `Qwen/Qwen2.5-0.5B-Instruct` at `7ae557604adf67be50417f59c2c2f167def9a775` chat template. Counts include system catalog, user, and assistant output.

| Split | Source | Rows | Total tokens | Median tokens | Max tokens |
|---|---|---:|---:|---:|---:|
| test | synthetic_device | 728 | 464,065 | 635 | 671 |
| train | general_chat | 796 | 149,650 | 128 | 1,958 |
| train | general_function | 2,391 | 428,348 | 174 | 496 |
| train | synthetic_device | 4,782 | 3,061,972 | 638 | 676 |

## Domain target types

| Split | Refusals | Clarifications | Multiple calls |
|---|---:|---:|---:|
| train | 825 | 282 | 809 |
| test | 106 | 71 | 64 |

## Filtering and split

Raw synthetic rows: 8,000. Domain exact duplicate prompts removed: 2,415; approximate MinHash near duplicates removed: 75.

The held-out device test uses unseen prompt templates and excludes `close_app` and `set_sleep_timer` from device training rows. All accepted function calls were checked against their schema. MinHash LSH is an approximate 5-gram Jaccard filter at a configured 0.8 threshold, not a mathematical guarantee that every near duplicate is removed.

The public function and chat samples carry their original licenses; see [SOURCES.md](SOURCES.md). The generated JSONL files are local artifacts, not committed to Git.
