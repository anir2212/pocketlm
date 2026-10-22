# Writing the human-written device-request set

This folder holds the format for an independent test set of device requests written by people. The code that checks and scores it is in `eval/human_set.py` and `eval/human_eval.py`. **No human-written rows exist yet.** `TEMPLATE_v0.*` shows the format only. Its four rows were written by the coding agent, are marked `"example": true`, and the scorer refuses them.

## Why the set matters

The existing 728-row holdout comes from the same templates as training. It cannot show how the model handles requests people phrase in their own way. A human-written set answers that question only if the text is new. Do not copy, edit or paraphrase prompts from `data/processed/`, `data/raw/` or `RESULTS.md`. Write requests the way you would type them to a laptop assistant.

## Rules for every row

- **Written by a person.** Do not use an LLM to draft, expand or paraphrase requests. Spelling fixes by the author are fine. Rows written with model help are not accepted (`written_by` must be `"human"` and `llm_assisted` must be `false`).
- **Permissioned.** Each author agrees in writing to the text being stored in this repository and used for evaluation. Record that statement in the manifest's `authors` list under a pseudonymous `author_id` such as `a01`. Do not record real names or contact details.
- **Non-sensitive.** No names, addresses, email addresses, phone numbers, account details, health information or anything private. The validator rejects obvious email and phone strings, but that check is not a privacy review. Set `non_sensitive` to `true` only after reading the row.
- **Mock tools only.** Targets use the 13 tools and allowed values in `tools_api/schema.json`. For example, DPI must be 400, 800, 1200, 1600 or 3200, and app names must be browser, notes or calculator.
- **One intended answer.** Write the JSON object the model should return:
  - a call list: `{"calls": [{"tool": "set_volume", "arguments": {"percent": 30}}]}`;
  - an out-of-scope refusal, which must be exactly `{"refusal": "I can only control the local demo device."}`;
  - a request for a missing value, which must read `{"clarification": "What <setting> value would you like?"}`, for example `"What volume value would you like?"`.
- **Alternatives only for real ambiguity.** `targets` is a list. The first item is the main answer. Add more items only when a different answer is equally correct, such as two call orders for “open notes and the calculator.” Alternatives must be the same kind (all calls, all refusals or all clarifications). If you cannot decide between a call and a clarification, rewrite or drop the request.
- **Include hard cases on purpose.** Aim for a spread across tools, multi-step requests, vague requests and out-of-scope requests. Record any deliberate choice in `collection_protocol`, for example “about 15% out-of-scope requests by design.” The validator counts the kinds; it does not judge balance.

## File format

Copy `TEMPLATE_v0.manifest.json` to a new name such as `human_v1.manifest.json`, and create the matching `human_v1.jsonl`.

Manifest fields:

| Field | Meaning |
|---|---|
| `format` | Always `pocketlm-human-eval/1`. |
| `set_version` | A new identifier for each frozen version, such as `human_v1`. Never edit a frozen file in place; copy it to `human_v2`. |
| `rows_file` | The JSONL file name, in the same folder. |
| `rows_sha256`, `tool_schema_sha256` | Leave as `""`; `--freeze` fills them in. |
| `example` | `false` for a real set. |
| `created` | Date the set was frozen (YYYY-MM-DD). |
| `authors` | One entry per author: `author_id`, `permission` (their consent statement), `consent_date`, and `seen_project_prompts` (true if they have read the generator templates or test prompts). |
| `collection_protocol` | How the requests were collected, in a few sentences. |

Each JSONL line is one request:

```json
{"id": "human_v1-0001", "prompt": "the screen's too bright, take it down to 40", "targets": [{"calls": [{"tool": "set_brightness", "arguments": {"percent": 40}}]}], "author_id": "a01", "written_at": "2026-10-21", "written_by": "human", "llm_assisted": false, "non_sensitive": true, "example": false}
```

The line above shows the shape only. Do not reuse it. An optional `notes` field may explain an unusual target.

## Check, freeze and run

```sh
make human-validate HUMAN_SET=data/human_eval/human_v1.manifest.json
.venv/bin/python -m eval.human_set --manifest data/human_eval/human_v1.manifest.json --freeze
LLAMA_CPP_DIR=/path/to/pinned/llama.cpp make human-eval HUMAN_SET=data/human_eval/human_v1.manifest.json HUMAN_TAG=human_v1
```

Validation needs the local `data/processed/train.jsonl` and `test.jsonl` (`make data` rebuilds them). It rejects duplicate IDs, invalid targets, missing provenance, and any prompt that matches a training, SFT-validation or holdout prompt exactly or after lower-casing and removing punctuation. It also checks `data/raw/domain.jsonl` when present. Close MinHash matches are listed as warnings and scored separately as `metrics_excluding_near_duplicates`.

`make human-eval` runs the selected rank-8 adapter and the Q4_K_M GGUF with grammar on the same frozen rows. It writes raw predictions to ignored `runs/human_eval/`, and scores plus the paired comparison to `bench/human_eval/`. Change `HUMAN_GGUF` or `HUMAN_GRAMMAR` only **before** looking at any results, and record the choice.

## How large the set should be

Small sets give wide intervals. With 100 requests and an accuracy near 85%, a 95% interval is roughly ±7 percentage points. A paired comparison needs enough requests on which the two models disagree. A few hundred requests from more than one author make a more useful first version than a perfectly balanced but tiny set.

## Storage

The rows file is plain text and is tracked by Git unless you choose otherwise. Commit it only if every author agreed to that storage. Do not copy it into a public snapshot unless the permission statement covers public release.
