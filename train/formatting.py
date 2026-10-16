"""One chat-template contract for SFT, DPO, and evaluation."""

from __future__ import annotations

import json

from tools_api.executor import SCHEMAS


def messages(row: dict, include_answer: bool) -> list[dict[str, str]]:
    if row["source"] == "general_chat":
        system = "Answer the user helpfully and concisely."
        answer = row["target"]["text"]
    else:
        tools = row.get("tools") or [{"name": name, "parameters": {
            "type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False,
        }} for name, properties in SCHEMAS.items()]
        catalog = [{"name": tool["name"], "parameters": tool["parameters"]}
                   for tool in tools]
        system = ("You control only this mock local device. Return one JSON object with "
                  "calls, clarification, or refusal. Calls must use an available tool and "
                  "its exact argument schema. No prose outside JSON. Available tools: " +
                  json.dumps(catalog, separators=(",", ":"), sort_keys=True))
        answer = json.dumps(row["target"], separators=(",", ":"), sort_keys=True)
    turns = [{"role": "system", "content": system},
             {"role": "user", "content": row["prompt"]}]
    if include_answer:
        turns.append({"role": "assistant", "content": answer})
    return turns


def render(row: dict, tokenizer, include_answer: bool = True) -> str:
    return tokenizer.apply_chat_template(messages(row, include_answer), tokenize=False,
                                         add_generation_prompt=not include_answer)
