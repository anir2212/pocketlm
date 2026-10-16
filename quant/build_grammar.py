"""Compile the 13 mock device schemas to a strict compact-JSON GBNF grammar."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools_api.executor import SCHEMAS

REFUSAL = "I can only control the local demo device."
CLARIFICATIONS = (
    "What DPI value would you like?", "What lights value would you like?",
    "What volume value would you like?", "What brightness value would you like?",
)


def symbol(*parts: str) -> str:
    """GBNF rule identifiers accept hyphens but not underscores."""
    return "-".join(part.replace("_", "-") for part in parts)


def terminal(value: str) -> str:
    """Quote an entire literal fragment in GBNF syntax."""
    return json.dumps(value)


def value_rule(tool: str, argument: str, schema: dict) -> tuple[str, str]:
    rule = symbol("value", tool, argument)
    if "enum" in schema:
        choices = [terminal(json.dumps(value, separators=(",", ":")))
                   for value in schema["enum"]]
    elif schema.get("type") == "integer":
        choices = [terminal(str(value)) for value in range(
            schema["minimum"], schema["maximum"] + 1)]
    elif schema.get("type") == "boolean":
        choices = [terminal("true"), terminal("false")]
    else:
        raise ValueError(f"unsupported schema: {tool}.{argument}")
    return rule, f"{rule} ::= " + " | ".join(choices)


def build() -> str:
    rules = [
        "root ::= call-output | refusal-output | clarification-output",
        'call-output ::= "{\\\"calls\\\":[" call ("," call)* "]}"',
    ]
    tool_rules = []
    for tool, properties in sorted(SCHEMAS.items()):
        fragments = []
        for index, argument in enumerate(sorted(properties)):
            value_name, value_definition = value_rule(tool, argument, properties[argument])
            rules.append(value_definition)
            prefix = ("," if index else "") + json.dumps(argument) + ":"
            fragments.extend([terminal(prefix), value_name])
        body = [terminal('{"arguments":{'), *fragments,
                terminal('},"tool":' + json.dumps(tool) + '}')]
        rule_name = symbol("tool", tool)
        rules.append(f"{rule_name} ::= " + " ".join(body))
        tool_rules.append(rule_name)
    rules.append("call ::= " + " | ".join(tool_rules))
    rules.append("refusal-output ::= " + terminal('{"refusal":' + json.dumps(REFUSAL) + '}'))
    rules.append("clarification-output ::= " + " | ".join(terminal(
        '{"clarification":' + json.dumps(text) + '}') for text in CLARIFICATIONS))
    return "\n".join(rules) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("quant/grammar.gbnf"))
    args = parser.parse_args()
    args.output.write_text(build())
    print(f"wrote {args.output}")
