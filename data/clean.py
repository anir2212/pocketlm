"""Schema, length, language, and obvious contact-string filters for corpus rows."""

from __future__ import annotations

import re

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, UnknownType

from tools_api.executor import validate_target

EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
PHONE = re.compile(r"(?:\+?\d[\s.-]?){10,}")


def clean_row(row: dict) -> tuple[dict | None, str | None]:
    prompt = row.get("prompt")
    target = row.get("target")
    if not isinstance(prompt, str) or not isinstance(target, dict):
        return None, "shape"
    prompt = " ".join(prompt.split())
    if not 8 <= len(prompt) <= 4000:
        return None, "length"
    if (EMAIL.search(prompt) or PHONE.search(prompt) or
            EMAIL.search(str(target)) or PHONE.search(str(target))):
        return None, "pii_like"
    letters = [character for character in prompt if character.isalpha()]
    if letters and sum(character.isascii() for character in letters) / len(letters) < 0.8:
        return None, "language"
    if row.get("source") == "general_chat":
        if set(target) != {"text"} or not isinstance(target["text"], str):
            return None, "schema"
    elif row.get("source") == "general_function":
        tools = row.get("tools", [])
        calls = target.get("calls")
        if not isinstance(tools, list) or not isinstance(calls, list) or not calls:
            return None, "schema"
        specs = {tool.get("name"): tool.get("parameters") for tool in tools}
        try:
            if not all(call["tool"] in specs and Draft202012Validator(
                specs[call["tool"]]).is_valid(call["arguments"]) for call in calls):
                return None, "schema"
        except (KeyError, TypeError, ValueError, SchemaError, UnknownType):
            return None, "schema"
    elif not validate_target(target):
        return None, "schema"
    return {**row, "prompt": prompt}, None
