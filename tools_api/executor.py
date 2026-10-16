"""Validate model output against a local device API; never call real system APIs."""

from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator

SCHEMAS = json.loads(Path(__file__).with_name("schema.json").read_text())


def call_schema(name: str) -> dict:
    if name not in SCHEMAS:
        raise ValueError(f"unknown tool: {name}")
    return {"type": "object", "properties": SCHEMAS[name],
            "required": list(SCHEMAS[name]), "additionalProperties": False}


def validate_target(target: dict) -> bool:
    if not isinstance(target, dict):
        return False
    if set(target) == {"refusal"}:
        return isinstance(target["refusal"], str) and bool(target["refusal"].strip())
    if set(target) == {"clarification"}:
        return isinstance(target["clarification"], str) and bool(target["clarification"].strip())
    calls = target.get("calls")
    if set(target) != {"calls"} or not isinstance(calls, list) or not calls:
        return False
    for call in calls:
        if not isinstance(call, dict) or set(call) != {"tool", "arguments"}:
            return False
        try:
            schema = call_schema(call["tool"])
        except (ValueError, TypeError):
            return False
        if not Draft202012Validator(schema).is_valid(call["arguments"]):
            return False
    return True


class MockDevice:
    def __init__(self) -> None:
        self.state: dict = {
            "lighting": {}, "dpi": 800, "macros": {}, "fan": "balanced",
            "performance": "balanced", "battery": 76, "apps": [], "volume": 50,
            "mic_enabled": True, "brightness": 70, "sleep_minutes": None,
        }

    def execute(self, target: dict) -> dict:
        if not validate_target(target):
            raise ValueError("invalid tool output")
        if "calls" not in target:
            return dict(self.state)
        for call in target["calls"]:
            name, args = call["tool"], call["arguments"]
            if name == "set_lighting":
                self.state["lighting"][args["zone"]] = (args["color"], args["effect"])
            elif name == "set_dpi":
                self.state["dpi"] = args["dpi"]
            elif name == "set_macro":
                self.state["macros"][args["key"]] = args["action"]
            elif name == "set_fan_profile":
                self.state["fan"] = args["profile"]
            elif name == "set_performance_profile":
                self.state["performance"] = args["profile"]
            elif name == "launch_app":
                self.state["apps"] = sorted(set(self.state["apps"]) | {args["name"]})
            elif name == "close_app":
                self.state["apps"] = [app for app in self.state["apps"] if app != args["name"]]
            elif name == "set_volume":
                self.state["volume"] = args["percent"]
            elif name == "toggle_mic":
                self.state["mic_enabled"] = args["enabled"]
            elif name == "set_brightness":
                self.state["brightness"] = args["percent"]
            elif name == "set_sleep_timer":
                self.state["sleep_minutes"] = args["minutes"]
        return dict(self.state)
