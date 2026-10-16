"""Deterministically generate synthetic device requests and held-out templates/tools."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import yaml

from tools_api.executor import SCHEMAS, validate_target

PREFIXES = ("please", "could you", "on this laptop", "for this session", "right now",
            "before I start", "I need you to", "on my setup", "for my workspace", "quickly")
SUFFIXES = ("", "for my game", "for work", "before I play", "in my room", "today",
            "for this session", "if possible", "on this computer", "while I am here")
OUT_OF_SCOPE = (
    "book a flight", "send an email", "order dinner", "trade a stock",
    "tell me a medical diagnosis", "unlock a bank account", "write a tax return",
    "reserve a hotel", "post to a social account", "transfer money", "buy groceries",
    "change a calendar event", "call a taxi", "print confidential files", "write a legal contract",
    "reset a website password", "download a movie", "message my coworkers",
    "order prescription medicine", "delete a cloud account", "purchase concert tickets",
    "pay an electricity bill", "schedule a dentist visit", "check my bank balance",
)
REFUSAL = "I can only control the local demo device."


def arguments(tool: str, rng: random.Random) -> dict:
    result = {}
    for name, spec in SCHEMAS[tool].items():
        if "enum" in spec:
            result[name] = rng.choice(spec["enum"])
        elif spec["type"] == "integer":
            result[name] = rng.randrange(spec["minimum"], spec["maximum"] + 1)
        elif spec["type"] == "boolean":
            result[name] = bool(rng.randrange(2))
        else:
            raise ValueError(f"unsupported slot type: {spec}")
    return result


def phrase(tool: str, args: dict, heldout_template: bool) -> str:
    if tool == "set_lighting":
        return (f"make my {args['zone']} glow {args['color']} in {args['effect']} mode"
                if heldout_template else
                f"set {args['zone']} lighting to {args['color']} with {args['effect']} effect")
    if tool == "set_dpi":
        return (f"adjust pointer sensitivity to {args['dpi']} DPI" if heldout_template else
                f"set mouse DPI to {args['dpi']}")
    if tool == "set_macro":
        return (f"bind {args['action']} to {args['key']}" if heldout_template else
                f"set {args['key']} macro to {args['action']}")
    if tool in {"set_fan_profile", "set_performance_profile"}:
        noun = "fan" if tool == "set_fan_profile" else "performance"
        return (f"switch {noun} into {args['profile']} mode" if heldout_template else
                f"set {noun} profile to {args['profile']}")
    if tool == "get_battery":
        return "how much charge remains?" if heldout_template else "show battery level"
    if tool in {"launch_app", "close_app"}:
        verb = "open" if tool == "launch_app" else "quit"
        return (f"{verb} the {args['name']} app" if heldout_template else
                f"{'launch' if tool == 'launch_app' else 'close'} {args['name']}")
    if tool in {"set_volume", "set_brightness"}:
        noun = "volume" if tool == "set_volume" else "screen brightness"
        return (f"move {noun} to {args['percent']} percent" if heldout_template else
                f"set {noun} to {args['percent']}%")
    if tool == "toggle_mic":
        return (("unmute" if args["enabled"] else "mute") + " the microphone"
                if heldout_template else
                ("turn on" if args["enabled"] else "turn off") + " the microphone")
    if tool == "get_device_status":
        return "what settings are active?" if heldout_template else "show device status"
    if tool == "set_sleep_timer":
        return (f"sleep after {args['minutes']} minutes" if heldout_template else
                f"set sleep timer to {args['minutes']} minutes")
    raise ValueError(tool)


def generate(config: dict) -> list[dict]:
    rng = random.Random(config["seed"])
    heldout_tools = set(config["heldout_tools"])
    train_tools = sorted(set(SCHEMAS) - heldout_tools)
    rows = []
    count = int(config["domain_examples"])
    test_period = round(1 / config.get("test_fraction", 0.1))
    for index in range(count):
        holdout = index % test_period == 0
        kind = rng.random()
        if kind < config["out_of_scope_fraction"]:
            prompt = f"{rng.choice(PREFIXES)} {rng.choice(OUT_OF_SCOPE)} {rng.choice(SUFFIXES)}"
            target = {"refusal": REFUSAL}
            template = "out_of_scope"
        elif kind < config["out_of_scope_fraction"] + config["clarification_fraction"]:
            tool = rng.choice(["set_dpi", "set_lighting", "set_volume", "set_brightness"])
            noun = {"set_dpi": "DPI", "set_lighting": "lights",
                    "set_volume": "volume", "set_brightness": "brightness"}[tool]
            prompt = f"{rng.choice(PREFIXES)} change my {noun} {rng.choice(SUFFIXES)}"
            target = {"clarification": f"What {noun} value would you like?"}
            template = "missing_argument"
        else:
            use_heldout_tool = holdout and (index // test_period) % 2 == 0
            tool = rng.choice(sorted(heldout_tools) if use_heldout_tool else train_tools)
            args = arguments(tool, rng)
            calls = [{"tool": tool, "arguments": args}]
            prompt = phrase(tool, args, holdout)
            if rng.random() < config["multi_call_fraction"]:
                second = rng.choice([name for name in train_tools if name != tool])
                second_args = arguments(second, rng)
                calls.append({"tool": second, "arguments": second_args})
                prompt += " and " + phrase(second, second_args, holdout)
            target = {"calls": calls}
            template = "heldout" if holdout else "train"
            prompt = f"{rng.choice(PREFIXES)} {prompt} {rng.choice(SUFFIXES)}".strip()
        assert validate_target(target)
        rows.append({"id": f"domain_{index:05d}", "source": "synthetic_device",
                     "split": "test" if holdout else "train", "template": template,
                     "prompt": " ".join(prompt.split()), "target": target})
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/corpus.yaml"))
    parser.add_argument("--output", type=Path, default=Path("data/raw/domain.jsonl"))
    args = parser.parse_args()
    rows = generate(yaml.safe_load(args.config.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    print(json.dumps({"rows": len(rows), "test": sum(row["split"] == "test" for row in rows)}))
