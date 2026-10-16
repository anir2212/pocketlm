"""Local tool-call corpus and a small decoder trained from scratch."""
import argparse
import json
import random
import re
from pathlib import Path

TOOLS = {
    "set_light": {"zone": str, "color": str},
    "set_dpi": {"dpi": int},
    "set_fan": {"profile": str},
    "get_battery": {},
    "launch_app": {"name": str},
}
COLORS = ("purple", "blue", "green", "red")
ZONES = ("keyboard", "mouse")


def validate(call):
    if not isinstance(call, dict) or call.get("tool") not in TOOLS:
        return False
    args = call.get("arguments")
    spec = TOOLS[call["tool"]]
    return isinstance(args, dict) and args.keys() == spec.keys() and all(
        type(args[key]) is typ for key, typ in spec.items()
    )


def corpus(seed=7):
    rng = random.Random(seed)
    examples = []
    for zone in ZONES:
        for color in COLORS:
            call = {"tool": "set_light", "arguments": {"zone": zone, "color": color}}
            for pattern in ("make my {zone} {color}", "set {zone} lighting to {color}"):
                examples.append({"prompt": pattern.format(zone=zone, color=color), "target": call})
    for dpi in (400, 800, 1200, 1600, 3200):
        examples.append({"prompt": f"set mouse DPI to {dpi}", "target": {"tool": "set_dpi", "arguments": {"dpi": dpi}}})
    for profile in ("quiet", "balanced", "performance"):
        examples.append({"prompt": f"use {profile} fan mode", "target": {"tool": "set_fan", "arguments": {"profile": profile}}})
    examples.extend([
        {"prompt": "what is the battery level?", "target": {"tool": "get_battery", "arguments": {}}},
        {"prompt": "launch my browser", "target": {"tool": "launch_app", "arguments": {"name": "browser"}}},
        {"prompt": "order me lunch", "target": {"refusal": "I can only control local device settings."}},
        {"prompt": "change my DPI", "target": {"clarification": "What DPI value would you like?"}},
    ])
    assert all(validate(e["target"]) for e in examples if "tool" in e["target"])
    rng.shuffle(examples)
    return examples


def dedup(items):
    seen = set()
    out = []
    for item in items:
        key = (item["prompt"].strip().lower(), json.dumps(item["target"], sort_keys=True))
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


def build_data(path):
    rows = dedup(corpus())
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return len(rows)


def parse_request(prompt):
    """Deterministic local baseline, separate from the experimental Transformer."""
    q = prompt.lower().strip()
    color = next((c for c in COLORS if re.search(rf"\b{c}\b", q)), None)
    zone = next((z for z in ZONES if re.search(rf"\b{z}\b", q)), None)
    if zone and color:
        return {"tool": "set_light", "arguments": {"zone": zone, "color": color}}
    if "dpi" in q:
        match = re.search(r"\b(400|800|1200|1600|3200)\b", q)
        return {"tool": "set_dpi", "arguments": {"dpi": int(match.group(1))}} if match else {"clarification": "What DPI value would you like?"}
    if "fan" in q:
        profile = next((p for p in ("quiet", "balanced", "performance") if p in q), None)
        return {"tool": "set_fan", "arguments": {"profile": profile}} if profile else {"clarification": "Which fan profile would you like?"}
    if "battery" in q:
        return {"tool": "get_battery", "arguments": {}}
    if "launch" in q and "browser" in q:
        return {"tool": "launch_app", "arguments": {"name": "browser"}}
    return {"refusal": "I can only control local device settings."}


class MockDevice:
    def __init__(self):
        self.state = {"lighting": {"keyboard": "blue", "mouse": "blue"}, "dpi": 800, "fan": "balanced", "battery": 76, "last_app": None}

    def execute(self, call):
        if not validate(call):
            raise ValueError("invalid tool call")
        name, args = call["tool"], call["arguments"]
        if name == "set_light":
            if args["zone"] not in ZONES or args["color"] not in COLORS:
                raise ValueError("unsupported lighting value")
            self.state["lighting"][args["zone"]] = args["color"]
        elif name == "set_dpi":
            if args["dpi"] not in (400, 800, 1200, 1600, 3200):
                raise ValueError("unsupported DPI")
            self.state["dpi"] = args["dpi"]
        elif name == "set_fan":
            if args["profile"] not in ("quiet", "balanced", "performance"):
                raise ValueError("unsupported fan profile")
            self.state["fan"] = args["profile"]
        elif name == "launch_app":
            if args["name"] != "browser":
                raise ValueError("unsupported app")
            self.state["last_app"] = args["name"]
        return self.state.copy()


def evaluate_baseline(rows):
    correct = sum(parse_request(row["prompt"]) == row["target"] for row in rows)
    return {"correct": correct, "total": len(rows), "exact_match": correct/len(rows) if rows else 0}


def train(path, steps=30, seed=7):
    import torch
    from torch import nn
    torch.manual_seed(seed)
    text = Path(path).read_text()
    chars = sorted(set(text))
    ids = torch.tensor([chars.index(c) for c in text], dtype=torch.long)
    block = 32

    class TinyDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.token = nn.Embedding(len(chars), 48)
            self.pos = nn.Embedding(block, 48)
            self.attn = nn.MultiheadAttention(48, 4, batch_first=True)
            self.norm = nn.LayerNorm(48)
            self.head = nn.Linear(48, len(chars))

        def forward(self, x):
            t = x.shape[1]
            h = self.token(x) + self.pos(torch.arange(t))
            mask = torch.triu(torch.ones(t, t, dtype=torch.bool), diagonal=1)
            h = self.norm(h + self.attn(h, h, h, attn_mask=mask, need_weights=False)[0])
            return self.head(h)

    model = TinyDecoder()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    losses = []
    for _ in range(steps):
        starts = torch.randint(0, len(ids)-block-1, (16,))
        x = torch.stack([ids[s:s+block] for s in starts])
        y = torch.stack([ids[s+1:s+block+1] for s in starts])
        loss = nn.functional.cross_entropy(model(x).reshape(-1, len(chars)), y.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(round(loss.item(), 4))
    torch.save({"state": model.state_dict(), "chars": chars, "seed": seed, "steps": steps}, Path(path).with_suffix(".pt"))
    Path(path).with_suffix(".loss.json").write_text(json.dumps(losses))
    return losses[-1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["data", "train", "validate", "ask", "eval"])
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--prompt", default="set keyboard lighting to purple")
    args = p.parse_args()
    path = Path("data.jsonl")
    if args.action == "data":
        print(f"wrote {build_data(path)} examples")
    elif args.action == "train":
        if not path.exists():
            build_data(path)
        print(f"final training loss: {train(path, args.steps)}")
    else:
        if args.action == "validate":
            print(json.dumps({"valid": sum(validate(e["target"]) for e in corpus() if "tool" in e["target"])}))
        elif args.action == "eval":
            print(json.dumps(evaluate_baseline(corpus()), indent=2))
        else:
            result = parse_request(args.prompt)
            state = MockDevice().execute(result) if "tool" in result else None
            print(json.dumps({"result": result, "mock_device": state}, indent=2))


if __name__ == "__main__":
    main()
