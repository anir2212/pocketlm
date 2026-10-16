"""Gradio mock-device panel backed by local grammar-aware llama.cpp inference."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from transformers import AutoTokenizer

from eval.tool_eval import BASE, REVISION
from quant.runtime import GGUFServer
from tools_api.executor import MockDevice, validate_target
from train.formatting import render
from train.model_files import cached_snapshot

COLORS = {"purple", "blue", "green", "red", "white"}


def panel(state: dict) -> str:
    lighting = state.get("lighting", {})
    color = lighting.get("keyboard", ["purple"])[0]
    if color not in COLORS:
        color = "purple"
    return (
        '<div style="padding:24px;border:2px solid #333;border-radius:12px">'
        '<div style="font-size:22px">Mock device</div>'
        f'<div style="background:{html.escape(color)};height:30px;margin:12px 0">'
        '</div>'
        f'<p>DPI: {int(state.get("dpi", 800))} · '
        f'Volume: {int(state.get("volume", 50))}% · '
        f'Brightness: {int(state.get("brightness", 70))}%</p>'
        f'<p>Fan: {html.escape(str(state.get("fan", "balanced")))}</p></div>'
    )


def apply_response(raw: str, current_state: dict) -> tuple[str, dict, str]:
    try:
        output = json.loads(raw)
    except json.JSONDecodeError:
        return ("Model output is not JSON; device state unchanged.", current_state,
                panel(current_state))
    if not validate_target(output):
        return ("Model output failed the tool schema; device state unchanged.", current_state,
                panel(current_state))
    device = MockDevice()
    device.state = json.loads(json.dumps(current_state))
    updated = device.execute(output)
    if "calls" in output:
        message = f"Executed {len(output['calls'])} mock tool call(s)."
    else:
        message = output.get("refusal") or output.get("clarification")
    return message, updated, panel(updated)


def create_app(server: GGUFServer, grammar: str):
    try:
        import gradio as gr
    except ImportError as error:
        raise RuntimeError("Install demo dependencies with uv sync --extra demo") from error
    tokenizer = AutoTokenizer.from_pretrained(cached_snapshot(BASE, REVISION))
    initial = MockDevice().state

    def respond(instruction: str, use_grammar: bool, state: dict):
        row = {"source": "synthetic_device", "prompt": instruction}
        completion = server.complete(render(row, tokenizer, include_answer=False),
                                     grammar=grammar if use_grammar else None)
        raw = completion["content"]
        message, updated, display = apply_response(raw, state)
        return raw, message, updated, display

    with gr.Blocks(title="PocketLM mock device") as app:
        gr.Markdown("# PocketLM device demo\nAll controls affect a mock device in this page.")
        state = gr.State(initial)
        instruction = gr.Textbox(label="Instruction", placeholder="Set mouse DPI to 1600")
        use_grammar = gr.Checkbox(value=True, label="Constrain output with tool grammar")
        run = gr.Button("Run")
        raw = gr.Code(label="Model JSON", language="json")
        status = gr.Textbox(label="Result")
        display = gr.HTML(panel(initial))
        device_json = gr.JSON(label="Mock device state", value=initial)
        run.click(respond, [instruction, use_grammar, state],
                  [raw, status, state, display]).then(lambda value: value, state, device_json)
    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--server-binary", type=Path, required=True)
    parser.add_argument("--grammar", type=Path, default=Path("quant/grammar.gbnf"))
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    with GGUFServer(args.server_binary, args.model) as server:
        create_app(server, args.grammar.read_text()).launch(
            server_name="127.0.0.1", server_port=args.port,
        )
