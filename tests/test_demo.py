"""Mock-device panel never mutates state for invalid model output."""

import json

from demo.app import apply_response
from tools_api.executor import MockDevice


def test_demo_executes_valid_call_and_rejects_invalid_call():
    initial = MockDevice().state
    valid = json.dumps({"calls": [{"tool": "set_dpi", "arguments": {"dpi": 1600}}]})
    message, updated, display = apply_response(valid, initial)
    assert message == "Executed 1 mock tool call(s)."
    assert updated["dpi"] == 1600
    assert "DPI: 1600" in display
    assert initial["dpi"] == 800

    invalid = json.dumps({"calls": [{"tool": "set_dpi", "arguments": {"dpi": 9999}}]})
    _, after, _ = apply_response(invalid, updated)
    assert after == updated
