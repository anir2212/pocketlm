"""Protect audit denominators and the exclusive error taxonomy."""

import json

import pytest

from eval.error_audit import audit, classify


def test_classify_structured_failures():
    gold = {"calls": [{"tool": "set_dpi", "arguments": {"dpi": 800}}]}
    examples = {
        "invalid_json_object": "not JSON",
        "invalid_target_schema": '{"calls":[{"tool":"set_dpi","arguments":{"dpi":999}}]}',
        "wrong_response_kind": '{"refusal":"I can only control the local demo device."}',
        "wrong_tool_sequence": '{"calls":[{"tool":"set_volume","arguments":{"percent":50}}]}',
        "wrong_arguments_or_payload": '{"calls":[{"tool":"set_dpi","arguments":{"dpi":1200}}]}',
        "exact": json.dumps(gold),
    }
    for expected, raw in examples.items():
        assert classify(gold, raw) == expected


def test_audit_rejects_misaligned_gold_and_counts_each_mode(tmp_path):
    input_path = tmp_path / "test.jsonl"
    rows = [
        {"id": "one", "source": "synthetic_device", "target": {
            "calls": [{"tool": "set_dpi", "arguments": {"dpi": 800}}]}},
        {"id": "two", "source": "synthetic_device", "target": {
            "refusal": "I can only control the local demo device."}},
    ]
    input_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    for mode in ("plain", "grammar"):
        predictions = [
            {"id": row["id"], "gold": row["target"],
             "content": json.dumps(row["target"]) if mode == "grammar" else "bad"}
            for row in rows
        ]
        (runs_dir / f"gguf_eval_F16_{mode}.json").write_text(json.dumps({
            "model": "artifacts/model-F16.gguf",
            "grammar": "quant/grammar.gbnf" if mode == "grammar" else None,
            "predictions": predictions,
        }))
    report = audit(input_path, runs_dir, tmp_path / "audit.json", formats=("F16",))
    assert report["modes"]["F16_plain"]["counts"] == {"invalid_json_object": 2}
    assert report["modes"]["F16_grammar"]["counts"] == {"exact": 2}
    assert report["modes"]["F16_plain"]["strata"]["set_dpi"]["errors"] == 1

    path = runs_dir / "gguf_eval_F16_plain.json"
    broken = json.loads(path.read_text())
    broken["predictions"][0]["gold"] = rows[1]["target"]
    path.write_text(json.dumps(broken))
    with pytest.raises(ValueError, match="gold targets differ"):
        audit(input_path, runs_dir, tmp_path / "audit.json", formats=("F16",))
