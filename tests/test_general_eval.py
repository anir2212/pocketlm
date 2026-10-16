"""Keep broad-capability evaluation on the pinned local model snapshot."""

import json
import sys
from types import SimpleNamespace

from eval import general_eval


def test_general_eval_uses_cached_pinned_model(monkeypatch, tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    calls = []

    def simple_evaluate(**kwargs):
        calls.append(kwargs)
        return {"results": {"arc_easy": {"acc,none": 0.5}},
                "versions": {"arc_easy": 1}, "n-samples": {"arc_easy": 2}}

    monkeypatch.setattr(general_eval, "cached_snapshot",
                        lambda model, revision: str(snapshot))
    monkeypatch.setitem(sys.modules, "lm_eval", SimpleNamespace(
        simple_evaluate=simple_evaluate))
    monkeypatch.setitem(sys.modules, "lm_eval.utils", SimpleNamespace(
        handle_non_serializable=str))

    output = tmp_path / "result.json"
    report = general_eval.evaluate(output, adapter, limit=2)

    assert calls[0]["model_args"]["pretrained"] == str(snapshot)
    assert calls[0]["model_args"]["peft"] == str(adapter.resolve())
    assert "revision" not in calls[0]["model_args"]
    assert calls[0]["limit"] == 2
    assert report["model"] == general_eval.BASE
    assert report["revision"] == general_eval.REVISION
    assert json.loads(output.read_text()) == report
