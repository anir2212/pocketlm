"""Check a bounded, resumable QLoRA handoff without loading a model."""

import json
import zipfile

import pytest

from colab.checkpoint_bundle import (
    REQUIRED,
    RUN_NAME,
    digest,
    export_latest,
    latest_complete,
    restore,
)


def fixture_run(tmp_path):
    output = tmp_path / RUN_NAME
    checkpoint = output / "trainer/checkpoint-100"
    checkpoint.mkdir(parents=True)
    (output / "run_config.yaml").write_text("checkpoint_steps: 100\n")
    (output / "source_config.yaml").write_text("quantization: nf4\n")
    for name in REQUIRED - {"trainer_state.json"}:
        (checkpoint / name).write_bytes(name.encode())
    (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": 100}))
    data = tmp_path / "train.jsonl"
    data.write_text('{"id":"training-only"}\n')
    (output / "train_data.sha256").write_text(digest(data) + "\n")
    return output, data


def test_export_restores_only_complete_checkpoint_with_matching_data(tmp_path):
    output, data = fixture_run(tmp_path)
    partial = output / "trainer/checkpoint-200"
    partial.mkdir()
    (partial / "adapter_config.json").write_text("partial")
    assert latest_complete(output).name == "checkpoint-100"

    archive = export_latest(output, tmp_path / "checkpoint.zip", data)
    with zipfile.ZipFile(archive) as zipped:
        names = set(zipped.namelist())
        assert "trainer/checkpoint-100/optimizer.pt" in names
        assert not any("checkpoint-200" in name for name in names)
    restored = restore(archive, tmp_path / "fresh" / RUN_NAME, data)
    assert restored.name == "checkpoint-100"
    assert (restored / "rng_state.pth").read_bytes() == b"rng_state.pth"

    changed = tmp_path / "changed.jsonl"
    changed.write_text('{"id":"different"}\n')
    with pytest.raises(ValueError, match="training data differs"):
        restore(archive, tmp_path / "other" / RUN_NAME, changed)


def test_export_refuses_oversized_checkpoint(monkeypatch, tmp_path):
    output, data = fixture_run(tmp_path)
    monkeypatch.setattr("colab.checkpoint_bundle.MAX_BYTES", 10)
    with pytest.raises(ValueError, match="handoff limit"):
        export_latest(output, tmp_path / "checkpoint.zip", data)
