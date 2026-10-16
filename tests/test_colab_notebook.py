"""Prevent default Colab setup and accidental reruns from spending GPU quota."""

import json
import subprocess
from pathlib import Path

import pytest

NOTEBOOK = Path(__file__).resolve().parents[1] / "colab/Run_PocketLM.ipynb"


def training_cell(kind: str = "none") -> str:
    notebook = json.loads(NOTEBOOK.read_text())
    source = "".join(notebook["cells"][4]["source"])
    return source.replace("RUN_KIND = 'none'", f"RUN_KIND = '{kind}'")


def test_default_does_not_touch_gpu_or_install_packages(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: pytest.fail(
        "default Colab run must not launch a subprocess"))
    exec(compile(training_cell(), str(NOTEBOOK), "exec"), {})


def test_incomplete_run_requires_deliberate_restart(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    partial = tmp_path / "runs/sft_qlora_r16_2k"
    partial.mkdir(parents=True)
    (partial / "run_config.yaml").write_text("partial")
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: pytest.fail(
        "incomplete Colab run must not silently restart"))
    with pytest.raises(FileNotFoundError, match="No complete QLoRA checkpoint"):
        exec(compile(training_cell("qlora"), str(NOTEBOOK), "exec"), {})


def test_complete_run_skips_training_and_can_be_downloaded(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    complete = tmp_path / "runs/sft_qlora_r16_2k"
    (complete / "adapter").mkdir(parents=True)
    (complete / "metrics.json").write_text('{"steps_completed":950}')
    (complete / "adapter/adapter_model.safetensors").write_bytes(b"adapter")
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: pytest.fail(
        "completed Colab run must not retrain"))
    exec(compile(training_cell("qlora"), str(NOTEBOOK), "exec"), {})
