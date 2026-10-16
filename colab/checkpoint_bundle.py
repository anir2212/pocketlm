"""Bounded QLoRA Trainer checkpoint handoff between disposable Colab runtimes."""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
import zipfile
from pathlib import Path

RUN_NAME = "sft_qlora_r16_2k"
MAX_BYTES = 512 * 1024 * 1024
REQUIRED = {
    "adapter_config.json", "adapter_model.safetensors", "optimizer.pt",
    "scheduler.pt", "rng_state.pth", "trainer_state.json",
}
DATA = Path("data/processed/train.jsonl")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def latest_complete(output: Path) -> Path:
    """Ignore a partially written newest checkpoint and select the latest complete one."""
    candidates = []
    for path in (output / "trainer").glob("checkpoint-*"):
        if not path.is_dir() or not re.fullmatch(r"checkpoint-[1-9][0-9]*", path.name):
            continue
        step = int(path.name.split("-")[1])
        if not all((path / name).is_file() and (path / name).stat().st_size for name in REQUIRED):
            continue
        try:
            state = json.loads((path / "trainer_state.json").read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if state.get("global_step") == step:
            candidates.append((step, path))
    if not candidates:
        raise FileNotFoundError("No complete QLoRA checkpoint is available yet")
    return max(candidates)[1]


def export_latest(output: Path, destination: Path, data: Path = DATA) -> Path:
    if output.name != RUN_NAME:
        raise ValueError("only the QLoRA run may be exported")
    checkpoint = latest_complete(output)
    files = [output / "run_config.yaml", output / "source_config.yaml",
             output / "train_data.sha256"]
    files.extend(path for path in checkpoint.rglob("*") if path.is_file())
    if any(not path.is_file() or path.is_symlink() for path in files):
        raise ValueError("checkpoint contains missing or linked files")
    if (output / "train_data.sha256").read_text().strip() != digest(data):
        raise ValueError("checkpoint training data differs")
    size = sum(path.stat().st_size for path in files)
    if size > MAX_BYTES:
        raise ValueError(f"checkpoint is {size} bytes, above the {MAX_BYTES}-byte handoff limit")
    manifest = {
        "version": 1, "run": RUN_NAME, "checkpoint": checkpoint.name,
        "data_sha256": digest(data),
        "files": {path.relative_to(output).as_posix(): {
            "size": path.stat().st_size, "sha256": digest(path)} for path in files},
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest, sort_keys=True))
        for path in files:
            archive.write(path, path.relative_to(output).as_posix())
    return destination


def restore(archive_path: Path, output: Path, data: Path = DATA) -> Path:
    """Verify size, hashes, run identity, and data before extracting one checkpoint."""
    if output.name != RUN_NAME or (output.exists() and any(output.iterdir())):
        raise ValueError("restore needs an empty QLoRA run directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.namelist()
        if len(members) != len(set(members)) or "manifest.json" not in members:
            raise ValueError("duplicate or missing checkpoint archive entries")
        if archive.getinfo("manifest.json").file_size > 128 * 1024:
            raise ValueError("checkpoint manifest is too large")
        manifest = json.loads(archive.read("manifest.json"))
        checkpoint_name = manifest.get("checkpoint", "")
        if (manifest.get("version") != 1 or manifest.get("run") != RUN_NAME or
                not re.fullmatch(r"checkpoint-[1-9][0-9]*", checkpoint_name) or
                manifest.get("data_sha256") != digest(data)):
            raise ValueError("checkpoint run, version, or training data differs")
        expected = manifest["files"]
        prefix = f"trainer/{checkpoint_name}/"
        if set(members) != set(expected) | {"manifest.json"} or not REQUIRED <= {
            name.removeprefix(prefix) for name in expected if name.startswith(prefix)
        } or not {"run_config.yaml", "source_config.yaml",
                  "train_data.sha256"} <= set(expected):
            raise ValueError("checkpoint archive is incomplete")
        if any(name not in {"run_config.yaml", "source_config.yaml",
                            "train_data.sha256"} and
               not name.startswith(prefix) for name in expected):
            raise ValueError("unexpected checkpoint archive path")
        if sum(info["size"] for info in expected.values()) > MAX_BYTES:
            raise ValueError("checkpoint archive exceeds the handoff limit")
        with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
            staged = Path(temporary) / RUN_NAME
            for name, metadata in expected.items():
                if (".." in Path(name).parts or name.startswith("/") or "\\" in name or
                        archive.getinfo(name).file_size != metadata["size"]):
                    raise ValueError("unsafe checkpoint archive entry")
                target = staged / name
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(name) as source, target.open("wb") as sink:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        sink.write(chunk)
                if digest(target) != metadata["sha256"]:
                    raise ValueError(f"checkpoint hash differs: {name}")
            if output.exists():
                output.rmdir()
            staged.rename(output)
    return output / "trainer" / checkpoint_name
