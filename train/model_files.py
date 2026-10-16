"""Resolve a pinned cached model snapshot without network calls."""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download


def cached_snapshot(model_id: str, revision: str) -> str:
    local = Path(model_id)
    if local.is_dir():
        return str(local)
    try:
        return snapshot_download(repo_id=model_id, revision=revision, local_files_only=True)
    except Exception as error:
        raise FileNotFoundError(
            f"Pinned model {model_id}@{revision} is not cached; "
            "download it before an offline run"
        ) from error
