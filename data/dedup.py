"""Exact and approximate word 5-gram MinHash dedup, prioritising held-out rows."""

from __future__ import annotations

import hashlib
import re

from datasketch import MinHash, MinHashLSH


def shingles(text: str) -> set[bytes]:
    words = re.findall(r"\w+", text.casefold())
    return {" ".join(words[index:index + 5]).encode()
            for index in range(max(1, len(words) - 4))}


def signature(text: str) -> MinHash:
    result = MinHash(num_perm=128, seed=17)
    for shingle in shingles(text):
        result.update(shingle)
    return result


def deduplicate(rows: list[dict], threshold: float = 0.8) -> tuple[list[dict], dict]:
    """An LSH threshold is approximate, not an exact Jaccard guarantee."""
    index = MinHashLSH(threshold=threshold, num_perm=128)
    exact: set[str] = set()
    kept = []
    counts = {"exact_removed": 0, "near_removed": 0}
    # Preserve test examples before training examples. Near duplicates in train
    # are discarded rather than allowing the model to see a held-out prompt.
    ordered = sorted(rows, key=lambda row: row.get("split") != "test")
    for row in ordered:
        normalized = " ".join(row["prompt"].casefold().split())
        digest = hashlib.sha256(normalized.encode()).hexdigest()
        if digest in exact:
            counts["exact_removed"] += 1
            continue
        fingerprint = signature(normalized)
        if index.query(fingerprint):
            counts["near_removed"] += 1
            continue
        key = str(len(kept))
        exact.add(digest)
        index.insert(key, fingerprint)
        kept.append(row)
    return kept, counts
