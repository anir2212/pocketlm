"""Validate a versioned, human-written device-request evaluation set.

A set is a JSON manifest plus a JSONL rows file. The validator checks provenance,
target schemas, duplicate IDs and prompt overlap with the generated corpus. It never
writes or edits evaluation text; ``--freeze`` only records file hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import date
from pathlib import Path

from datasketch import MinHashLSH

from data.build_synthetic import REFUSAL
from data.clean import EMAIL, PHONE
from data.dedup import signature
from tools_api.executor import validate_target

FORMAT = "pocketlm-human-eval/1"
TOOL_SCHEMA = Path("tools_api/schema.json")
# Training and the SFT internal validation split both come from train.jsonl
# (train.sft.split_train_validation); test.jsonl holds the 728-row template holdout.
# The raw generator output also contains prompts later removed as duplicates.
REQUIRED_REFERENCES = (Path("data/processed/train.jsonl"), Path("data/processed/test.jsonl"))
OPTIONAL_REFERENCES = (Path("data/raw/domain.jsonl"),)
MANIFEST_FIELDS = {"format", "set_version", "rows_file", "rows_sha256", "tool_schema_sha256",
                   "example", "created", "authors", "collection_protocol"}
AUTHOR_FIELDS = {"author_id", "permission", "consent_date", "seen_project_prompts"}
ROW_FIELDS = {"id", "prompt", "targets", "author_id", "written_at", "written_by",
              "llm_assisted", "non_sensitive", "example"}
OPTIONAL_ROW_FIELDS = {"notes"}
ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
CLARIFICATION = re.compile(r"What [A-Za-z][A-Za-z ]{0,30} value would you like\?")
EXAMPLE_AUTHOR = "template_placeholder"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collapse(text: str) -> str:
    """The exact-overlap key: whitespace collapsed as in data.clean.clean_row."""
    return " ".join(text.split())


def normalize(text: str) -> str:
    """The normalized-overlap key: case-folded word tokens without punctuation."""
    return " ".join(re.findall(r"\w+", text.casefold()))


def kind(target: dict) -> str:
    return next(key for key in ("calls", "clarification", "refusal") if key in target)


def _target_errors(prefix: str, targets: object) -> list[str]:
    if not isinstance(targets, list) or not targets:
        return [f"{prefix}: targets must be a nonempty list"]
    errors = []
    for index, target in enumerate(targets):
        if not validate_target(target):
            errors.append(f"{prefix}: target {index} fails tools_api/schema.json")
            continue
        if "refusal" in target and target["refusal"] != REFUSAL:
            errors.append(f"{prefix}: target {index} refusal must be exactly {REFUSAL!r}")
        if "clarification" in target and not CLARIFICATION.fullmatch(target["clarification"]):
            errors.append(f"{prefix}: target {index} clarification must read "
                          "'What <setting> value would you like?'")
    if errors:
        return errors
    if len({kind(target) for target in targets}) != 1:
        errors.append(f"{prefix}: alternative targets must share the first target's kind")
    canonical = [json.dumps(target, sort_keys=True) for target in targets]
    if len(set(canonical)) != len(canonical):
        errors.append(f"{prefix}: duplicate alternative targets")
    return errors


def _row_errors(row: object, line: int, authors: dict, example: bool) -> list[str]:
    prefix = f"line {line}"
    if not isinstance(row, dict):
        return [f"{prefix}: row must be a JSON object"]
    if isinstance(row.get("id"), str):
        prefix = f"line {line} ({row['id']})"
    missing = sorted(ROW_FIELDS - set(row))
    unknown = sorted(set(row) - ROW_FIELDS - OPTIONAL_ROW_FIELDS)
    errors = [f"{prefix}: missing field {name}" for name in missing]
    errors += [f"{prefix}: unknown field {name}" for name in unknown]
    if missing:
        return errors
    if not isinstance(row["id"], str) or not ID_PATTERN.fullmatch(row["id"]):
        errors.append(f"{prefix}: id must be 1-64 letters, digits, '_', '.' or '-'")
    prompt = row["prompt"]
    if not isinstance(prompt, str) or not 8 <= len(collapse(prompt)) <= 4000:
        errors.append(f"{prefix}: prompt must be text of 8-4000 characters")
    elif EMAIL.search(prompt) or PHONE.search(prompt):
        errors.append(f"{prefix}: prompt contains an email- or phone-like string")
    errors += _target_errors(prefix, row["targets"])
    if not isinstance(row["author_id"], str) or row["author_id"] not in authors:
        errors.append(f"{prefix}: author_id is not listed in the manifest authors")
    try:
        date.fromisoformat(str(row["written_at"]))
    except ValueError:
        errors.append(f"{prefix}: written_at must be an ISO date (YYYY-MM-DD)")
    if row["non_sensitive"] is not True:
        errors.append(f"{prefix}: non_sensitive must be true (attested non-sensitive text)")
    if row["example"] is not example:
        errors.append(f"{prefix}: example must be {str(example).lower()} to match the manifest")
    if example:
        if row["written_by"] != EXAMPLE_AUTHOR:
            errors.append(f"{prefix}: template rows must say written_by={EXAMPLE_AUTHOR!r}")
    elif row["written_by"] != "human" or row["llm_assisted"] is not False:
        errors.append(f"{prefix}: scored rows must be written_by='human' with "
                      "llm_assisted=false")
    return errors


def _manifest_errors(manifest: dict) -> tuple[list[str], dict]:
    if not isinstance(manifest, dict):
        return ["manifest must be a JSON object"], {}
    errors = [f"manifest: missing field {name}" for name in sorted(MANIFEST_FIELDS - set(manifest))]
    if errors:
        return errors, {}
    if manifest["format"] != FORMAT:
        errors.append(f"manifest: format must be {FORMAT!r}")
    if not isinstance(manifest["example"], bool):
        errors.append("manifest: example must be true or false")
    if not isinstance(manifest["set_version"], str) or not ID_PATTERN.fullmatch(
        manifest["set_version"]
    ):
        errors.append("manifest: set_version must be a short identifier such as human_v1")
    if not isinstance(manifest["collection_protocol"], str) or not manifest[
        "collection_protocol"
    ].strip():
        errors.append("manifest: collection_protocol must describe how rows were written")
    authors = {}
    if not isinstance(manifest["authors"], list) or not manifest["authors"]:
        errors.append("manifest: authors must be a nonempty list")
    else:
        for index, author in enumerate(manifest["authors"]):
            if not isinstance(author, dict) or AUTHOR_FIELDS - set(author):
                errors.append(f"manifest: author {index} needs {sorted(AUTHOR_FIELDS)}")
                continue
            if not isinstance(author["author_id"], str) or not ID_PATTERN.fullmatch(
                author["author_id"]
            ):
                errors.append(f"manifest: author {index} author_id must be a short pseudonym")
                continue
            try:
                date.fromisoformat(str(author["consent_date"]))
            except ValueError:
                errors.append(f"manifest: author {index} consent_date must be YYYY-MM-DD")
            if not isinstance(author["permission"], str) or not author["permission"].strip():
                errors.append(f"manifest: author {index} permission statement is empty")
            if not isinstance(author["seen_project_prompts"], bool):
                errors.append(f"manifest: author {index} seen_project_prompts must be boolean")
            if author["author_id"] in authors:
                errors.append(f"manifest: duplicate author_id {author['author_id']}")
            authors[author["author_id"]] = author
    return errors, authors


def _reference_index(paths: list[Path]) -> tuple[dict, dict, MinHashLSH, list[dict], list[str]]:
    exact: dict[str, str] = {}
    normalized: dict[str, str] = {}
    lsh = MinHashLSH(threshold=0.8, num_perm=128)
    meta = []
    errors = []
    if not paths:
        errors.append("no reference files: overlap with training/holdout cannot be checked")
    for path in paths:
        if not path.is_file():
            errors.append(f"reference file is missing: {path} (run `make data` first)")
            continue
        count = 0
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            prompt = row.get("prompt")
            if not isinstance(prompt, str):
                continue
            label = f"{path.name}:{row.get('id', count)}"
            count += 1
            exact.setdefault(collapse(prompt), label)
            key = normalize(prompt)
            if key not in normalized:
                normalized[key] = label
                lsh.insert(label, signature(key))
        meta.append({"path": str(path), "sha256": sha256(path), "prompts": count})
    return exact, normalized, lsh, meta, errors


def validate_set(manifest_path: Path, references: list[Path],
                 tool_schema: Path = TOOL_SCHEMA, check_hashes: bool = True) -> dict:
    """Return a report; the set is usable only when ``report['errors']`` is empty."""
    manifest = json.loads(manifest_path.read_text())
    errors, authors = _manifest_errors(manifest)
    report = {"format": FORMAT, "manifest": {"path": str(manifest_path),
                                             "sha256": sha256(manifest_path)},
              "errors": errors, "warnings": [], "near_duplicates": {}}
    if errors:
        return report
    example = manifest["example"]
    rows_path = manifest_path.parent / manifest["rows_file"]
    if not rows_path.is_file():
        errors.append(f"rows file is missing: {rows_path}")
        return report
    rows_sha = sha256(rows_path)
    schema_sha = sha256(tool_schema)
    report.update({"set_version": manifest["set_version"], "example": example,
                   "rows": {"path": str(rows_path), "sha256": rows_sha},
                   "tool_schema_sha256": schema_sha})
    if check_hashes and manifest["rows_sha256"] != rows_sha:
        errors.append(f"manifest rows_sha256 does not match the rows file ({rows_sha}); "
                      "bump set_version for changed rows, then freeze")
    if check_hashes and manifest["tool_schema_sha256"] != schema_sha:
        errors.append(f"manifest tool_schema_sha256 does not match {tool_schema} ({schema_sha})")
    for author_id, author in authors.items():
        if author["seen_project_prompts"]:
            report["warnings"].append(f"author {author_id} has seen project prompts; "
                                      "report this when interpreting results")

    rows = []
    for line_number, line in enumerate(rows_path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            errors.append(f"line {line_number}: invalid JSON ({error.msg})")
            continue
        row_errors = _row_errors(row, line_number, authors, example)
        errors += row_errors
        if not row_errors:
            rows.append(row)
    if not rows and not errors:
        errors.append("rows file has no rows")

    ids = [row["id"] for row in rows]
    for duplicate in sorted({value for value in ids if ids.count(value) > 1}):
        errors.append(f"duplicate id {duplicate}")
    seen: dict[str, str] = {}
    for row in rows:
        key = normalize(row["prompt"])
        if key in seen:
            errors.append(f"{row['id']}: prompt duplicates {seen[key]} after normalization")
        seen.setdefault(key, row["id"])

    exact, normalized, lsh, meta, reference_errors = _reference_index(references)
    report["references"] = meta
    errors += reference_errors
    if not reference_errors:
        for row in rows:
            exact_key, normal_key = collapse(row["prompt"]), normalize(row["prompt"])
            if exact_key in exact:
                errors.append(f"{row['id']}: prompt exactly matches {exact[exact_key]}")
            elif normal_key in normalized:
                errors.append(f"{row['id']}: normalized prompt matches {normalized[normal_key]}")
            else:
                near = sorted(lsh.query(signature(normal_key)))
                if near:
                    report["near_duplicates"][row["id"]] = near[:5]
        if report["near_duplicates"]:
            report["warnings"].append(
                f"{len(report['near_duplicates'])} rows are approximate MinHash near "
                "duplicates of generated prompts; report them as a separate stratum")
    counts: dict[str, int] = {}
    for row in rows:
        counts[kind(row["targets"][0])] = counts.get(kind(row["targets"][0]), 0) + 1
    report["row_count"] = len(rows)
    report["kind_counts"] = dict(sorted(counts.items()))
    report["alternative_target_rows"] = sum(len(row["targets"]) > 1 for row in rows)
    report["rows_list"] = rows
    return report


def default_references() -> list[Path]:
    return [*REQUIRED_REFERENCES, *(path for path in OPTIONAL_REFERENCES if path.is_file())]


def load_set(manifest_path: Path, references: list[Path], allow_examples: bool = False,
             tool_schema: Path = TOOL_SCHEMA) -> dict:
    """Validate and return the set, refusing template rows unless explicitly allowed."""
    report = validate_set(manifest_path, references, tool_schema)
    if report["errors"]:
        raise ValueError("invalid evaluation set:\n" + "\n".join(report["errors"]))
    if report["example"] and not allow_examples:
        raise ValueError("this is a TEMPLATE set of placeholder rows; it cannot be scored "
                         "(predict accepts --allow-examples only as a plumbing check)")
    return report


def freeze(manifest_path: Path, references: list[Path], tool_schema: Path = TOOL_SCHEMA) -> dict:
    """Write the current rows and tool-schema hashes after every other check passes."""
    report = validate_set(manifest_path, references, tool_schema, check_hashes=False)
    if report["errors"]:
        raise ValueError("refusing to freeze an invalid set:\n" + "\n".join(report["errors"]))
    manifest = json.loads(manifest_path.read_text())
    manifest["rows_sha256"] = report["rows"]["sha256"]
    manifest["tool_schema_sha256"] = report["tool_schema_sha256"]
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return validate_set(manifest_path, references, tool_schema)


def public_report(report: dict) -> dict:
    """Drop the row text so the saved summary contains IDs and hashes only."""
    return {key: value for key, value in report.items() if key != "rows_list"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference", type=Path, action="append",
                        help="override the training/holdout prompt files to check against")
    parser.add_argument("--freeze", action="store_true",
                        help="record the rows and tool-schema hashes in the manifest")
    parser.add_argument("--output", type=Path, help="optional JSON report path")
    args = parser.parse_args()
    reference_paths = args.reference or default_references()
    try:
        result = (freeze(args.manifest, reference_paths) if args.freeze
                  else validate_set(args.manifest, reference_paths))
    except ValueError as error:
        raise SystemExit(str(error)) from error
    summary = public_report(result)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if result["errors"]:
        raise SystemExit(1)
