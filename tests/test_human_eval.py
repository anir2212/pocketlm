"""Protect the human-written evaluation set contract, metric math and paired test."""

import json
from pathlib import Path

import pytest

from eval.compare_runs import exact_mcnemar, paired_summary
from eval.human_eval import compare, score, score_item
from eval.human_set import freeze, load_set, sha256, validate_set

REFUSAL = {"refusal": "I can only control the local demo device."}
DPI = {"calls": [{"tool": "set_dpi", "arguments": {"dpi": 800}}]}
TWO_APPS = {"calls": [{"tool": "launch_app", "arguments": {"name": "notes"}},
                      {"tool": "launch_app", "arguments": {"name": "browser"}}]}
SWAPPED = {"calls": TWO_APPS["calls"][::-1]}
ASK = {"clarification": "What volume value would you like?"}


def _row(identifier, prompt, targets, **changes):
    row = {"id": identifier, "prompt": prompt, "targets": targets, "author_id": "a01",
           "written_at": "2026-10-21", "written_by": "human", "llm_assisted": False,
           "non_sensitive": True, "example": False}
    row.update(changes)
    return row


ROWS = [
    _row("h-001", "mouse sensitivity to eight hundred dpi", [DPI]),
    _row("h-002", "pull up my notes then the web browser", [TWO_APPS, SWAPPED]),
    _row("h-003", "can you book me a haircut on friday", [REFUSAL]),
    _row("h-004", "the sound is wrong, fix the level", [ASK]),
]


def _write_set(tmp_path: Path, rows=ROWS, example=False, frozen=True, **manifest_changes):
    rows_path = tmp_path / "set.jsonl"
    rows_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = {
        "format": "pocketlm-human-eval/1", "set_version": "human_test", "rows_file": "set.jsonl",
        "rows_sha256": "", "tool_schema_sha256": "", "example": example, "created": "2026-10-21",
        "authors": [{"author_id": "a01", "permission": "Agreed to storage and evaluation.",
                     "consent_date": "2026-10-21", "seen_project_prompts": False}],
        "collection_protocol": "Typed from scratch without viewing project prompts.",
    }
    manifest.update(manifest_changes)
    path = tmp_path / "set.manifest.json"
    path.write_text(json.dumps(manifest))
    references = _references(tmp_path)
    if frozen:
        freeze(path, references)
    return path, references


def _references(tmp_path: Path) -> list[Path]:
    train = tmp_path / "train.jsonl"
    test = tmp_path / "test.jsonl"
    train.write_text(json.dumps({"id": "t1", "prompt": "please set my dpi to 800"}) + "\n")
    test.write_text(json.dumps({"id": "d1", "prompt": "please quit the calculator app"}) + "\n")
    return [train, test]


def _errors(tmp_path, rows, **manifest_changes):
    path, references = _write_set(tmp_path, rows, frozen=False, **manifest_changes)
    return validate_set(path, references, check_hashes=False)["errors"]


def test_valid_set_freezes_and_loads(tmp_path):
    path, references = _write_set(tmp_path)
    manifest = json.loads(path.read_text())
    assert manifest["rows_sha256"] == sha256(tmp_path / "set.jsonl")
    loaded = load_set(path, references)
    assert loaded["row_count"] == 4
    assert loaded["kind_counts"] == {"calls": 2, "clarification": 1, "refusal": 1}
    assert loaded["alternative_target_rows"] == 1
    assert [item["prompts"] for item in loaded["references"]] == [1, 1]


def test_rejects_duplicate_ids_and_normalized_duplicates(tmp_path):
    rows = [ROWS[0], _row("h-001", "turn the speakers down to ten percent", [ASK]),
            _row("h-009", "Mouse sensitivity, to EIGHT hundred DPI!", [DPI])]
    errors = _errors(tmp_path, rows)
    assert "duplicate id h-001" in errors
    assert any("duplicates h-001 after normalization" in error for error in errors)


def test_rejects_invalid_or_noncanonical_targets(tmp_path):
    rows = [
        _row("bad-1", "mouse dpi to nine nine nine", [{"calls": [
            {"tool": "set_dpi", "arguments": {"dpi": 999}}]}]),
        _row("bad-2", "do a cartwheel for me please", [{"calls": [
            {"tool": "invented", "arguments": {}}]}]),
        _row("bad-3", "order a pizza to my door", [{"refusal": "No."}]),
        _row("bad-4", "set things to something", [DPI, ASK]),
        _row("bad-5", "launch my two apps now", []),
    ]
    errors = "\n".join(_errors(tmp_path, rows))
    assert "bad-1): target 0 fails tools_api/schema.json" in errors
    assert "bad-2): target 0 fails" in errors
    assert "bad-3): target 0 refusal must be exactly" in errors
    assert "bad-4): alternative targets must share" in errors
    assert "bad-5): targets must be a nonempty list" in errors


def test_rejects_missing_or_false_provenance(tmp_path):
    no_author = {key: value for key, value in ROWS[0].items() if key != "author_id"}
    rows = [
        no_author,
        _row("p-2", "mute the microphone for now", [DPI], author_id="ghost"),
        _row("p-3", "mute the microphone right away", [DPI], written_by="model"),
        _row("p-4", "mute the microphone quickly", [DPI], llm_assisted=True),
        _row("p-5", "mute the microphone at once", [DPI], non_sensitive=False),
        _row("p-6", "email me at someone@example.com", [REFUSAL]),
    ]
    errors = "\n".join(_errors(tmp_path, rows))
    assert "missing field author_id" in errors
    assert "p-2): author_id is not listed" in errors
    assert "p-3): scored rows must be written_by='human'" in errors
    assert "p-4): scored rows must be written_by='human'" in errors
    assert "p-5): non_sensitive must be true" in errors
    assert "p-6): prompt contains an email- or phone-like string" in errors
    assert "manifest: authors must be a nonempty list" in _errors(tmp_path, ROWS[:1], authors=[])
    manifest_errors = _errors(tmp_path, ROWS[:1], authors=[{"author_id": "a01"}])
    assert any("author 0 needs" in error for error in manifest_errors)
    bad_author = [{"author_id": ["a01"], "permission": "yes", "consent_date": "2026-10-21",
                   "seen_project_prompts": False},
                  {"author_id": "a02", "permission": " ", "consent_date": "soon",
                   "seen_project_prompts": "no"}]
    manifest_errors = "\n".join(_errors(tmp_path, ROWS[:1], authors=bad_author))
    assert "author 0 author_id must be a short pseudonym" in manifest_errors
    assert "author 1 consent_date must be YYYY-MM-DD" in manifest_errors
    assert "author 1 permission statement is empty" in manifest_errors
    assert "author 1 seen_project_prompts must be boolean" in manifest_errors


def test_rejects_training_and_holdout_overlap(tmp_path):
    rows = [_row("o-1", "please set my dpi to 800", [DPI]),
            _row("o-2", "Please, quit the Calculator app!", [{"calls": [
                {"tool": "close_app", "arguments": {"name": "calculator"}}]}])]
    errors = "\n".join(_errors(tmp_path, rows))
    assert "o-1: prompt exactly matches train.jsonl:t1" in errors
    assert "o-2: normalized prompt matches test.jsonl:d1" in errors


def test_rejects_missing_references_and_changed_files(tmp_path):
    path, references = _write_set(tmp_path)
    assert any("reference file is missing" in error for error in validate_set(
        path, [tmp_path / "absent.jsonl"])["errors"])
    assert any("no reference files" in error for error in validate_set(path, [])["errors"])
    rows_path = tmp_path / "set.jsonl"
    rows_path.write_text(rows_path.read_text().replace("friday", "monday"))
    assert any("rows_sha256 does not match" in error
               for error in validate_set(path, references)["errors"])
    with pytest.raises(ValueError, match="invalid evaluation set"):
        load_set(path, references)


def test_template_rows_validate_but_are_never_scored(tmp_path):
    rows = [_row("EXAMPLE-1", "[EXAMPLE] set mouse dpi to 800 now", [DPI],
                 example=True, written_by="template_placeholder", llm_assisted=True)]
    path, references = _write_set(tmp_path, rows, example=True)
    assert load_set(path, references, allow_examples=True)["example"] is True
    with pytest.raises(ValueError, match="TEMPLATE"):
        load_set(path, references)
    predictions = tmp_path / "pred.json"
    predictions.write_text(json.dumps({"predictions": []}))
    with pytest.raises(ValueError, match="TEMPLATE"):
        score(path, predictions, tmp_path / "score.json", references)
    mixed = [ROWS[0], rows[0]]
    assert any("example must be false" in error for error in _errors(tmp_path, mixed))
    human_example = [{**rows[0], "written_by": "human"}]
    assert any("template rows must say" in error
               for error in _errors(tmp_path, human_example, example=True))


def test_repository_template_is_valid_and_frozen():
    manifest = Path("data/human_eval/TEMPLATE_v0.manifest.json")
    rows_path = manifest.with_name("TEMPLATE_v0.jsonl")
    data = json.loads(manifest.read_text())
    assert data["example"] is True
    assert data["rows_sha256"] == sha256(rows_path)
    assert data["tool_schema_sha256"] == sha256(Path("tools_api/schema.json"))
    rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line]
    assert rows and all(row["example"] and row["prompt"].startswith("[EXAMPLE]") for row in rows)
    assert all(row["written_by"] == "template_placeholder" for row in rows)


def _predictions(tmp_path, path, references, outputs, name):
    loaded = load_set(path, references)
    report = {"set": {"set_version": loaded["set_version"],
                      "rows_sha256": loaded["rows"]["sha256"],
                      "tool_schema_sha256": loaded["tool_schema_sha256"]},
              "contains_example_rows": False, "backend": {"kind": name},
              "predictions": [{"id": row["id"], "targets": row["targets"], "output": output}
                              for row, output in zip(loaded["rows_list"], outputs, strict=True)]}
    target = tmp_path / f"{name}.json"
    target.write_text(json.dumps(report))
    return target


def test_score_counts_denominators_alternatives_and_failures(tmp_path):
    path, references = _write_set(tmp_path)
    outputs = [
        '{"calls":[{"tool":"set_dpi","arguments":{"dpi":1600}}]}',  # wrong argument
        json.dumps(SWAPPED),  # matches the listed alternative order
        json.dumps(REFUSAL),
        "not json",
    ]
    report = score(path, _predictions(tmp_path, path, references, outputs, "a"),
                   tmp_path / "score.json", references, repetitions=20)
    metrics = report["metrics"]
    assert metrics["overall_target_exact_match"] == {"correct": 2, "denominator": 4, "rate": 0.5}
    assert metrics["full_call_exact_match"]["denominator"] == 2
    assert metrics["full_call_exact_match"]["correct"] == 1
    assert metrics["tool_name_exact_match_on_calls"]["correct"] == 2
    assert metrics["json_validity"]["correct"] == 3
    assert metrics["correct_refusal"] == {"correct": 1, "denominator": 1, "rate": 1.0}
    assert metrics["correct_clarification"] == {"correct": 0, "denominator": 1, "rate": 0.0}
    assert metrics["exact_via_alternative_target"]["correct"] == 1
    assert metrics["argument_micro"] == {"true_positive": 2, "predicted": 3, "gold": 3,
                                         "precision": 2 / 3, "recall": 2 / 3, "f1": 2 / 3}
    assert report["failures"] == {"wrong_arguments_or_payload": 1, "exact": 2,
                                  "invalid_json_object": 1}
    assert report["strata"]["multi_call"]["exact"] == 1
    assert [item["exact"] for item in report["items"]] == [False, True, True, False]


def test_score_item_uses_first_target_kind_for_denominators():
    item = score_item([TWO_APPS, SWAPPED], json.dumps(ASK))
    assert item["has_calls"] and not item["full_exact"] and not item["kind_correct"]
    assert item["failure"] == "wrong_response_kind"


def test_score_rejects_predictions_from_another_set(tmp_path):
    path, references = _write_set(tmp_path)
    good = _predictions(tmp_path, path, references, ["x"] * 4, "a")
    report = json.loads(good.read_text())
    report["set"]["rows_sha256"] = "0" * 64
    stale = tmp_path / "stale.json"
    stale.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="rows_sha256 differs"):
        score(path, stale, tmp_path / "score.json", references)
    report = json.loads(good.read_text())
    report["predictions"] = report["predictions"][::-1]
    reordered = tmp_path / "reordered.json"
    reordered.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="IDs or targets differ"):
        score(path, reordered, tmp_path / "score.json", references)
    report = json.loads(good.read_text())
    report["contains_example_rows"] = True
    flagged = tmp_path / "flagged.json"
    flagged.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="never scored"):
        score(path, flagged, tmp_path / "score.json", references)


def test_exact_mcnemar_matches_binomial_tail():
    assert exact_mcnemar(0, 0) == 1.0
    assert exact_mcnemar(0, 5) == pytest.approx(2 / 32)
    assert exact_mcnemar(1, 4) == pytest.approx(2 * 6 / 32)
    assert exact_mcnemar(4, 1) == exact_mcnemar(1, 4)
    assert exact_mcnemar(3, 3) == 1.0
    pmf, tail = 0.5 ** 28, 0.0
    for k in range(11):  # independent float recurrence for P(X <= 10), X ~ Binomial(28, 0.5)
        tail += pmf
        pmf *= (28 - k) / (k + 1)
    assert exact_mcnemar(18, 10) == pytest.approx(2 * tail, rel=1e-12)
    assert exact_mcnemar(18, 10) == pytest.approx(0.184933, abs=1e-6)
    with pytest.raises(ValueError):
        exact_mcnemar(-1, 2)


def test_paired_summary_counts_all_four_cells():
    values = [(1, 1), (1, 0), (0, 1), (0, 1), (0, 0)]
    summary = paired_summary(values, repetitions=50)
    assert {key: summary[key] for key in (
        "both_correct", "left_only", "right_only", "both_wrong")} == {
        "both_correct": 1, "left_only": 1, "right_only": 2, "both_wrong": 1}
    assert summary["right_minus_left"] == pytest.approx(0.2)
    assert summary["exact_mcnemar_p_two_sided"] == 1.0


def test_compare_pairs_two_runs_on_the_same_set(tmp_path):
    path, references = _write_set(tmp_path)
    left = _predictions(tmp_path, path, references,
                        [json.dumps(DPI), "bad", json.dumps(REFUSAL), "bad"], "left")
    right = _predictions(tmp_path, path, references,
                         [json.dumps(DPI), json.dumps(TWO_APPS), "bad", json.dumps(ASK)],
                         "right")
    report = compare(path, left, right, tmp_path / "pair.json", references, repetitions=20)
    exact = report["comparisons"]["full_exact"]
    assert (exact["both_correct"], exact["left_only"], exact["right_only"],
            exact["both_wrong"]) == (1, 1, 2, 0)
    assert exact["exact_mcnemar_p_two_sided"] == 1.0
    assert report["left"]["backend"] == {"kind": "left"}
