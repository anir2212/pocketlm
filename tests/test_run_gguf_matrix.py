"""Check resume decisions before an expensive full GGUF matrix is launched."""

import json

from eval.run_gguf_matrix import complete_latency, complete_quality


def test_quality_resume_requires_matching_model_mode_and_full_count(tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"weights")
    grammar = tmp_path / "grammar.gbnf"
    grammar.write_text("root ::= \"x\"\n")
    result = tmp_path / "quality.json"
    report = {"model": str(model), "grammar": str(grammar),
              "metrics": {"examples": 2}, "predictions": [{"id": 1}, {"id": 2}]}
    result.write_text(json.dumps(report))
    assert complete_quality(result, model, grammar, 2)
    assert not complete_quality(result, model, None, 2)
    assert not complete_quality(result, model, grammar, 728)
    assert not complete_quality(result, model, grammar, 2,
                                [{"id": 1, "target": {}}, {"id": 2, "target": {}}])
    report["predictions"] = [{"id": 1, "gold": {}}, {"id": 2, "gold": {}}]
    result.write_text(json.dumps(report))
    assert complete_quality(result, model, grammar, 2,
                            [{"id": 1, "target": {}}, {"id": 2, "target": {}}])
    report["predictions"].pop()
    result.write_text(json.dumps(report))
    assert not complete_quality(result, model, grammar, 2)


def test_latency_resume_requires_matching_file_and_bench_rows(tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"weights")
    result = tmp_path / "latency.json"
    report = {"model": str(model), "grammar": None,
              "file_size_bytes": model.stat().st_size,
              "llama_bench": [{"n_prompt": 512, "n_gen": 0},
                              {"n_prompt": 0, "n_gen": 128}],
              "request_repetitions": 20, "request_samples": [{} for _ in range(20)],
              "time_to_first_token_seconds": 0.5}
    result.write_text(json.dumps(report))
    assert complete_latency(result, model, None)
    report["request_samples"].pop()
    result.write_text(json.dumps(report))
    assert not complete_latency(result, model, None)
    report["request_samples"].append({})
    result.write_text(json.dumps(report))
    model.write_bytes(b"new weights")
    assert not complete_latency(result, model, None)
