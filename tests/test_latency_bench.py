"""Check that request distributions, not one lucky request, enter the report."""

import json
from types import SimpleNamespace

from eval import latency_bench


def test_http_latency_aggregation_keeps_distribution(monkeypatch, tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model")
    output = tmp_path / "latency.json"
    bench_rows = [{"n_prompt": 512, "n_gen": 0, "avg_ts": 100},
                  {"n_prompt": 0, "n_gen": 128, "avg_ts": 50}]
    monkeypatch.setattr(latency_bench, "subprocess", SimpleNamespace(
        run=lambda *args, **kwargs: SimpleNamespace(stdout=json.dumps(bench_rows))))
    monkeypatch.setattr(latency_bench, "cached_snapshot", lambda *args: str(tmp_path))
    monkeypatch.setattr(latency_bench.AutoTokenizer, "from_pretrained",
                        lambda *args: object())
    monkeypatch.setattr(latency_bench, "render", lambda *args, **kwargs: "prompt")

    class FakeServer:
        count = 0

        def __init__(self, *args):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def complete(self, *args, **kwargs):
            self.count += 1
            index = self.count - 1
            return {"content": "answer", "ttft_seconds": float(index),
                    "elapsed_seconds": float(index + 1),
                    "peak_server_rss_bytes_sampled": 100 + index,
                    "timings": {"predicted_per_second": float(10 + index),
                                "predicted_n": 5}}

    monkeypatch.setattr(latency_bench, "GGUFServer", FakeServer)
    report = latency_bench.benchmark(model, tmp_path / "bench", tmp_path / "server",
                                     output, request_repetitions=20)
    assert report["request_repetitions"] == 20
    assert len(report["request_samples"]) == 20
    assert report["time_to_first_token_seconds"] == 10.5
    assert report["time_to_first_token_p95_seconds"] == 19.0
    assert report["end_to_end_seconds"] == 11.5
    assert report["request_generation_tokens_per_second_median"] == 20.5
    assert report["peak_server_rss_bytes_sampled"] == 120
    assert json.loads(output.read_text())["request_repetitions"] == 20
