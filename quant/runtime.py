"""Run a pinned llama.cpp server and read native completion timing fields."""

from __future__ import annotations

import json
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import psutil


def free_port() -> int:
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        return server.getsockname()[1]


class GGUFServer:
    def __init__(self, binary: Path, model: Path, context: int = 2048,
                 threads: int = 4) -> None:
        self.binary = binary
        self.model = model
        self.context = context
        self.threads = threads
        self.port = free_port()
        self.process: subprocess.Popen | None = None
        self.log = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def __enter__(self):
        if not self.binary.is_file() or not self.model.is_file():
            raise FileNotFoundError("llama-server binary or GGUF model is missing")
        self.log = tempfile.TemporaryFile(mode="w+t")
        self.process = subprocess.Popen(
            [str(self.binary), "--model", str(self.model), "--ctx-size", str(self.context),
             "--threads", str(self.threads), "--n-gpu-layers", "0", "--host", "127.0.0.1",
             "--port", str(self.port)],
            stdout=self.log, stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    self.log.seek(0)
                    raise RuntimeError("llama-server exited:\n" + self.log.read()[-3000:])
                try:
                    with urllib.request.urlopen(self.url + "/health", timeout=2) as response:
                        if response.status == 200:
                            return self
                except (urllib.error.URLError, TimeoutError):
                    pass
                time.sleep(0.5)
            raise TimeoutError("llama-server did not become healthy within 180 seconds")
        except BaseException:
            self.__exit__()
            raise

    def __exit__(self, *_):
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        if self.log:
            self.log.close()

    def complete(self, prompt: str, max_tokens: int = 160,
                 grammar: str | None = None) -> dict:
        payload = {"prompt": prompt, "n_predict": max_tokens, "temperature": 0,
                   "cache_prompt": False, "stream": True, "seed": 7}
        if grammar:
            payload["grammar"] = grammar
        request = urllib.request.Request(
            self.url + "/completion", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        started = time.perf_counter()
        first_token = None
        content = []
        timings = {}
        process = psutil.Process(self.process.pid)
        peak_rss = process.memory_info().rss
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                for line in response:
                    peak_rss = max(peak_rss, process.memory_info().rss)
                    if not line.startswith(b"data: "):
                        continue
                    event = json.loads(line[6:])
                    piece = event.get("content", "")
                    if piece:
                        if first_token is None:
                            first_token = time.perf_counter() - started
                        content.append(piece)
                    if event.get("timings"):
                        timings = event["timings"]
        except urllib.error.HTTPError as error:
            raise RuntimeError(error.read().decode()) from error
        elapsed = time.perf_counter() - started
        return {"content": "".join(content), "ttft_seconds": first_token,
                "elapsed_seconds": elapsed, "peak_server_rss_bytes_sampled": peak_rss,
                "timings": timings}
