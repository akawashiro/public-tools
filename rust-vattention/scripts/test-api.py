#!/usr/bin/env python3
"""Exercise OpenAI-compatible endpoints over HTTP using a tiny local CPU fixture."""
import json
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

root = Path(__file__).resolve().parents[1]
with socket.socket() as s:
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
model = str(root / "tests/fixtures/qwen3-bf16-sharded")
base = f"http://127.0.0.1:{port}"
p = subprocess.Popen([str(root / "target/release/rust-vattention"), "--serve", "--backend", "cpu", "--model", model, "--port", str(port)], stdout=subprocess.DEVNULL)
def request(path, body=None):
    req = urllib.request.Request(base+path, data=json.dumps(body).encode() if body is not None else None, headers={"Content-Type":"application/json"})
    return urllib.request.urlopen(req, timeout=30)
try:
    for _ in range(100):
        try:
            with request("/health") as r: assert r.read() == b"ok"
            break
        except (OSError, urllib.error.URLError): time.sleep(.1)
    else: raise AssertionError("server not ready")
    with request("/v1/models") as r: assert json.load(r)["data"][0]["id"] == model
    body = {"model":model,"prompt":"t9 t11","max_tokens":4,"temperature":0,"ignore_eos":True}
    with request("/v1/completions", body) as r: plain = json.load(r)
    assert plain["usage"] == {"prompt_tokens":3,"completion_tokens":4,"total_tokens":7}, plain
    streamed = dict(body, stream=True, stream_options={"include_usage":True})
    with request("/v1/completions", streamed) as r: lines = [l.decode().strip()[6:] for l in r if l.startswith(b"data: ")]
    assert lines[-1] == "[DONE]", lines
    events = [json.loads(l) for l in lines[:-1]]
    assert "".join(e["choices"][0]["text"] for e in events if e["choices"]) == plain["choices"][0]["text"]
    assert events[-1]["usage"] == plain["usage"] and events[-1]["choices"] == []
    chat = {"model":model,"messages":[{"role":"user","content":"t9 t11"}],"max_tokens":4,"temperature":0,"ignore_eos":True}
    with request("/v1/chat/completions", chat) as r: response=json.load(r)
    assert response["object"] == "chat.completion" and response["choices"][0]["message"]["role"] == "assistant"
    with request("/v1/chat/completions", dict(chat, stream=True)) as r:
        events = [json.loads(l.decode().strip()[6:]) for l in r if l.startswith(b"data: ") and b"[DONE]" not in l]
    assert "".join(e["choices"][0]["delta"].get("content", "") for e in events) == response["choices"][0]["message"]["content"]
    for change in [{"model":"bad"},{"n":2},{"prompt":[16]},{"max_tokens":33},{"logprobs":1},{"tools":[]}]:
        try: request("/v1/completions", dict(body, **change)); raise AssertionError(change)
        except urllib.error.HTTPError as e:
            assert e.code == 400 and "error" in json.load(e)
    # A stop string spanning decoder chunks must not leak through SSE.
    output = plain["choices"][0]["text"]
    if output:
        stop = output[:3]
        with request("/v1/completions", dict(body, stop=stop, stream=True)) as r:
            events = [json.loads(l.decode().strip()[6:]) for l in r if l.startswith(b"data: ") and b"[DONE]" not in l]
        assert "".join(e["choices"][0]["text"] for e in events) == ""
        assert events[-1]["choices"][0]["finish_reason"] == "stop"
    print("API checks passed: models, completions, chat, SSE, usage, stop, validation")
finally:
    p.terminate()
    p.wait(timeout=10)
