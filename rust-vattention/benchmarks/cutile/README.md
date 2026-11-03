# cuTile attention benchmark

## Repeated OpenAI API comparison

2048 input / 64 output tokens, context 4096, concurrency one, seed 42, greedy,
EOS ignored, prefix caching disabled; ten measured requests and two warmups
per run, three runs per engine/model. Both servers use FP16 weights; vLLM uses
FP16 KV, while both Rust paths keep FP32 VMM KV. vLLM KV budget is 1024 MiB.
The table reports medians of run metrics, including startup-excluded HTTP/SSE
serving. TTFT is the median of run-mean first-token latency.

| Model | Existing Rust tok/s | cuTile Rust tok/s | vLLM tok/s | cuTile throughput gain | Existing / cuTile TTFT ms |
|---|---:|---:|---:|---:|---:|
| Qwen3-0.6B | 151.14 | 160.73 | 189.89 | 6.34% | 142.09 / 116.85 |
| Qwen3-4B-Instruct-2507 | 31.79 | 33.39 | 37.12 | 5.04% | 607.90 / 510.75 |

All 180 measured requests succeeded with exact input/output token counts.
cuTile matches vLLM on all 60 complete continuations. Against the existing Rust
path, whole-continuation agreement is Qwen3-0.6B: 27/30, Qwen3-4B-Instruct-2507: 30/30.
These checks do not constitute a model-quality evaluation. The cuTile MMA
precision is different from the existing TF32 path, despite equal KV storage.

The experiment improves prefill and total request throughput. Decode remains
unchanged at approximately 4.46 / 22.31 ms per token. vLLM remains faster:
this is an opt-in improvement over the existing Rust backend, not a vLLM win.
The ordinary backend remains the default.

Raw results: [`long/`](long/) (cuTile and vLLM),
[`legacy-long/`](legacy-long/) (existing Rust); aggregate metrics:
[`summary.json`](summary.json). Each Rust run records the same engine binary
hash; cuTile runs additionally record compiler/source/cubin provenance.

Build instructions, precision, correctness tests and tuning results are in
[`docs/cutile.md`](../../docs/cutile.md).
