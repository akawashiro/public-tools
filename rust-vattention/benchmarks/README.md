# OpenAI API benchmark

This directory contains measurements made with the same `vllm bench serve
--backend openai --endpoint /v1/completions` client against rust-vattention and
vLLM. The benchmark measures HTTP serving, including prefill, sampling, decoding,
and streaming. Model loading, downloads, and server compilation are excluded
from request latency.

Hardware: NVIDIA GeForce RTX 5060 Ti 16 GB. Both servers run sequentially on the
same GPU. Weights are FP16 in both servers; Rust uses FP32 activations and KV,
while vLLM uses FP16 activations and KV. Rust has ordinary CUDA matrix-vector
kernels and processes prefill one token at a time. vLLM retains its default
optimized execution and CUDA graphs, with prefix caching disabled. Consequently
this is an end-to-end engine comparison; it does not isolate VMM versus paged KV
cache management.

Workload: random prompts, exactly 128 input tokens and 64 output tokens, seed 42,
temperature 0, EOS ignored, concurrency 1, 10 measured requests per run, two warmup
requests, three runs per model/server. Both use a context capacity of 512.
vLLM's KV cache budget is explicitly 256 MiB. Rust maps KV memory as needed and
retains mappings between requests. Different KV precision, allocation policies,
CUDA graphs, and runtime overhead affect the memory comparison.

`results/environment.json` records package versions and GPU details.
`*-run.json` records exact commands, model revisions, and observed GPU memory.
GPU memory is sampled using `nvidia-smi` every 250 ms, includes the desktop and
runtime, and records the peak across server initialization and requests. It is
not a tensor-only memory measurement. `*-0.json` etc. contain original vLLM client
results, including every request's latency, output, and errors. Server and client
logs are retained next to them.

The report uses the median of three run means for TTFT (first streamed token),
TPOT (time per subsequent output token), and E2E latency. Output tok/s counts
all output tokens over the full measured serving interval, including prefill;
decode tok/s is `1000 / TPOT_ms`. Small sample sizes describe this workload and
hardware rather than general production capacity.

The vLLM JIT uses the installed GCC 13 through `CC`, `CXX`, and
`NVCC_PREPEND_FLAGS`, because the host default GCC 15 is newer than CUDA 12.9
supports. Initial compiler setup and model downloads are outside all latency
measurements.

Reproduce after installing vLLM in `target/bench-venv` and downloading the selected
models into `models/hub`:

```bash
cargo build --release --locked
HF_HUB_DISABLE_XET=1 target/bench-venv/bin/python - <<'PYTHON'
from huggingface_hub import snapshot_download
for model in ["Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B-Instruct-2507"]:
    snapshot_download(model, cache_dir="models/hub", allow_patterns=["*.json", "*.safetensors"])
PYTHON
target/bench-venv/bin/python scripts/benchmark-servers.py
python3 scripts/summarize-benchmarks.py
```

The driver runs only on cached snapshots; model IDs and pinned revisions are in
the run metadata. Completed runs are skipped, so use a new `--result-dir` to
repeat a complete experiment.

Measured on 2026-11-02–03 (JST), using vLLM 0.33.0, PyTorch 2.14.0+cu130,
Transformers 5.20.2, Rust 1.98.1, and NVRTC 12.9.86. All 180 measured requests
succeeded, with exactly 128 input and 64 output tokens each.

| Model | Rust API tok/s | vLLM API tok/s | vLLM speedup | Rust TTFT ms | vLLM TTFT ms |
|---|---:|---:|---:|---:|---:|
| [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | 48.09 | 266.30 | 5.54× | 872.09 | 12.86 |
| [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) | 23.41 | 110.90 | 4.74× | 1812.39 | 15.96 |
| [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) | 11.12 | 48.01 | 4.32× | 3823.94 | 30.98 |

vLLM is 4.32–5.54× faster in end-to-end output throughput on this workload.
The largest latency difference is prefill: Rust performs sequential matrix-vector
inference for every input token, whereas vLLM processes the prompt with optimized
batched kernels. Decode alone differs by approximately 1.49–2.02×. These results
compare the complete implementations and do not establish a performance difference
between VMM and paged KV cache allocation.

| Model / server | Runs | Output tok/s | TTFT ms | TPOT ms | Decode tok/s | E2E ms | Peak GPU MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-0.6B-rust | 3 | 48.09 | 872.09 | 7.28 | 137.45 | 1330.60 | 1416 |
| Qwen3-0.6B-vllm | 3 | 266.30 | 12.86 | 3.61 | 277.32 | 240.05 | 1844 |
| Qwen3-1.7B-rust | 3 | 23.41 | 1812.39 | 14.63 | 68.36 | 2734.03 | 3560 |
| Qwen3-1.7B-vllm | 3 | 110.90 | 15.96 | 8.90 | 112.34 | 576.78 | 4296 |
| Qwen3-4B-Instruct-2507-rust | 3 | 11.12 | 3823.94 | 30.69 | 32.58 | 5757.65 | 7984 |
| Qwen3-4B-Instruct-2507-vllm | 3 | 48.01 | 30.98 | 20.66 | 48.40 | 1332.71 | 8986 |

Pinned model revisions and exact Rust binary SHA-256 are also recorded in
[summary.json](summary.json). The first two models produced identical text for
all 10 prompts in each of three runs (30/30 paired continuations). The 4B model
matched 9/10 prompts in each run (27/30); one prompt diverged consistently.
The engines use different activation precision and floating-point kernels, so
these are not bit-exact computations. Matching greedy text on this random
workload is an additional check, not a quality evaluation or a claim of universal
numerical equivalence.

[validation.json](validation.json) records the CPU/GPU tests and real Qwen3 OpenAI
SDK/Transformers checks. Tiny independent Transformers reference fixtures cover
both Llama and Qwen3, GQA, nondefault normalization/RoPE parameters, FP16/BF16,
shards, tied output weights, and rejection of corrupt checkpoints.
