# cuTile Rust attention experiment

This optional backend replaces long-prompt attention with a fused online-softmax
kernel authored in Rust using NVIDIA cutile-rs. All model loading, projections,
CUDA Graph decode, VMM reservations, physical mapping and request handling remain
in the existing engine. The standard build has no cutile dependency or CUDA 13
build requirement. The compiler tool is a separate Cargo workspace under
`kernels/cutile`, pinned to NVIDIA/cuda-rust revision
`cced2a35bcc557b30f27209ff3887b4eb728cb63` (cutile 0.5.0).

## Memory and precision

Q and FP32 K/V are viewed with explicit strides over the existing token-major
allocations. The kernel reads VMM `base + token * stride` directly. It introduces
no KV copy, second cache, block table, new context, or new stream. The output
uses existing FP16 projection-input scratch. Launches use the session stream
and follow QKV production without extra per-layer synchronization.

The compiler supports `--mma-type tf32` and `--mma-type f16`. Both retain FP32
cache storage and FP32 accumulation, with FP16 output. The selected experiment
uses FP16 Q/K/P/V operands inside the kernel; the reference attention uses TF32
products. This precision change can affect logits and greedy continuations.
It must not be described as an identical-precision performance comparison.

The backend only handles prompt chunks of at least 16 tokens with a resulting
KV length greater than 256. Short prompts retain the existing fused MMA path;
other small chunks retain the existing fallback. Decode uses the existing
implementation. The optional backend currently requires FP16 weight mode and a
compiled artifact matching the GPU and the exact head layout.

## Setup

Tested on Linux x86_64, RTX 5060 Ti (sm_120), driver 580.178.04, Rust 1.98.1.
The compiler needs CUDA 13.2 headers, `tileiras`, and libNVVM. The normal engine
continues using NVRTC/cuBLAS 12.9. The setup below installs only under `target/`;
no driver or system toolkit installation is changed. Python 3.12+ is required.
Archives are checked against the SHA256 values in NVIDIA's release manifest.

```bash
python3 scripts/setup-cutile-toolkit.py
export CUDA_TOOLKIT_PATH="$PWD/target/cuda-toolkit-13.2"
cargo build --release --manifest-path kernels/cutile/Cargo.toml --locked

# Qwen3-0.6B and Qwen3-1.7B share this head layout.
kernels/cutile/target/release/vattention-cutile-compiler \
  --heads 16 --kv-heads 8 --head-dim 128 \
  --query-tile 128 --key-tile 64 --mma-type f16 --output target/cutile-selected

# Qwen3-4B-Instruct-2507.
kernels/cutile/target/release/vattention-cutile-compiler \
  --heads 32 --kv-heads 8 --head-dim 128 \
  --query-tile 128 --key-tile 64 --mma-type f16 --output target/cutile-selected

cargo build --release --locked
LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
 RUST_VATTENTION_PREFILL_ATTENTION=cutile \
 RUST_VATTENTION_CUTILE_DIR=target/cutile-selected \
 target/release/rust-vattention --serve \
 --model Qwen/Qwen3-0.6B --weight-dtype fp16 --context-length 4096 --port 8000
```

The compiler writes `.cubin`, readable `.tileir`, and a `.json` manifest with
head layout, tiles, precision, source and cubin hashes, compiler revision and
compiler-generated launch checks. The host enforces the two launch-grid checks:
heads match the first Q dimension, and the second grid is exactly `ceil(n/BM)`.
It constructs positive dimensions, the declared strides, matching K/V lengths,
and disjoint output partitions, and maps the complete input prefix first.

Unlike a PTX JIT using a newer PTX version, the tested path loads a finished
sm_120 cubin with the existing driver. Support on other GPUs/drivers requires
its own compilation and execution validation. Artifacts are not checked in.

Unset `RUST_VATTENTION_PREFILL_ATTENTION` to select the existing default backend.
Missing artifacts or manifest/model layout mismatches produce an initialization error for an explicit
cuTile request. This prevents accidentally benchmarking a silent fallback.

## Correctness

Compile the independent test fixture layout first:

```bash
CUDA_TOOLKIT_PATH="$PWD/target/cuda-toolkit-13.2" \
 kernels/cutile/target/release/vattention-cutile-compiler \
 --heads 2 --kv-heads 1 --head-dim 128 \
 --query-tile 128 --key-tile 64 --mma-type f16 --output target/cutile-selected

LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
 RUST_VATTENTION_CUTILE_DIR=target/cutile-selected \
 cargo test --locked cuda::cutile::tests -- --ignored --test-threads=1 --nocapture
```

The direct operator test compares FP16 outputs to independent dense FP64 math,
covering GQA, odd lengths, nonzero positions, 5/17-token windows, all-masked
initial tiles, VMM physical-page growth and release/remapping with unchanged
virtual addresses. Maximum observed absolute error was below 0.00058.

The full-model test compares against independent Transformers Mistral logits,
including prefill sizes 17/128/256/257/273, appended and chunked prompts, and
continuing decode. Its absolute-logit tolerance is 0.0005. Standard CPU tests
and existing GPU regressions also pass (8 CPU / 13 GPU tests in this run). Real Qwen3-0.6B matches Transformers
greedy output for the checked completion/chat prompts and 120/400-token inputs;
OpenAI SDK JSON/SSE checks pass with the cuTile backend selected.

Running every ignored GPU test requires both the legacy test checkpoint and
these optional cuTile artifacts. To run the original GPU tests without building
cuTile, append `--skip cuda::cutile::tests` to the test harness arguments.

## Performance

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

The initial TF32 32x64 kernel was slower than the existing attention. Larger
query tiles and FP16 MMA improved performance. These are hardware-specific
screening results, not a general claim that Rust or cuTile outperforms CUDA C++:

| MMA operands | Query x key tile | 0.6B prefill ms (2048 tokens) |
|---|---:|---:|
| Existing default | — | 137.9 |
| TF32 | 32x64 | 194.6 |
| TF32 | 64x64 | 163.1 |
| TF32 | 128x64 | 142.9 |
| FP16 | 64x64 | 132.0 |
| FP16 | 128x64 | 112.5 |
| FP16 | 64x128 | 129.8 |
| FP16 | 64x256 | 138.8 |
| FP16 | 128x128 | 119.5 |
| FP16 | 128x256 | 168.4 |

Screening used fixed valid token IDs, two or three repeats, a warmup, and no HTTP.
The later API comparison uses `vllm bench serve` with identical request settings
for the existing Rust backend, cuTile Rust backend and vLLM. Logs, raw request
metrics, hashes and model revisions are retained separately under
`benchmarks/cutile/`. No original benchmark data was overwritten.

Validation summary: [`benchmarks/cutile/validation.json`](../benchmarks/cutile/validation.json).
A formatting/comment-only rebuild changed cubin metadata hashes; the complete
`cuobjdump --dump-sass` output, including instruction encodings and control
words, remained identical. The benchmark records the actual measured artifact hashes.
