# Optimization work log

The objective is to maximize inference speed while keeping vAttention-style CUDA
VMM management: stable virtual K/V addresses, contiguous token strides, physical
pages mapped on demand, no application block table, and no copying old KV data.
Final repeated measurements are separated from the smaller screening runs below.
Results apply to this single-request workload and GPU.

The original implementation and vLLM reference measurements remain in
[`../results`](../results), with their original report in [`../README.md`](../README.md).
New runs use separate directories; baseline results are never overwritten.

## Repeated API comparison

Final raw results are in [`final-short/`](final-short/) and
[`final-long/`](final-long/). The short comparison covers all three models at
128 input / 64 output tokens (context 512, vLLM KV budget 256 MiB). The long
comparison covers 0.6B and 4B at 2048 / 64 tokens (context 4096, KV budget
1024 MiB). Each model/server uses ten measured requests per run, three runs,
two warmups per run, concurrency one, greedy sampling, seed 42, ignored EOS,
and disabled prefix caching. Servers run sequentially on one RTX 5060 Ti.
Both use FP16 weights, with the activation/KV precision differences below.

The reported throughput includes prefill, decoding, sampling, HTTP, and SSE.
It excludes startup/compilation and model loading. Numbers are medians of
three run metrics, not one pooled request average. TTFT and TPOT are medians
of run means. Tiny throughput differences do not establish a meaningful win.

### 128 input / 64 output tokens

| Model | Original Rust tok/s | Optimized Rust tok/s | Fresh vLLM tok/s | Rust speedup | Rust / vLLM TTFT ms |
|---|---:|---:|---:|---:|---:|
| Qwen3-0.6B | 48.09 | 259.91 | 266.55 | 5.40× | 8.30 / 12.60 |
| Qwen3-1.7B | 23.41 | 107.22 | 111.21 | 4.58× | 15.10 / 15.70 |
| Qwen3-4B-Instruct-2507 | 11.12 | 48.04 | 48.03 | 4.32× | 32.90 / 30.53 |

All 180 measured requests succeeded with exactly 128 input and 64 output
tokens. Complete continuations match vLLM on 30/30 prompts for 0.6B and 1.7B,
and 27/30 for 4B. The same 4B prompt diverges in each repetition; the different
numerical paths can change a greedy decision and its subsequent continuation.
This checks execution consistency rather than model quality.

Rust is 2.49% behind vLLM in throughput for 0.6B and 3.59% for 1.7B. The 4B
throughput difference is below 0.1%, so it is effectively a tie on this workload.
Rust reaches the first token faster on 0.6B and 1.7B, while vLLM has lower TTFT
on 4B. The former 4–5× end-to-end gap has been substantially closed; these
measurements do not support claiming an overall vLLM throughput win.

[`final-short/summary.json`](final-short/summary.json) includes TPOT, peak
observed GPU memory, continuation counts, and the measured binary hashes.

### 2048 input / 64 output tokens

| Model | Rust tok/s | vLLM tok/s | Rust / vLLM TTFT ms | Rust / vLLM TPOT ms | Identical continuations |
|---|---:|---:|---:|---:|---:|
| Qwen3-0.6B | 151.14 | 190.34 | 141.98 / 66.62 | 4.46 / 4.27 | 27/30 |
| Qwen3-4B-Instruct-2507 | 31.79 | 37.12 | 607.33 / 367.62 | 22.31 / 21.53 | 30/30 |

All 120 measured long-prompt requests succeeded with exactly 2048 input and
64 output tokens. vLLM remains faster on these longer inputs, particularly
prefill. Parallel decode substantially improves the earlier Rust long-context
result, but does not eliminate the long-prefill gap. FP32 KV and the different
attention kernels are relevant differences; this experiment does not isolate
their individual contributions. Numerical-path differences also affect the
generated continuations, as reported above.

Full metrics and checks: [`final-long/summary.json`](final-long/summary.json).

Every result directory retains exact commands, pinned model revisions, runtime
variables, package versions, binary hashes, and client/server logs. Summaries
check every request's success and exact input/output token count; the greedy
comparison also records whole-continuation text agreement, without assuming
that equal text proves equal logits or identical precision.

The measured environment used vLLM 0.33.0, PyTorch 2.14.0+cu130, and
Transformers 5.20.2. Rust used NVRTC / cuBLAS 12.9 and driver 580.178.04.
GPU peaks include initialization, the desktop, and runtime allocations, sampled
every 250 ms; differing KV budgets and dtypes prevent a tensor-memory comparison.

Reproduce after downloading the snapshots and installing the benchmark venv:

```bash
cargo build --release --locked
target/bench-venv/bin/python scripts/benchmark-servers.py \
  --result-dir benchmarks/optimization/reproduce-short
python3 scripts/summarize-benchmarks.py \
  --result-dir benchmarks/optimization/reproduce-short \
  --json-output benchmarks/optimization/reproduce-short/summary.json

target/bench-venv/bin/python scripts/benchmark-servers.py \
  --models Qwen/Qwen3-0.6B Qwen/Qwen3-4B-Instruct-2507 \
  --input-len 2048 --context-length 4096 --kv-cache-mib 1024 \
  --result-dir benchmarks/optimization/reproduce-long
```

Original baseline results remain unchanged. `short-before-cdf-fix/` preserves
an earlier short run before correcting the stochastic inverse-CDF boundary;
that correction is tested with exact within-block and cross-block boundaries.
The retained final short Rust runs were repeated using the corrected binary.
The vLLM runs were retained from the same comparison session.

## Selected implementation

- Batched cuBLAS prefill: compute all prompt tokens together and project only the
  final token to logits. Tensor Core prefill uses FP16 inputs/weights with FP32
  accumulation/output. Decode activations and VMM K/V remain FP32.
- Warp-level vectorized matrix-vector products with shuffle reductions.
- Short prefill (up to 256 KV tokens, head dimension 128, SM80+) uses fused
  online TF32 Tensor Core attention without a global score allocation. Longer
  prefill uses strided cuBLAS TF32 QK/PV with bounded query tiles; other shapes
  use an online FP32 fallback. All variants directly read the FP32 VMM K/V.
- Decode attention beyond 256 KV tokens runs 256-token compute partitions in
  parallel and merges stable softmax statistics. These partitions are not
  physical memory blocks and require no KV allocation table.
- Reusable dynamic-position decode graphs, cached by power-of-two KV length
  ranges. All mapping changes happen outside
  capture/replay; reset/release retains reservations and captured KV pointers.
- Fused Q/K normalization/RoPE and residual addition/RMSNorm.
- GPU greedy argmax, preserving lowest-ID tie breaking and nonfinite rejection.
  A pinned four-byte token transfer is captured in a reusable graph. Prompt
  staging and all dependent work use the same private stream. Pageable weight
  uploads complete before that stream starts using them.
- TCP_NODELAY for streaming HTTP; previously the first SSE write could wait for
  delayed ACK even though GPU prefill had already completed.
- Adaptive prefill scratch capacity (up to 4096 tokens, bounded by free GPU
  memory). `RUST_VATTENTION_PREFILL_BATCH` can cap it for experiments.
- GPU-only weight packing/fused QKV and gate/up projections, without duplicate
  weights or any change to the CPU checkpoint layout.
- Full-vocabulary GPU temperature/top-p sampling, using stable radix sorting
  rather than a top-k approximation. It shares the pinned four-byte result
  path; invalid logits are rejected without advancing the sampler RNG.

`RUST_VATTENTION_DISABLE_GRAPHS=1` provides an eager reference using the same
kernels. It does not change the VMM allocation scheme.
`RUST_VATTENTION_PREFILL_ATTENTION=online|blas|mma` selects diagnostic attention
paths; forcing `mma` removes the default 256-token limit and is slower on long
prompts on this GPU. Unset these variables for the reported default.

FP16 mode now rounds projection inputs to FP16 during prefill, and uses TF32
attention products with FP32 accumulation. Decode activation/accumulation and
VMM KV storage remain FP32. vLLM uses FP16 KV and different kernels, so output
identity and equal memory consumption are not implied by equal weight dtype.
Throughput differences cannot be attributed to VMM alone.

## Preliminary screening

RTX 5060 Ti 16 GB, FP16 weights, 128 input/64 output tokens, concurrency 1,
greedy, fixed seed. These screening runs use **three measured requests and one
run**, unlike the original reference's three runs of ten. Full repeated
comparisons are required before reporting small wins over vLLM.

| Stage | 0.6B API tok/s | 0.6B TTFT ms | 1.7B API tok/s | 4B API tok/s |
|---|---:|---:|---:|---:|
| Original Rust (30 requests) | 48.09 | 872.09 | 23.41 | 11.12 |
| Batched prefill | 137.39 | 41.51 | — | — |
| Vectorized decode | 196.30 | 41.04 | — | — |
| TCP_NODELAY | 196.17 | 11.71 | — | — |
| Decode graph/fused attention | 235.24 | 11.59 | 103.09 | 47.05 |
| Fused normalization/GPU greedy | 250.89 | 10.92 | 105.54 | 47.46 |
| Ordered pinned transfers | 251.23 | 10.81 | 105.71 | 47.46 |
| Packed projections | 257.39 | 10.43 | 106.90 | 47.87 |
| Original vLLM reference (30 requests) | 266.30 | 12.86 | 110.90 | 48.01 |

A wider-load/four-accumulator GEMV variant (`ilp/`) did not produce a meaningful
improvement and was removed. A cuBLAS GEMV/GEMM alternative was also rejected:
0.6B raw decode took 3.975 ms/token versus 3.775 for the selected custom kernel.
A native FlashInfer wrapper experiment did not build with the installed CUDA /
header combination and was not added as a dependency or deployed.

Long input (2048/64, context 4096) screening exposed sequential attention as a
second bottleneck. Parallel decode partitions reduced 0.6B raw decode from
about 9.9 to 4.45 ms/token; API throughput rose from 83.14 to 151.35 tok/s.
The fused MMA prefill prototype improved short-prompt raw prefill from about
9.4 to 7.33 ms, but regressed 2048-token prefill from about 137 to 335 ms.
Therefore the selected default only uses it up to 256 tokens. The regressed
long-prompt experiment remains in `mma-long/` for reproducibility.

Stochastic screening at temperature 0.8/top_p 0.9 (`stochastic/`) measured
248.95 vs 262.37 tok/s (Rust/vLLM) for 0.6B, and 47.54 vs 47.89 for 4B.
These are three-request screening runs, not a claim of a statistically
significant difference. All retained changes passed numerical and VMM
lifecycle regressions.

## Validation and profiling

GPU tests compare CPU/Transformers logits, batched/append/chunked prefill,
continuing decode, scratch growth after graph capture, physical mapping growth
while reusing a captured graph, release/remapping, stable virtual addresses,
and argmax boundary/tie/NaN behavior. Full-vocabulary sampling is tested with
partial blocks and vocabularies up to 262401 tokens. Fused attention is compared
against an independent dense FP32 reference, directly from mapped VMM storage.
The real Qwen3-0.6B also matches Transformers greedy output after 120- and
400-token prompts, covering both default attention paths and continuing decode.
See [`validation.json`](validation.json) for passed checks.

```bash
LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
 RUST_VATTENTION_TEST_MODEL=models/stories15M.bin \
 cargo test --locked -- --ignored --test-threads=1

cargo build --release --locked --example profile_cuda
LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
 target/release/examples/profile_cuda Qwen/Qwen3-0.6B --greedy
```

The stage profiler supports `--tokens`, `--steps`, and `--repeats`. API comparison
supports `--context-length` and `--kv-cache-mib` for longer prompts. Use a fresh
`--result-dir` for each variant and retain both server/client logs.

An Nsight Systems capture of the first batched-prefill implementation attributed
about 73% of measured GPU kernel time to GEMV, motivating decode optimization.
Full traces are kept under `target/` because they include initialization and are
large. To repeat the trace, run the profiler example under `nsys profile
--trace=cuda,cublas --sample=none --cpuctxsw=none`, then request
`cuda_gpu_kern_sum,cuda_api_sum` from `nsys stats`. Initialization/weight-upload
costs in a trace are not per-request inference costs.

Remaining limits include FP32 KV traffic, long-prefill attention overhead,
GEMV bandwidth, and single-request scheduling. There is no quantization,
continuous batching, speculative decoding, or multi-GPU support. These results
do not establish a general win over vLLM or production throughput capacity.
