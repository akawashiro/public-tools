#!/usr/bin/env python3
"""Summarize vLLM bench serve results; preserve raw measurements alongside this file."""
import argparse
import json
import statistics
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--result-dir", type=Path, default=Path("benchmarks/results"))
p.add_argument("--json-output", type=Path, help="Write audited metrics and greedy output comparisons")
a = p.parse_args()
environment = json.loads((a.result_dir / "environment.json").read_text())
summary = {"environment": environment, "models": {}, "measured_requests": 0}
groups = {}
for path in sorted(a.result_dir.glob("*.json")):
    d = json.loads(path.read_text())
    if "output_throughput" not in d: continue
    prefix, run = path.stem.rsplit("-", 1)
    if not run.isdigit(): continue
    groups.setdefault(prefix, []).append(d)
print("| Model / server | Runs | Output tok/s | TTFT ms | TPOT ms | Decode tok/s | E2E ms | Peak GPU MiB |")
print("|---|---:|---:|---:|---:|---:|---:|---:|")
for name, runs in sorted(groups.items()):
    for d in runs:
        assert d["failed"] == 0 and d["completed"] == d["num_prompts"], name
    def median(key): return statistics.median(d[key] for d in runs)
    runpath = a.result_dir / f"{name}-run.json"
    meta = json.loads(runpath.read_text()) if runpath.exists() else {}
    if a.json_output:
        assert len(runs) == environment["repeats"], (name, len(runs))
        for d in runs:
            n = environment["num_prompts"]
            assert d["completed"] == n and not any(d["errors"]), name
            assert d["input_lens"] == [environment["input_len"]] * n, name
            assert d["output_lens"] == [environment["output_len"]] * n, name
        model, engine = name.rsplit("-", 1)
        record = {key: median(key) for key in ("output_throughput", "mean_ttft_ms", "mean_tpot_ms", "mean_e2el_ms")}
        record.update(revision=meta.get("revision"), peak_gpu_mib=meta.get("peak_gpu_mib"),
                      binary_sha256=meta.get("rust_binary_sha256") if engine == "rust" else None)
        summary["models"].setdefault(model, {})[engine] = record
        summary["measured_requests"] += sum(d["completed"] for d in runs)
    tpot = median("mean_tpot_ms")
    print(f'| {name} | {len(runs)} | {median("output_throughput"):.2f} | {median("mean_ttft_ms"):.2f} | {tpot:.2f} | {1000/tpot:.2f} | {median("mean_e2el_ms"):.2f} | {meta.get("peak_gpu_mib", "pending")} |')

if a.json_output:
    for model, engines in summary["models"].items():
        if not {"rust", "vllm"}.issubset(engines):
            continue
        assert engines["rust"]["revision"] == engines["vllm"]["revision"], model
        engines["rust_over_vllm_throughput"] = engines["rust"]["output_throughput"] / engines["vllm"]["output_throughput"]
        if environment["temperature"] == 0:
            same, compared = 0, 0
            for i in range(environment["repeats"]):
                left = json.loads((a.result_dir / f"{model}-rust-{i}.json").read_text())
                right = json.loads((a.result_dir / f"{model}-vllm-{i}.json").read_text())
                assert len(left["generated_texts"]) == len(right["generated_texts"])
                same += sum(x == y for x, y in zip(left["generated_texts"], right["generated_texts"]))
                compared += len(left["generated_texts"])
            engines.update(identical_continuations=same, compared_continuations=compared)
    a.json_output.write_text(json.dumps(summary, indent=2) + "\n")
