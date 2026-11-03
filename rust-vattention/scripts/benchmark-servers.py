#!/usr/bin/env python3
"""Compare both servers using the same vLLM OpenAI HTTP benchmark client.

Prerequisites: cargo build --release; a venv with vllm; HF snapshots in models/hub.
Example: target/bench-venv/bin/python scripts/benchmark-servers.py
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--models", nargs="+", default=["Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B-Instruct-2507"])
p.add_argument("--engines", nargs="+", default=["rust", "vllm"], choices=["rust", "vllm"])
p.add_argument("--venv", type=Path, default=ROOT / "target/bench-venv")
p.add_argument("--result-dir", type=Path, default=ROOT / "benchmarks/results")
p.add_argument("--port", type=int, default=8100)
p.add_argument("--temperature", type=float, default=0.0)
p.add_argument("--top-p", type=float, default=1.0)
p.add_argument("--input-len", type=int, default=128)
p.add_argument("--output-len", type=int, default=64)
p.add_argument("--context-length", type=int, default=512)
p.add_argument("--kv-cache-mib", type=int, default=256)
p.add_argument("--num-prompts", type=int, default=10)
p.add_argument("--repeats", type=int, default=3)
p.add_argument("--force", action="store_true", help="Repeat and replace completed runs")
a = p.parse_args()
a.result_dir.mkdir(parents=True, exist_ok=True)
env = dict(os.environ, LD_LIBRARY_PATH="/usr/local/cuda/lib64" + (":"+os.environ["LD_LIBRARY_PATH"] if os.environ.get("LD_LIBRARY_PATH") else ""), VLLM_USAGE_STATS_ENABLED="0", DO_NOT_TRACK="1")
# CUDA 12.9 does not accept the host default GCC 15.
if Path("/usr/bin/g++-13").exists():
    env.update(CC="/usr/bin/gcc-13", CXX="/usr/bin/g++-13", NVCC_PREPEND_FLAGS="-ccbin=/usr/bin/g++-13")

def gpu_memory():
    return int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True).strip().splitlines()[0])

metadata = {"date":time.strftime("%Y-%m-%dT%H:%M:%S%z"), "input_len":a.input_len, "output_len":a.output_len, "num_prompts":a.num_prompts, "repeats":a.repeats, "concurrency":1, "temperature":a.temperature, "top_p":a.top_p, "seed":42, "context_length":a.context_length, "vllm_kv_cache_mib":a.kv_cache_mib, "gpu":subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], text=True).strip(), "versions":subprocess.check_output([str(a.venv/"bin/python"), "-c", "import vllm,torch,transformers;print(vllm.__version__,torch.__version__,transformers.__version__)"], text=True).strip()}
(a.result_dir/"environment.json").write_text(json.dumps(metadata, indent=2)+"\n")

for model in a.models:
    cache = ROOT / "models/hub" / ("models--"+model.replace("/", "--"))
    revision = (cache/"refs/main").read_text().strip() if (cache/"refs/main").exists() else next((cache/"snapshots").iterdir()).name
    snapshot = cache / "snapshots" / revision
    assert (snapshot/"config.json").exists(), snapshot
    for engine in a.engines:
        name = model.split("/")[-1]+"-"+engine
        if not a.force and all((a.result_dir/f"{name}-{i}.json").exists() for i in range(a.repeats)):
            print("skip completed", name, flush=True)
            continue
        if engine == "rust":
            cmd = [str(ROOT/"target/release/rust-vattention"), "--serve", "--model", str(snapshot), "--weight-dtype", "fp16", "--context-length", str(a.context_length), "--port", str(a.port)]
            served_name = str(snapshot)
        else:
            cmd = [str(a.venv/"bin/vllm"), "serve", str(snapshot), "--served-model-name", model, "--dtype", "half", "--max-model-len", str(a.context_length), "--max-num-seqs", "1", "--max-num-batched-tokens", str(a.context_length), "--disable-prefix-caching", "--kv-cache-memory-bytes", str(a.kv_cache_mib*1024**2), "--generation-config", "vllm", "--host", "127.0.0.1", "--port", str(a.port)]
            served_name = model
        run = {"engine":engine,"model":model,"revision":revision,"command":cmd,"baseline_gpu_mib":gpu_memory(),"benchmark_commands":[],"rust_binary_sha256":hashlib.file_digest((ROOT/"target/release/rust-vattention").open("rb"),"sha256").hexdigest(),"build_environment":{k:env[k] for k in ["CC","CXX","NVCC_PREPEND_FLAGS"] if k in env},"runtime_environment":{k:v for k,v in env.items() if k.startswith("RUST_VATTENTION_")}}
        with (a.result_dir/f"{name}-server.log").open("w") as log:
            server = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            stop = threading.Event()
            samples = []
            def sample_memory():
                while not stop.wait(.25):
                    try: samples.append(gpu_memory())
                    except Exception: pass
            monitor = threading.Thread(target=sample_memory, daemon=True)
            monitor.start()
            try:
                deadline = time.monotonic()+600
                while time.monotonic() < deadline:
                    if server.poll() is not None: raise RuntimeError(f"{name} server exited: see {log.name}")
                    try:
                        with urllib.request.urlopen(f"http://127.0.0.1:{a.port}/health",timeout=1) as r:
                            if r.status == 200: break
                    except OSError: time.sleep(1)
                else: raise TimeoutError(name)
                run["ready_gpu_mib"] = gpu_memory()
                print(name, "ready", "GPU MiB",run["ready_gpu_mib"], flush=True)
                for i in range(a.repeats):
                    filename = f"{name}-{i}.json"
                    if not a.force and (a.result_dir/filename).exists(): continue
                    bench = [str(a.venv/"bin/vllm"), "bench", "serve", "--backend", "openai", "--base-url", f"http://127.0.0.1:{a.port}", "--endpoint", "/v1/completions", "--model", str(snapshot), "--served-model-name", served_name, "--tokenizer", str(snapshot), "--dataset-name", "random", "--random-input-len", str(a.input_len), "--random-output-len", str(a.output_len), "--random-range-ratio", "0", "--num-prompts", str(a.num_prompts), "--num-warmups", "2", "--max-concurrency", "1", "--request-rate", "inf", "--temperature", str(a.temperature), "--top-p", str(a.top_p), "--ignore-eos", "--seed", "42", "--save-result", "--save-detailed", "--result-dir", str(a.result_dir), "--result-filename", filename, "--percentile-metrics", "ttft,tpot,itl,e2el"]
                    run["benchmark_commands"].append(bench)
                    with (a.result_dir/f"{name}-{i}.log").open("w") as benchlog:
                        subprocess.run(bench, env=env, stdout=benchlog, stderr=subprocess.STDOUT, check=True, timeout=1800)
                    result = json.loads((a.result_dir/filename).read_text())
                    assert result["completed"] == a.num_prompts and result.get("failed",0) == 0, result
                    print(name, i, "output tok/s",result.get("output_throughput"),"TTFT ms",result.get("mean_ttft_ms"), flush=True)
            finally:
                stop.set()
                monitor.join()
                run["peak_gpu_mib"] = max(samples, default=0)
                (a.result_dir/f"{name}-run.json").write_text(json.dumps(run, indent=2)+"\n")
                try: os.killpg(server.pid, signal.SIGTERM)
                except ProcessLookupError: pass
                try: server.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, signal.SIGKILL)
                    server.wait()
                time.sleep(3)
