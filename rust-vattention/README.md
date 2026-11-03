# rust-vattention

Rust 製の CUDA Llama 推論エンジン。KV キャッシュは **vAttention 型の仮想連続メモリ**で管理します。
[vAttention のスライド](https://akawashiro.com/slides/vattention.pdf)の 18–21 ページを設計の基礎にしています。

## 実行

64-bit Linux、NVIDIA GPU、CUDA ドライバー、NVRTC、cuBLAS が必要です。CUDA 12.6 の Driver API を使用し、
カーネルは実機の compute capability に合わせて NVRTC でコンパイルします。
Blackwell GPU では CUDA 12.8 以降の NVRTC を使用してください。
Rust のバージョンは `rust-toolchain.toml` に固定しています。

```bash
bash scripts/download-model.sh
cargo build --release --locked
LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} \
  ./target/release/rust-vattention \
  --model models/stories15M.bin --tokenizer models/tokenizer.bin \
  --prompt "Once upon a time" --max-tokens 128 --temperature 0
```

デフォルトのバックエンドは CUDA。`--device 0`、`--kv-chunk-kib 64`、
`--temperature 0.8 --top-p 0.9 --seed 42` を指定できます。
`--max-tokens` は入力に追加するトークン数です。入力と生成がコンテキスト上限に達すると停止します。
生成文字列は stdout、速度と KV メモリ統計は stderr に出力します。
`--backend cpu --threads 4` で数値比較用の CPU 実装を利用できます。

## Hugging Face のモデル ID で実行

`--model` にはモデル ID、取得済みの Hugging Face snapshot ディレクトリ、従来の
`llama2.c` ファイルを指定できます。モデル ID の場合、設定・tokenizer・safetensors を
取得し、`models/hub` にキャッシュします。各ファイルは同じコミットに固定して取得します。

```bash
LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} \
  cargo run --release --locked -- \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --weight-dtype fp16 --context-length 4096 \
  --prompt "The capital of France is" --temperature 0 --max-tokens 64
```

- `model_type=llama` / `LlamaForCausalLM` と `model_type=qwen3` / `Qwen3ForCausalLM` に対応。
- 必要ファイルは `config.json`、`tokenizer.json`、`model.safetensors` または
  `model.safetensors.index.json` とその分割ファイル。F32/F16/BF16 を読み込みます。
- GQA、Qwen3 の Q/K RMSNorm、独立した `head_dim`、出力重み共有、
  `rms_norm_eps` / `rope_theta`、モデル固有の BOS/EOS に対応します。
- `--revision` でブランチ・タグ・コミット、`--cache-dir` でキャッシュ先を指定できます。
  private / gated モデルには利用許可と `HF_TOKEN`（または既存の HF token ファイル）が必要です。
- 再取得せずに実行する場合は `--model models/hub/models--ORG--NAME/snapshots/COMMIT` を指定します。
- GPU の `--weight-dtype fp16` は重みを半精度で保存します。prefill の行列積は FP16 入力・
  FP32 積和、attention は対応 GPU で TF32 Tensor Core を使います。decode の活性・積和と
  VMM の KV キャッシュは FP32 です。FP32 経路や vLLM と出力が完全一致する保証はありません。
  デフォルトは `fp32`。ホスト RAM 上の重みはどちらも FP32 なので十分な RAM が必要です。
- `--context-length` は元のコンテキスト上限を縮小します。大きな既定値のモデルでは指定を推奨します。
- 量子化、MoE、バイアス付きモデル、scaled RoPE（Llama 3.1 / YaRN など）、
  sliding-window attention、GGUF、PyTorch pickle は未対応です。未対応の設定は読み込み時に拒否します。
  Hugging Face にある任意のモデルが動くわけではありません。

## OpenAI 互換 API

```bash
LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} \
  ./target/release/rust-vattention --serve \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --weight-dtype fp16 --context-length 4096 --port 8000

curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-4B-Instruct-2507","messages":[{"role":"user","content":"日本の首都は？"}],"max_tokens":64,"temperature":0,"stream":true}'
```

`GET /health`、`GET /v1/models`、`POST /v1/completions`、
`POST /v1/chat/completions` を提供します。通常 JSON と SSE（`[DONE]`、
`stream_options.include_usage`）に対応し、OpenAI SDK の `base_url` に
`http://127.0.0.1:8000/v1` を指定して利用できます。API キーは検証しません。

chat は `tokenizer_config.json` の Jinja chat template を適用します。
テンプレートのない base モデルでは `/v1/completions` を使用してください。
Qwen3 の chat template には `enable_thinking=false` を渡します。
テキストの system/user/assistant メッセージ、`max_tokens` / `max_completion_tokens`、
`temperature`、`top_p`、`seed`、`stop`、`n=1` に対応します。
completions の `prompt` は文字列または token ID 配列を受け付けます。
ベンチマーク向けに `ignore_eos` も受け付けます。
リクエストは逐次実行し、KV マッピングを再利用します。
ツール呼び出し、画像・音声、logprobs、structured output、Responses API は未実装です。

API の HTTP 検証と、同じ `vllm bench serve --backend openai` を両サーバーに向ける比較スクリプト:

```bash
python3 scripts/test-api.py
python3 -m venv target/bench-venv
target/bench-venv/bin/pip install vllm
target/bench-venv/bin/python scripts/benchmark-servers.py
```

ベンチマークは事前取得済みの `models/hub` の snapshot を使い、同じ seed・入力長・出力長・
同時実行数で測定します。条件、起動コマンド、コミット、GPU メモリと vLLM の生データは
`benchmarks/results` に保存します。詳細は [初期ベンチマーク結果](benchmarks/README.md) と
[高速化後の比較・実験記録](benchmarks/optimization/README.md) を参照してください。
追加測定時は `--result-dir` で別の保存先を指定してください。

## KV メモリの設計

各レイヤーの K/V ごとに最大コンテキスト長に対応する連続仮想アドレス領域を `cuMemAddressReserve` で予約します。
予約直後の KV 物理メモリ割り当ては 0 バイトです。生成が進むと必要なプレフィックスに対して
`cuMemCreate` → `cuMemMap` → `cuMemSetAccess` を行います。
拡張時もベースアドレスは変わらず、既存の KV をコピーしません。

Attention カーネルは `base + token * stride` でアクセスします。PagedAttention やアプリケーションの
ブロックテーブルは使用しません。物理ハンドルの管理はホスト側のメモリ解放にのみ使用します。
ページサイズは `cuMemGetAllocationGranularity` の返す最小粒度に従い、要求サイズを切り上げます。
`64 KiB` を指定しても GPU によって実際の粒度は異なります。ドライバーの改変は行いません。

`CudaSession::reset()` はマッピングを維持して次のリクエストで再利用します。
`reset_and_release()` は同期後に物理メモリを解放し、仮想アドレス予約を維持します。
セッションの破棄時は物理メモリと仮想領域の両方を解放します。
`cache_stats()` の `reserved_bytes` は予約仮想領域、`mapped_bytes` は物理割り当て済みの KV 容量です。
重みとスクラッチ領域はこの統計に含めません。

## 推論と対応形式

- Rust 側でモデル読み込み、CUDA の起動、VMM のライフサイクル、BPE、サンプリングを実装。
- prompt は cuBLAS のバッチ行列積で処理し、末尾の logits だけを計算します。
- decode はベクトル化 GEMV、QKV / gate-up 重みのパッキング、融合 RMSNorm / RoPE を使用。
  CUDA Graph は KV 長の範囲ごとに再利用し、物理ページのマッピングは再生前に行います。
- 短い prompt は対応する head_dim=128 で融合 Tensor Core attention、長い prompt は
  strided cuBLAS attention を使用。長い decode は KV を計算区間に分割して並列化します。
  いずれも VMM の K/V を直接読み、ブロックテーブルや KV の詰め替えは使いません。
- greedy と temperature / top_p サンプリングを GPU で行い、選ばれた token ID の 4 バイトだけ
  pinned memory に転送します。nucleus sampling は語彙全体を安定ソートし、top-k 近似はしません。
  logits を必要とするライブラリ API と CPU バックエンドも維持しています。
- `llama2.c` の旧 FP32 形式（little endian、7 個の i32 ヘッダー）と `tokenizer.bin` に対応。
  正負の vocab_size による出力重みの共有・非共有をサポート。
- 同梱の取得スクリプトは MIT の TinyStories 15M モデル（約 61 MB）を SHA-256 で検証。
  このモデルは英語の物語補完用で、日本語対話向けではありません。

単一リクエストを逐次処理する実装です。GGUF、量子化、バッチ処理、
バックグラウンドの先行割り当てと演算のオーバーラップは未実装です。
資料の性能値を再現する実装ではなく、標準 CUDA VMM で設計を検証する実装です。
モデルの全重みをホスト RAM に読み込んでから GPU にコピーします。

## 検証

```bash
cargo test --locked
cargo clippy --locked --all-targets -- -D warnings
cargo fmt --check

# 実 GPU で VMM と CPU/CUDA の数値一致を検証
LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} \
  RUST_VATTENTION_TEST_MODEL=models/stories15M.bin \
  cargo test --release --locked -- --ignored --test-threads=1 --nocapture
```

Hugging Face 形式の追加テストは、ランダム初期化した小さな Llama / Qwen3 と
Transformers の logits を比較します。F32/F16/BF16、分割、Q/K 正規化、異なる head_dim、
不正な重み・index の拒否を検証します。`scripts/check-real-model.py` は取得済みの
Qwen3-0.6B で Transformers との greedy 出力一致、公式 chat template、OpenAI SDK の
通常応答とストリーミングを確認します（`target/bench-venv/bin/python` で実行）。

通常テストは tokenizer の Unicode/byte fallback、BPE、サンプリング、壊れたモデルの拒否、
RoPE、VMM のサイズ計算を確認します。GPU テストはページ境界を越える拡張、既存データと
アドレスの保持、解放・再割り当て、GQA、出力重みの共有/非共有、実モデル 80 ステップの logits を検証します。

2026-11-03 に masumi（RTX 5060 Ti 16 GB、ドライバー 580.178.04、NVRTC / cuBLAS 12.9）で
通常テスト 8 件・GPU テスト 11 件、Clippy、フォーマット検査の通過を確認しました。
TinyStories 15M の greedy 生成 64 ステップは、CUDA 版・CPU 版・`llama2.c` で完全一致しました。
この GPU の最小 VMM 割り当ては 2 MiB で、この小型モデルは最初のトークンで
各レイヤー K/V の予約領域がすべてマップされます（計 24 MiB）。
より大きな予約領域に対する段階的な割り当ては、別の VMM テストで実機確認しています。

## 参考・ライセンス

- [vAttention スライド](https://akawashiro.com/slides/vattention.pdf)
- [NVIDIA CUDA VMM](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VA.html)
- [llama2.c](https://github.com/karpathy/llama2.c): モデル形式・推論計算・BPE の参照元。MIT ライセンスを `LICENSE.llama2.c` に同梱。
- [TinyLlamas モデル](https://huggingface.co/karpathy/tinyllamas)

本プロジェクトは MIT ライセンスです。
