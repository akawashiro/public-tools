# rust-vattention

Rust 製の CUDA Llama 推論エンジン。KV キャッシュは **vAttention 型の仮想連続メモリ**で管理します。
[vAttention のスライド](https://akawashiro.com/slides/vattention.pdf)の 18–21 ページを設計の基礎にしています。

## 実行

64-bit Linux、NVIDIA GPU、CUDA ドライバー、NVRTC が必要です。CUDA 12.6 の Driver API を使用し、
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
- CUDA C の小さなカーネルで FP32 行列積、RMSNorm、RoPE、GQA、causal attention、SwiGLU を実行。
- KV キャッシュ、重み、活性は GPU に保持。サンプリング用の logits のみ毎ステップ CPU に転送。
- `llama2.c` の旧 FP32 形式（little endian、7 個の i32 ヘッダー）と `tokenizer.bin` に対応。
  正負の vocab_size による出力重みの共有・非共有をサポート。
- 同梱の取得スクリプトは MIT の TinyStories 15M モデル（約 61 MB）を SHA-256 で検証。
  このモデルは英語の物語補完用で、日本語対話向けではありません。

初版は単一リクエストを逐次処理する実装です。GGUF/safetensors、量子化、バッチ処理、
FlashAttention、バックグラウンドの先行割り当てと演算のオーバーラップは未実装です。
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

通常テストは tokenizer の Unicode/byte fallback、BPE、サンプリング、壊れたモデルの拒否、
RoPE、VMM のサイズ計算を確認します。GPU テストはページ境界を越える拡張、既存データと
アドレスの保持、解放・再割り当て、GQA、出力重みの共有/非共有、実モデル 80 ステップの logits を検証します。

2026-11-02 に masumi（RTX 5060 Ti 16 GB、ドライバー 580.178.04、NVRTC 12.9）で
通常テスト 5 件・GPU テスト 3 件、Clippy、フォーマット検査の通過を確認しました。
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
