# 対応モデル

Hugging Face の safetensors と tokenizer.json を読み込む dense テキスト生成モデルが対象です。
CLI と OpenAI 互換 API は同じローダーと推論実装を使います。

| 系統 | HF アーキテクチャ | 追加対応した処理 |
|---|---|---|
| Llama 2 / 3 / 3.1 / 3.2 | `LlamaForCausalLM` | Llama 3 の周波数帯別 RoPE 補間、linear RoPE |
| Qwen2 / Qwen2.5 | `Qwen2ForCausalLM` | QKV バイアス、Q/K バイアスの RoPE 配列変換 |
| Qwen3 | `Qwen3ForCausalLM` | 既存の Q/K RMSNorm と独立 head_dim |
| Mistral 7B v0.1 / v0.2 / v0.3 | `MistralForCausalLM` | full / sliding-window causal attention |

同じアーキテクチャ・設定の派生モデルにも適用します。MoE、画像・音声を含む派生モデルは対象外です。
Qwen2 のレイヤー混在型 sliding attention は未対応で、明示的に拒否します。
Qwen2.5 の `use_sliding_window=false` は、`sliding_window` の数値が残っていても full attention と解釈します。

## RoPE と KV

従来の `rope_scaling` と新しい `rope_parameters` の両形式で `default`、`linear`、`llama3` を読み込みます。
旧形式の `type` と現行の `rope_type` に対応します。YaRN、dynamic NTK、部分 RoPE、レイヤー別 RoPE は未対応です。
Llama 3 の `factor`、`low_freq_factor`、`high_freq_factor`、`original_max_position_embeddings` は
チェックポイントの値で計算し、`--context-length` による上限縮小とは独立して保持します。

Mistral の window は現在のトークンを含む直近の指定トークン数です。
prefill、継続 prefill、decode、CUDA Graph、長い KV の分割計算のすべてに同じマスクを適用します。
KV は従来どおり連続した仮想アドレスで管理し、トークン追加に応じて物理ページをマップします。
sliding window でも KV の物理容量は処理済みトークン数に応じて増えます。

## 実行例

```bash
LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
  target/release/rust-vattention \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --weight-dtype fp16 --context-length 4096 \
  --prompt "The capital of France is" --temperature 0 --max-tokens 32

# 同じモデルを OpenAI 互換 API で提供
LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
  target/release/rust-vattention --serve \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --weight-dtype fp16 --context-length 4096 --port 8000
```

Llama の gated モデルは Hugging Face 側の利用許可と認証が必要です。
GPU に重みと KV が収まる必要があるため、モデルサイズに合わせて dtype と context を選択してください。

## 検証

`tests/fixtures` の 10 種類の小型チェックポイントを Transformers 5.20.2 で独立に生成し、
CPU、CUDA FP32 / FP16 重みの各経路で logits を比較しています。

- 非ゼロ QKV バイアスと Q/K バイアスの配列変換。
- Llama 3 の低周波・高周波・補間帯と、元の上限を超える位置。
- linear RoPE と新旧の設定形式。
- Mistral の全域 attention と、5 トークン window・273 トークン入力。
- 16 / 256 / 257 トークン prefill、継続 prefill、その後の decode。
- head_dim=128 の融合 MMA / cuBLAS と、head_dim=264 の fallback。
- 既存の VMM 拡張・解放・再マッピングと CUDA Graph の再利用。

```bash
cargo test --locked
LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
 RUST_VATTENTION_TEST_MODEL=models/stories15M.bin \
 cargo test --locked -- --ignored --test-threads=1
```

Llama 3 と Mistral はこの小型モデルでのアーキテクチャ検証です。
各公開モデルの全サイズをダウンロードして検証したものではありません。

2026-11-03、RTX 5060 Ti 16 GB で通常テスト 8 件・GPU テスト 11 件、Clippy、format、API 検証を通過しました。
GPU テストの件数は 11 のままですが、HF の各比較テストは上記 10 チェックポイントを巡回します。

公開実モデルでは次の 2 つで、FP16 重みの greedy 生成、120 / 400 トークンの prefill と継続生成、
chat template の token 数、OpenAI SDK の JSON / SSE を検証しました。

| 実モデル | 固定リビジョン |
|---|---|
| Qwen/Qwen2.5-0.5B-Instruct | `7ae557604adf67be50417f59c2c2f167def9a775` |
| Qwen/Qwen3-0.6B | `c1899de289a04d12100db370d81485cdf75e47ca` |

Qwen2.5 の配布 generation_config は repetition penalty を指定するため、比較時は Transformers 側も
API と同じ penalty なしの greedy 条件に揃えます。モデル配布元の生成パラメータを API に自動適用する仕様ではありません。

既存 Qwen3 の直接計測は prefill 約 7.33 ms、decode 約 3.77 ms/token（128 入力 / 64 decode、3 回）で、
変更前の約 7.33 ms / 3.77 ms と同水準でした。これは HTTP を含まない回帰確認です。

実モデルの greedy 生成・公式 chat template・OpenAI SDK の JSON / SSE 検証は
事前に対象 snapshot を取得してから次のコマンドで再現できます。

```bash
LD_LIBRARY_PATH=/usr/local/cuda/lib64 target/bench-venv/bin/python \
 scripts/check-real-model.py --model Qwen/Qwen2.5-0.5B-Instruct
```

仕様の参照元:
[Qwen2.5 の公式 config](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/main/config.json)、
[Mistral 7B の公式 config](https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/config.json)、
[Transformers の RoPE 実装](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py)。
