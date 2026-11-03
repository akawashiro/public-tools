#!/usr/bin/env python3
"""Regenerate tiny deterministic HF checkpoints and independent Transformers logits.

Run with a Python environment containing torch, transformers, safetensors, tokenizers.
No model downloads are needed.
"""
import json
from pathlib import Path
import torch
from transformers import LlamaConfig, LlamaForCausalLM, Qwen3Config, Qwen3ForCausalLM, Qwen2Config, Qwen2ForCausalLM, MistralConfig, MistralForCausalLM
from safetensors.torch import save_file
from tokenizers import Tokenizer, models, pre_tokenizers, processors

root = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
torch.set_num_threads(1)
for name, cls, config_cls, dtype, shared in [
    ("llama-f32", LlamaForCausalLM, LlamaConfig, torch.float32, False),
    ("qwen3-f16", Qwen3ForCausalLM, Qwen3Config, torch.float16, False),
    ("qwen3-bf16-sharded", Qwen3ForCausalLM, Qwen3Config, torch.bfloat16, True),
    ("qwen2-f16", Qwen2ForCausalLM, Qwen2Config, torch.float16, True),
    ("qwen2-wide-f16", Qwen2ForCausalLM, Qwen2Config, torch.float16, False),
    ("llama3-f32", LlamaForCausalLM, LlamaConfig, torch.float32, False),
    ("llama-linear-f16", LlamaForCausalLM, LlamaConfig, torch.float16, False),
    ("mistral-window-f16", MistralForCausalLM, MistralConfig, torch.float16, False),
    ("mistral-wide-f16", MistralForCausalLM, MistralConfig, torch.float16, False),
    ("mistral-full-f32", MistralForCausalLM, MistralConfig, torch.float32, False),
]:
    path = root / name
    path.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(73)
    extra = {}
    if name == "llama3-f32":
        extra["rope_parameters"] = dict(rope_type="llama3", rope_theta=1234., factor=8.,
                                       low_freq_factor=1., high_freq_factor=4.,
                                       original_max_position_embeddings=64)
    elif name == "llama-linear-f16":
        extra["rope_parameters"] = dict(rope_type="linear", rope_theta=1234., factor=4.)
    if "mistral" in name:
        extra["sliding_window"] = None if "full" in name else 5 if "window" in name else 3
    head = 264 if "wide" in name else 128 if name == "mistral-window-f16" else 8
    context = 600 if name == "mistral-window-f16" else 96 if name == "llama3-f32" else 32
    cfg = config_cls(hidden_size=12 if "qwen3" in name else 16,
        intermediate_size=24, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=1, head_dim=head, vocab_size=16,
        max_position_embeddings=context, rms_norm_eps=0.002, rope_theta=1234.,
        bos_token_id=5, eos_token_id=7, tie_word_embeddings=shared, **extra)
    cfg._attn_implementation = "eager"
    m = cls(cfg).eval()
    if name not in {"llama-f32", "qwen3-f16", "qwen3-bf16-sharded"}:
        # Larger projections make wrong biases, RoPE, or window masks observable.
        with torch.no_grad():
            for n, param in m.named_parameters():
                if n.endswith(".bias"):
                    param.copy_(torch.linspace(-0.15, 0.25, param.numel()).reshape(param.shape))
                elif param.ndim == 2:
                    param.normal_(mean=0., std=0.07)
    # Nonconstant head normalization catches permutation mistakes that unit weights hide.
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "q_norm" in n or "k_norm" in n:
                p.copy_(torch.linspace(0.5, 1.5, p.numel()).reshape(p.shape))
    weights = {k: v.detach().to(dtype).contiguous().clone() for k, v in m.state_dict().items()}
    if shared: weights.pop("lm_head.weight", None)
    # Reference computes FP32 using exactly the rounded storage values, like Rust.
    m.load_state_dict({k: v.float() for k, v in weights.items()}, strict=not shared)
    tokens = [5, 9, 11, 4, 13]
    if name not in {"llama-f32", "qwen3-f16", "qwen3-bf16-sharded", "mistral-full-f32"}:
        count = 273 if name == "mistral-window-f16" else 80 if name == "llama3-f32" else 17
        tokens = [(i * 7 + 5) % 16 for i in range(count)]
    with torch.no_grad(): logits = m(torch.tensor([tokens])).logits[0].tolist()
    cfg_dict = cfg.to_dict()
    if name == "llama3-f32":
        scaling = dict(cfg_dict.pop("rope_parameters"))
        scaling.pop("rope_theta", None)
        cfg_dict["rope_theta"] = 1234.
        cfg_dict["rope_scaling"] = scaling
    elif name not in {"llama-linear-f16", "mistral-window-f16"}:
        cfg_dict.pop("rope_parameters", None)
        cfg_dict["rope_theta"] = 1234.
        cfg_dict["rope_scaling"] = None
    if name == "qwen2-f16":
        cfg_dict["sliding_window"] = 32768
        cfg_dict["use_sliding_window"] = False
    cfg_dict["architectures"] = [cls.__name__]
    (path / "config.json").write_text(json.dumps(cfg_dict, indent=2) + "\n")
    if "sharded" in name:
        items = list(weights.items())
        index = {}
        for i, subset in enumerate([items[::2], items[1::2]], 1):
            filename = f"model-{i:05d}-of-00002.safetensors"
            save_file(dict(subset), path / filename)
            index.update({k: filename for k, _ in subset})
        (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": index}, indent=2) + "\n")
    else: save_file(weights, path / "model.safetensors")
    vocab = {f"t{i}": i for i in range(16)}
    t = Tokenizer(models.WordLevel(vocab=vocab, unk_token="t0"))
    t.pre_tokenizer = pre_tokenizers.Whitespace()
    t.post_processor = processors.TemplateProcessing(single="t5 $A", special_tokens=[("t5", 5)])
    t.save(str(path / "tokenizer.json"))
    template = "{% for message in messages %}{% if message.content.startswith('t') %}{{ message.content.strip() }} {% endif %}{% endfor %}"
    (path / "tokenizer_config.json").write_text(json.dumps({"chat_template": template}) + "\n")
    (path / "generation_config.json").write_text(json.dumps({"eos_token_id": [7, 8]}) + "\n")
    (path / "expected.json").write_text(json.dumps({"tokens": tokens, "logits": logits}, indent=2) + "\n")
    print(name, sum(v.numel() for v in weights.values()))
