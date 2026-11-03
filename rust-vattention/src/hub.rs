//! Hugging Face Llama checkpoints. Remote code and pickle weights are never loaded.
use crate::model::{Config, Layout, Model};
use crate::product;
use anyhow::{bail, ensure, Context, Result};
use half::{bf16, f16};
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
use safetensors::{Dtype, SafeTensors};
use serde::Deserialize;
use serde_json::Value;
use std::{
    collections::{BTreeMap, HashSet},
    fs,
    ops::Range,
    path::{Component, Path, PathBuf},
};

pub struct HubModel {
    pub model: Model,
    pub tokenizer: tokenizers::Tokenizer,
    pub bos: Option<usize>,
    pub eos: Vec<usize>,
    pub chat_template: Option<String>,
}

impl HubModel {
    pub fn encode(&self, text: &str) -> Result<Vec<usize>> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("encode prompt: {e}"))?;
        let mut ids: Vec<_> = encoding.get_ids().iter().map(|&v| v as usize).collect();
        // The tokenizer postprocessor controls BOS; use the config only for empty input.
        if ids.is_empty() {
            ids.push(
                self.bos
                    .context("empty prompt and no bos_token_id configured")?,
            );
        }
        ensure!(
            ids.iter().all(|&v| v < self.model.config().vocab_size),
            "tokenizer emitted an ID outside model vocabulary"
        );
        Ok(ids)
    }
}

/// A directory can be an already downloaded HF snapshot. Other values are Hub IDs.
pub fn load(id: &str, revision: &str, cache: &Path) -> Result<HubModel> {
    if Path::new(id).is_dir() {
        let dir = Path::new(id);
        return load_files(
            |name| Ok(dir.join(name)),
            dir.join("generation_config.json").is_file(),
            dir.join("tokenizer_config.json").is_file(),
            dir.join("model.safetensors.index.json").is_file(),
        );
    }
    ensure!(
        !id.is_empty() && !id.starts_with('/') && !id.contains("..") && id.split('/').count() <= 2,
        "expected a Hugging Face model ID such as organization/model"
    );
    let mut builder = ApiBuilder::from_env().with_cache_dir(cache.to_owned());
    if let Ok(token) = std::env::var("HF_TOKEN") {
        builder = builder.with_token(Some(token));
    }
    let api = builder.build()?;
    let repo = api.repo(Repo::with_revision(
        id.to_owned(),
        RepoType::Model,
        revision.to_owned(),
    ));
    let info = repo.info().with_context(|| format!("access Hugging Face model {id}@{revision}; private/gated models require HF_TOKEN and access approval"))?;
    // Pin every download to the same commit, even if a branch moves during loading.
    eprintln!("Hugging Face: {id}@{}", info.sha);
    let resolved = info.sha.clone();
    let repo = api.repo(Repo::with_revision(
        id.to_owned(),
        RepoType::Model,
        info.sha,
    ));
    let names: HashSet<_> = info.siblings.into_iter().map(|s| s.rfilename).collect();
    ensure!(
        names.contains("config.json") && names.contains("tokenizer.json"),
        "model requires config.json and tokenizer.json"
    );
    let loaded = load_files(
        |name| {
            ensure!(
                names.contains(name),
                "missing {name}; expected Llama safetensors (single file or indexed shards)"
            );
            repo.get(name).with_context(|| format!("download {name}"))
        },
        names.contains("generation_config.json"),
        names.contains("tokenizer_config.json"),
        names.contains("model.safetensors.index.json"),
    )?;
    hf_hub::Cache::new(cache.to_owned())
        .repo(Repo::with_revision(
            id.to_owned(),
            RepoType::Model,
            revision.to_owned(),
        ))
        .create_ref(&resolved)?;
    Ok(loaded)
}

#[derive(Deserialize)]
struct WeightIndex {
    weight_map: BTreeMap<String, String>,
}

fn load_files(
    mut get: impl FnMut(&str) -> Result<PathBuf>,
    generation_config: bool,
    tokenizer_config: bool,
    sharded: bool,
) -> Result<HubModel> {
    let config: Value = serde_json::from_slice(&fs::read(get("config.json")?)?)?;
    let c = parse_config(&config)?;
    let tokenizer = tokenizers::Tokenizer::from_file(get("tokenizer.json")?)
        .map_err(|e| anyhow::anyhow!("load tokenizer.json: {e}"))?;
    ensure!(
        tokenizer
            .get_vocab(true)
            .values()
            .all(|&id| (id as usize) < c.vocab_size),
        "tokenizer vocabulary exceeds model vocabulary"
    );
    let bos = token_ids(&config, "bos_token_id", c.vocab_size)?
        .first()
        .copied();
    let mut eos = token_ids(&config, "eos_token_id", c.vocab_size)?;
    if generation_config {
        let generation: Value = serde_json::from_slice(&fs::read(get("generation_config.json")?)?)?;
        if !generation["eos_token_id"].is_null() {
            eos = token_ids(&generation, "eos_token_id", c.vocab_size)?;
        }
    }
    let chat_template = if tokenizer_config {
        let cfg: Value = serde_json::from_slice(&fs::read(get("tokenizer_config.json")?)?)?;
        cfg["chat_template"].as_str().map(str::to_owned)
    } else {
        None
    };
    let shared = config["tie_word_embeddings"].as_bool().unwrap_or(false);
    let (layout, size) = layout(&c, shared, config["model_type"] == "qwen3")?;
    let mut model = Model {
        config: c,
        layout,
        weights: vec![0.; size],
    };
    let mut targets = targets(&model);
    let index = if sharded {
        Some(serde_json::from_slice::<WeightIndex>(&fs::read(get(
            "model.safetensors.index.json",
        )?)?)?)
    } else {
        None
    };
    let files: HashSet<String> = match &index {
        Some(index) => {
            ensure!(!index.weight_map.is_empty(), "empty safetensors index");
            for name in targets.keys() {
                ensure!(
                    index.weight_map.contains_key(name),
                    "tensor {name} missing from index"
                );
            }
            index.weight_map.values().cloned().collect()
        }
        None => ["model.safetensors".to_owned()].into_iter().collect(),
    };
    let mut files: Vec<_> = files.into_iter().collect();
    files.sort();
    for filename in files {
        validate_filename(&filename)?;
        let path = get(&filename)?;
        eprintln!("loading {filename}");
        let bytes = fs::read(&path).with_context(|| format!("read {}", path.display()))?;
        let tensors =
            SafeTensors::deserialize(&bytes).with_context(|| format!("parse {filename}"))?;
        for (name, tensor) in tensors.tensors() {
            if name.ends_with(".bias") {
                bail!("unsupported bias tensor: {name}");
            }
            if let Some(target) = targets.get(&name) {
                if let Some(index) = &index {
                    ensure!(
                        index.weight_map[&name] == filename,
                        "tensor {name} found in wrong shard"
                    );
                }
                let (range, shape, heads) = target;
                ensure!(
                    tensor.shape() == shape,
                    "tensor {name}: expected shape {shape:?}, got {:?}",
                    tensor.shape()
                );
                let values = floats(tensor.dtype(), tensor.data()).with_context(|| name.clone())?;
                let dst = &mut model.weights[range.clone()];
                if let Some(heads) = heads {
                    // Transformers uses split-half RoPE; our kernels rotate adjacent pairs.
                    permute_qk(
                        dst,
                        &values,
                        *heads,
                        if shape.len() == 1 { 1 } else { shape[1] },
                    );
                } else {
                    dst.copy_from_slice(&values);
                }
                targets.remove(&name);
            }
        }
    }
    ensure!(
        targets.is_empty(),
        "missing model tensors: {:?}",
        targets.keys()
    );
    Ok(HubModel {
        model,
        tokenizer,
        bos,
        eos,
        chat_template,
    })
}

fn token_ids(v: &Value, key: &str, vocab: usize) -> Result<Vec<usize>> {
    let values = match &v[key] {
        Value::Null => return Ok(vec![]),
        Value::Array(a) => a.clone(),
        id => vec![id.clone()],
    };
    values
        .iter()
        .map(|v| {
            let id = v
                .as_u64()
                .context("special token IDs must be nonnegative integers")?;
            ensure!(id < vocab as u64, "{key} outside model vocabulary");
            Ok(id as usize)
        })
        .collect()
}

fn parse_config(v: &Value) -> Result<Config> {
    let qwen = v["model_type"] == "qwen3";
    ensure!(
        v["model_type"] == "llama" || qwen,
        "unsupported model_type {:?}; supported: llama, qwen3",
        v["model_type"]
    );
    if let Some(architectures) = v["architectures"].as_array() {
        let expected = if qwen {
            "Qwen3ForCausalLM"
        } else {
            "LlamaForCausalLM"
        };
        ensure!(
            architectures.iter().all(|a| a == expected),
            "unsupported architecture; expected {expected}"
        );
    }
    ensure!(
        v["quantization_config"].is_null(),
        "quantized models are unsupported"
    );
    ensure!(
        v["rope_scaling"].is_null(),
        "scaled RoPE (including Llama 3.1) is unsupported"
    );
    ensure!(
        v["rope_parameters"].is_null(),
        "rope_parameters is unsupported; expected rope_theta without scaling"
    );
    ensure!(
        v["hidden_act"].as_str().unwrap_or("silu") == "silu",
        "only SiLU activation is supported"
    );
    for key in ["attention_bias", "mlp_bias", "use_qk_norm"] {
        ensure!(v[key].is_null() || v[key] == false, "{key} is unsupported");
    }
    ensure!(
        v["sliding_window"].is_null(),
        "sliding-window attention is unsupported"
    );
    ensure!(
        v["partial_rotary_factor"].is_null() || v["partial_rotary_factor"] == 1.0,
        "partial RoPE is unsupported"
    );
    let dim = dimension(v, "hidden_size")?;
    let heads = dimension(v, "num_attention_heads")?;
    let kv_heads = if v["num_key_value_heads"].is_null() {
        heads
    } else {
        dimension(v, "num_key_value_heads")?
    };
    let head_dim = if v["head_dim"].is_null() {
        ensure!(
            dim.is_multiple_of(heads),
            "hidden_size must divide attention heads when head_dim is omitted"
        );
        dim / heads
    } else {
        dimension(v, "head_dim")?
    };
    ensure!(
        head_dim.is_multiple_of(2) && heads.is_multiple_of(kv_heads),
        "invalid attention dimensions"
    );
    let eps = v["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32;
    let theta = v["rope_theta"].as_f64().unwrap_or(10000.) as f32;
    ensure!(
        eps.is_finite() && eps > 0. && theta.is_finite() && theta > 0.,
        "invalid RMSNorm epsilon or RoPE theta"
    );
    let c = Config {
        dim,
        heads,
        head_dim,
        kv_heads,
        hidden_dim: dimension(v, "intermediate_size")?,
        layers: dimension(v, "num_hidden_layers")?,
        vocab_size: dimension(v, "vocab_size")?,
        context: dimension(v, "max_position_embeddings")?,
        rms_norm_eps: eps,
        rope_theta: theta,
    };
    product(&[c.layers, c.context, c.head_dim, c.kv_heads])?;
    Ok(c)
}

fn dimension(v: &Value, key: &str) -> Result<usize> {
    let n = usize::try_from(
        v[key]
            .as_u64()
            .with_context(|| format!("missing or invalid {key}"))?,
    )?;
    ensure!(n > 0, "{key} must be positive");
    Ok(n)
}

fn validate_filename(name: &str) -> Result<()> {
    ensure!(
        name.ends_with(".safetensors")
            && Path::new(name)
                .components()
                .all(|c| matches!(c, Component::Normal(_))),
        "invalid shard filename {name:?}"
    );
    Ok(())
}

fn floats(dtype: Dtype, data: &[u8]) -> Result<Vec<f32>> {
    let result: Vec<_> = match dtype {
        Dtype::F32 => data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect(),
        Dtype::F16 => data
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| f16::from_bits(u16::from_le_bytes(*b)).to_f32())
            .collect(),
        Dtype::BF16 => data
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| bf16::from_bits(u16::from_le_bytes(*b)).to_f32())
            .collect(),
        _ => bail!("unsupported weight dtype {dtype:?}; expected F32/F16/BF16"),
    };
    ensure!(
        result.iter().all(|v| v.is_finite()),
        "weights contain NaN or infinity"
    );
    Ok(result)
}

fn permute_qk(out: &mut [f32], input: &[f32], heads: usize, cols: usize) {
    let head = input.len() / cols / heads;
    for h in 0..heads {
        for i in 0..head / 2 {
            for half in 0..2 {
                let src = (h * head + half * (head / 2) + i) * cols;
                let dst = (h * head + i * 2 + half) * cols;
                out[dst..dst + cols].copy_from_slice(&input[src..src + cols]);
            }
        }
    }
}

fn layout(c: &Config, shared: bool, qk_norm: bool) -> Result<(Layout, usize)> {
    let mut offset = 0usize;
    let mut take = |shape: &[usize]| -> Result<Range<usize>> {
        let start = offset;
        offset = offset
            .checked_add(product(shape)?)
            .context("weight layout overflow")?;
        product(&[offset])?;
        Ok(start..offset)
    };
    let kv = c.head_dim * c.kv_heads;
    let embedding = take(&[c.vocab_size, c.dim])?;
    let att_norm = take(&[c.layers, c.dim])?;
    let qdim = c.heads * c.head_dim;
    let q = take(&[c.layers, qdim, c.dim])?;
    let k = take(&[c.layers, kv, c.dim])?;
    let v = take(&[c.layers, kv, c.dim])?;
    let o = take(&[c.layers, c.dim, qdim])?;
    let ffn_norm = take(&[c.layers, c.dim])?;
    // The existing 'up' range is the SiLU branch (HF gate_proj).
    let up = take(&[c.layers, c.hidden_dim, c.dim])?;
    let down = take(&[c.layers, c.dim, c.hidden_dim])?;
    let gate = take(&[c.layers, c.hidden_dim, c.dim])?;
    let norm = take(&[c.dim])?;
    let classifier = if shared {
        embedding.clone()
    } else {
        take(&[c.vocab_size, c.dim])?
    };
    let q_norm = if qk_norm {
        Some(take(&[c.layers, c.head_dim])?)
    } else {
        None
    };
    let k_norm = if qk_norm {
        Some(take(&[c.layers, c.head_dim])?)
    } else {
        None
    };
    Ok((
        Layout {
            embedding,
            att_norm,
            q,
            k,
            v,
            o,
            ffn_norm,
            up,
            down,
            gate,
            norm,
            classifier,
            q_norm,
            k_norm,
        },
        offset,
    ))
}

type Target = (Range<usize>, Vec<usize>, Option<usize>);
fn targets(m: &Model) -> BTreeMap<String, Target> {
    let c = &m.config;
    let l = &m.layout;
    let kv = c.head_dim * c.kv_heads;
    let mut targets = BTreeMap::new();
    targets.insert(
        "model.embed_tokens.weight".into(),
        (l.embedding.clone(), vec![c.vocab_size, c.dim], None),
    );
    targets.insert(
        "model.norm.weight".into(),
        (l.norm.clone(), vec![c.dim], None),
    );
    if l.classifier != l.embedding {
        targets.insert(
            "lm_head.weight".into(),
            (l.classifier.clone(), vec![c.vocab_size, c.dim], None),
        );
    }
    let qdim = c.heads * c.head_dim;
    for i in 0..c.layers {
        for (name, base, shape, heads) in [
            ("input_layernorm", &l.att_norm, vec![c.dim], None),
            ("post_attention_layernorm", &l.ffn_norm, vec![c.dim], None),
            ("self_attn.q_proj", &l.q, vec![qdim, c.dim], Some(c.heads)),
            ("self_attn.k_proj", &l.k, vec![kv, c.dim], Some(c.kv_heads)),
            ("self_attn.v_proj", &l.v, vec![kv, c.dim], None),
            ("self_attn.o_proj", &l.o, vec![c.dim, qdim], None),
            ("mlp.gate_proj", &l.up, vec![c.hidden_dim, c.dim], None),
            ("mlp.up_proj", &l.gate, vec![c.hidden_dim, c.dim], None),
            ("mlp.down_proj", &l.down, vec![c.dim, c.hidden_dim], None),
        ] {
            let size: usize = shape.iter().product();
            let start = base.start + i * size;
            targets.insert(
                format!("model.layers.{i}.{name}.weight"),
                (start..start + size, shape, heads),
            );
        }
    }
    for (name, range) in [("q_norm", &l.q_norm), ("k_norm", &l.k_norm)] {
        if let Some(range) = range {
            for i in 0..c.layers {
                let start = range.start + i * c.head_dim;
                targets.insert(
                    format!("model.layers.{i}.self_attn.{name}.weight"),
                    (start..start + c.head_dim, vec![c.head_dim], Some(1)),
                );
            }
        }
    }
    targets
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    fn config() -> Value {
        json!({"model_type":"qwen3", "hidden_size":12, "head_dim":8,
            "intermediate_size":24,"num_hidden_layers":2,"num_attention_heads":2,
            "num_key_value_heads":1,"vocab_size":16,"max_position_embeddings":32})
    }
    #[test]
    fn rejects_unsupported_or_invalid_configs() {
        assert!(parse_config(&config()).is_ok());
        for (key, value) in [
            ("model_type", json!("qwen3_moe")),
            ("architectures", json!(["Qwen3ForSequenceClassification"])),
            ("quantization_config", json!({"quant_method":"awq"})),
            ("rope_scaling", json!({"rope_type":"yarn"})),
            ("attention_bias", json!(true)),
            ("hidden_act", json!("gelu")),
            ("num_attention_heads", json!(0)),
            ("num_key_value_heads", json!(3)),
            ("head_dim", json!(7)),
            ("rms_norm_eps", json!(-1)),
        ] {
            let mut v = config();
            v[key] = value;
            assert!(parse_config(&v).is_err(), "accepted unsupported {key}");
        }
        for filename in [
            "/weights.safetensors",
            "../weights.safetensors",
            "a/../weights.safetensors",
            "pytorch_model.bin",
        ] {
            assert!(validate_filename(filename).is_err());
        }
    }
}
