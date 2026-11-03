use crate::{product, softmax, word};
use anyhow::{ensure, Context, Result};
use rayon::prelude::*;
use std::{fs::File, io::Read, ops::Range, path::Path};

#[derive(Clone, Debug)]
pub struct Config {
    pub dim: usize,
    pub hidden_dim: usize,
    pub layers: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub kv_heads: usize,
    pub vocab_size: usize,
    pub context: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    /// Precomputed frequencies for static scaled RoPE; None uses the original formula.
    pub rope_inv_freq: Option<Vec<f32>>,
    /// Number of visible keys including the current token; None is full causal attention.
    pub sliding_window: Option<usize>,
}

#[derive(Clone)]
pub(crate) struct Layout {
    pub(crate) embedding: Range<usize>,
    pub(crate) att_norm: Range<usize>,
    pub(crate) q: Range<usize>,
    pub(crate) k: Range<usize>,
    pub(crate) v: Range<usize>,
    pub(crate) o: Range<usize>,
    pub(crate) ffn_norm: Range<usize>,
    pub(crate) up: Range<usize>,
    pub(crate) down: Range<usize>,
    pub(crate) gate: Range<usize>,
    pub(crate) norm: Range<usize>,
    pub(crate) classifier: Range<usize>,
    pub(crate) q_norm: Option<Range<usize>>,
    pub(crate) k_norm: Option<Range<usize>>,
    pub(crate) qkv_bias: Option<Range<usize>>,
}

/// Weights are immutable; each session owns its KV cache and activation buffers.
pub struct Model {
    pub(crate) config: Config,
    pub(crate) weights: Vec<f32>,
    pub(crate) layout: Layout,
}

impl Model {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path.as_ref())
            .with_context(|| format!("open {}", path.as_ref().display()))?;
        Self::read(file)
    }

    pub(crate) fn read(mut reader: impl Read) -> Result<Self> {
        let mut header = [0i32; 7];
        for value in &mut header {
            *value = i32::from_le_bytes(word(&mut reader).context("truncated checkpoint header")?);
        }
        let shared = header[5] > 0;
        header[5] = header[5].checked_abs().context("invalid vocabulary size")?;
        ensure!(
            header.iter().all(|v| *v > 0),
            "all model dimensions must be positive (expected llama2.c FP32)"
        );
        let [dim, hidden_dim, layers, heads, kv_heads, vocab_size, context] =
            header.map(|v| v as usize);
        ensure!(
            dim % heads == 0 && (dim / heads) % 2 == 0,
            "head size must be even and divide dim"
        );
        ensure!(
            heads % kv_heads == 0,
            "KV heads must divide attention heads"
        );
        ensure!(
            vocab_size >= 259,
            "vocabulary must contain BOS/EOS and 256 byte tokens"
        );
        let config = Config {
            dim,
            hidden_dim,
            layers,
            heads,
            head_dim: dim / heads,
            kv_heads,
            vocab_size,
            context,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.,
            rope_inv_freq: None,
            sliding_window: None,
        };
        let kv = product(&[dim / heads, kv_heads])?;
        product(&[layers, context, kv])?;
        let mut offset = 0usize;
        let mut take = |shape: &[usize]| -> Result<Range<usize>> {
            let start = offset;
            offset = offset
                .checked_add(product(shape)?)
                .context("weight layout overflow")?;
            Ok(start..offset)
        };
        let embedding = take(&[vocab_size, dim])?;
        let att_norm = take(&[layers, dim])?;
        let q = take(&[layers, dim, dim])?;
        let k = take(&[layers, kv, dim])?;
        let v = take(&[layers, kv, dim])?;
        let o = take(&[layers, dim, dim])?;
        let ffn_norm = take(&[layers, dim])?;
        let up = take(&[layers, hidden_dim, dim])?;
        let down = take(&[layers, dim, hidden_dim])?;
        let gate = take(&[layers, hidden_dim, dim])?;
        let norm = take(&[dim])?;
        take(&[context, dim / heads])?; // Legacy precomputed RoPE tables.
        let classifier = if shared {
            embedding.clone()
        } else {
            take(&[vocab_size, dim])?
        };
        let expected = product(&[offset, 4])?;
        let mut bytes = Vec::new();
        reader.take(expected as u64 + 1).read_to_end(&mut bytes)?;
        ensure!(
            bytes.len() == expected,
            "checkpoint weight size mismatch: expected {expected} bytes, got {}",
            bytes.len()
        );
        let weights: Vec<f32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        ensure!(
            weights.iter().all(|v| v.is_finite()),
            "checkpoint contains non-finite weights"
        );
        Ok(Self {
            config,
            weights,
            layout: Layout {
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
                q_norm: None,
                k_norm: None,
                qkv_bias: None,
            },
        })
    }

    /// Restrict the allocated context without changing positional encoding.
    pub fn limit_context(&mut self, context: usize) -> Result<()> {
        ensure!(
            context > 0 && context <= self.config.context,
            "context length must be in 1..={}",
            self.config.context
        );
        self.config.context = context;
        Ok(())
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    pub fn session(&self) -> Result<Session<'_>> {
        Session::new(self)
    }
}

pub struct Session<'a> {
    model: &'a Model,
    position: usize,
    keys: Vec<f32>,
    values: Vec<f32>,
    x: Vec<f32>,
    xb: Vec<f32>,
    residual: Vec<f32>,
    q: Vec<f32>,
    hidden: Vec<f32>,
    gate: Vec<f32>,
    scores: Vec<f32>,
    logits: Vec<f32>,
}

fn zeros(n: usize) -> Result<Vec<f32>> {
    let mut out = Vec::new();
    out.try_reserve_exact(n)
        .context("unable to allocate inference buffers")?;
    out.resize(n, 0.);
    Ok(out)
}

impl<'a> Session<'a> {
    fn new(model: &'a Model) -> Result<Self> {
        let c = &model.config;
        let cache = product(&[c.layers, c.context, c.head_dim, c.kv_heads])?;
        Ok(Self {
            model,
            position: 0,
            keys: zeros(cache)?,
            values: zeros(cache)?,
            x: zeros(c.dim)?,
            xb: zeros(c.dim.max(c.heads * c.head_dim))?,
            residual: zeros(c.dim)?,
            q: zeros(c.heads * c.head_dim)?,
            hidden: zeros(c.hidden_dim)?,
            gate: zeros(c.hidden_dim)?,
            scores: zeros(c.context)?,
            logits: zeros(c.vocab_size)?,
        })
    }

    pub fn position(&self) -> usize {
        self.position
    }

    /// Forget prior tokens. Old cache entries are overwritten before being read.
    pub fn reset(&mut self) {
        self.position = 0;
    }

    /// Consume exactly one token and return next-token logits.
    pub fn forward(&mut self, token: usize) -> Result<&[f32]> {
        let c = &self.model.config;
        ensure!(token < c.vocab_size, "token {token} is outside vocabulary");
        ensure!(
            self.position < c.context,
            "context limit ({}) reached",
            c.context
        );
        let pos = self.position;
        let d = c.dim;
        let head = c.head_dim;
        let qdim = c.heads * head;
        let kv = head * c.kv_heads;
        let w = &self.model.weights;
        let l = &self.model.layout;
        self.x.copy_from_slice(
            &w[l.embedding.start + token * d..l.embedding.start + (token + 1) * d],
        );
        for layer in 0..c.layers {
            rmsnorm(
                &mut self.xb,
                &self.x,
                &w[l.att_norm.start + layer * d..][..d],
                c.rms_norm_eps,
            );
            let cache = layer * c.context * kv;
            let current = cache + pos * kv;
            matvec(
                &mut self.q,
                &self.xb[..d],
                &w[l.q.start + layer * qdim * d..][..qdim * d],
            );
            matvec(
                &mut self.keys[current..current + kv],
                &self.xb[..d],
                &w[l.k.start + layer * kv * d..][..kv * d],
            );
            matvec(
                &mut self.values[current..current + kv],
                &self.xb[..d],
                &w[l.v.start + layer * kv * d..][..kv * d],
            );
            if let Some(bias) = &l.qkv_bias {
                let bias = &w[bias.start + layer * (qdim + 2 * kv)..][..qdim + 2 * kv];
                for (x, b) in self.q.iter_mut().zip(&bias[..qdim]) {
                    *x += b;
                }
                for (x, b) in self.keys[current..current + kv]
                    .iter_mut()
                    .zip(&bias[qdim..qdim + kv])
                {
                    *x += b;
                }
                for (x, b) in self.values[current..current + kv]
                    .iter_mut()
                    .zip(&bias[qdim + kv..])
                {
                    *x += b;
                }
            }
            if let (Some(qn), Some(kn)) = (&l.q_norm, &l.k_norm) {
                head_norm(
                    &mut self.q,
                    head,
                    &w[qn.start + layer * head..][..head],
                    c.rms_norm_eps,
                );
                head_norm(
                    &mut self.keys[current..current + kv],
                    head,
                    &w[kn.start + layer * head..][..head],
                    c.rms_norm_eps,
                );
            }
            rope(
                &mut self.q,
                head,
                pos,
                c.rope_theta,
                c.rope_inv_freq.as_deref(),
            );
            rope(
                &mut self.keys[current..current + kv],
                head,
                pos,
                c.rope_theta,
                c.rope_inv_freq.as_deref(),
            );
            self.xb.fill(0.);
            let first_key = c
                .sliding_window
                .map_or(0, |window| (pos + 1).saturating_sub(window));
            for h in 0..c.heads {
                let kh = h / (c.heads / c.kv_heads) * head;
                let query = &self.q[h * head..(h + 1) * head];
                for t in first_key..=pos {
                    let key = &self.keys[cache + t * kv + kh..][..head];
                    self.scores[t] = dot(query, key) / (head as f32).sqrt();
                }
                softmax(&mut self.scores[first_key..=pos]);
                for t in first_key..=pos {
                    let value = &self.values[cache + t * kv + kh..][..head];
                    for (out, v) in self.xb[h * head..(h + 1) * head].iter_mut().zip(value) {
                        *out += self.scores[t] * v;
                    }
                }
            }
            matvec(
                &mut self.residual,
                &self.xb[..qdim],
                &w[l.o.start + layer * d * qdim..][..d * qdim],
            );
            for (x, r) in self.x.iter_mut().zip(&self.residual) {
                *x += r;
            }
            rmsnorm(
                &mut self.xb,
                &self.x,
                &w[l.ffn_norm.start + layer * d..][..d],
                c.rms_norm_eps,
            );
            let f = d * c.hidden_dim;
            matvec(
                &mut self.hidden,
                &self.xb[..d],
                &w[l.up.start + layer * f..][..f],
            );
            matvec(
                &mut self.gate,
                &self.xb[..d],
                &w[l.gate.start + layer * f..][..f],
            );
            for (h, g) in self.hidden.iter_mut().zip(&self.gate) {
                *h = *h / (1. + (-*h).exp()) * g;
            }
            matvec(
                &mut self.residual,
                &self.hidden,
                &w[l.down.start + layer * f..][..f],
            );
            for (x, r) in self.x.iter_mut().zip(&self.residual) {
                *x += r;
            }
        }
        rmsnorm(&mut self.xb, &self.x, &w[l.norm.clone()], c.rms_norm_eps);
        matvec(&mut self.logits, &self.xb[..d], &w[l.classifier.clone()]);
        ensure!(
            self.logits.iter().all(|v| v.is_finite()),
            "non-finite logits; reset the session before retrying"
        );
        self.position += 1;
        Ok(&self.logits)
    }
}

fn head_norm(x: &mut [f32], head: usize, weights: &[f32], eps: f32) {
    for part in x.chunks_exact_mut(head) {
        let scale = (dot(part, part) / head as f32 + eps).sqrt().recip();
        for (v, w) in part.iter_mut().zip(weights) {
            *v *= scale * w;
        }
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

fn matvec(out: &mut [f32], input: &[f32], matrix: &[f32]) {
    out.par_iter_mut()
        .zip(matrix.par_chunks_exact(input.len()))
        .for_each(|(out, row)| *out = dot(row, input));
}

fn rmsnorm(out: &mut [f32], input: &[f32], weights: &[f32], eps: f32) {
    let scale = (dot(input, input) / input.len() as f32 + eps)
        .sqrt()
        .recip();
    for ((o, x), w) in out.iter_mut().zip(input).zip(weights) {
        *o = x * scale * w;
    }
}

fn rope(vector: &mut [f32], head: usize, position: usize, theta: f32, inv_freq: Option<&[f32]>) {
    for (i, pair) in vector.as_chunks_mut::<2>().0.iter_mut().enumerate() {
        let angle = inv_freq.map_or_else(
            || position as f32 / theta.powf((i * 2 % head) as f32 / head as f32),
            |freq| position as f32 * freq[i % (head / 2)],
        );
        let (sin, cos) = angle.sin_cos();
        let (a, b) = (pair[0], pair[1]);
        pair[0] = a * cos - b * sin;
        pair[1] = a * sin + b * cos;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_invalid_checkpoints() {
        assert!(Model::read(&[0u8; 28][..]).is_err());
        assert!(Model::read(&[0u8; 3][..]).is_err());
        let header: Vec<u8> = [4i32, 8, 1, 2, 1, 259, 8]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        assert!(Model::read(header.as_slice()).is_err());
    }
    #[test]
    fn rope_preserves_norm() {
        let mut v = vec![1., 2., 3., 4., 5., 6., 7., 8.];
        let before = dot(&v, &v);
        rope(&mut v, 4, 7, 10000., None);
        assert!((dot(&v, &v) - before).abs() < 1e-4);
    }
}
