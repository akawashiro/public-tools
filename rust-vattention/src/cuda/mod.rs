//! GPU transformer and vAttention-style KV allocation using CUDA Driver VMM.
mod vmm;
use crate::{
    model::{Config, Layout, Model},
    product,
};
use anyhow::{ensure, Context, Result};
use cudarc::{
    driver::{sys, CudaContext, CudaModule, LaunchConfig, PushKernelArg},
    nvrtc::{compile_ptx_with_opts, CompileOptions},
};
use std::{collections::HashMap, sync::Arc};
use vmm::VmmBuffer;

fn check(result: sys::CUresult, operation: &str) -> Result<()> {
    ensure!(
        result == sys::CUresult::CUDA_SUCCESS,
        "{operation}: {result:?}"
    );
    Ok(())
}

struct Buffer {
    ctx: Arc<CudaContext>,
    ptr: u64,
    len: usize,
}
impl Buffer {
    fn new(ctx: &Arc<CudaContext>, len: usize) -> Result<Self> {
        ctx.bind_to_thread()?;
        let bytes = product(&[len, 4])?;
        let mut ptr = 0;
        // SAFETY: allocation output pointer is valid; this object owns the result.
        unsafe {
            check(sys::cuMemAlloc_v2(&mut ptr, bytes), "cuMemAlloc")?;
        }
        Ok(Self {
            ctx: ctx.clone(),
            ptr,
            len,
        })
    }
    fn upload(&self, data: &[f32]) -> Result<()> {
        ensure!(data.len() == self.len, "upload length mismatch");
        self.ctx.bind_to_thread()?;
        // SAFETY: host source and device destination span exactly len floats;
        // synchronous transfer completes before the host borrow ends.
        unsafe {
            check(
                sys::cuMemcpyHtoD_v2(self.ptr, data.as_ptr().cast(), self.len * 4),
                "cuMemcpyHtoD",
            )
        }
    }
    fn download(&self, out: &mut [f32]) -> Result<()> {
        ensure!(out.len() == self.len, "download length mismatch");
        self.ctx.synchronize()?;
        // SAFETY: both allocations span len floats; preceding kernels completed.
        unsafe {
            check(
                sys::cuMemcpyDtoH_v2(out.as_mut_ptr().cast(), self.ptr, self.len * 4),
                "cuMemcpyDtoH",
            )
        }
    }
}
impl Drop for Buffer {
    fn drop(&mut self) {
        if self.ctx.synchronize().is_ok() {
            // SAFETY: no kernel can still use this exclusively owned allocation.
            unsafe {
                let _ = sys::cuMemFree_v2(self.ptr);
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CacheStats {
    pub reserved_bytes: usize,
    pub mapped_bytes: usize,
}

/// Single-request decoder. GPU weights, scratch, and cache all live on device.
/// Host sampling transfers only the next-token logits back to the CPU.
pub struct CudaSession {
    ctx: Arc<CudaContext>,
    module: Arc<CudaModule>,
    functions: HashMap<&'static str, cudarc::driver::CudaFunction>,
    config: Config,
    layout: Layout,
    weights: Buffer,
    keys: Vec<VmmBuffer>,
    values: Vec<VmmBuffer>,
    x: Buffer,
    xb: Buffer,
    residual: Buffer,
    q: Buffer,
    hidden: Buffer,
    gate: Buffer,
    scores: Buffer,
    logits: Buffer,
    host_logits: Vec<f32>,
    position: usize,
}

impl CudaSession {
    pub fn new(model: &Model, device: usize, chunk_bytes: usize) -> Result<Self> {
        let ctx = CudaContext::new(device).context("initialize CUDA device")?;
        let (major, minor) = ctx.compute_capability()?;
        let options = CompileOptions {
            fmad: Some(false),
            options: vec![format!("--gpu-architecture=compute_{major}{minor}")],
            ..Default::default()
        };
        let ptx = compile_ptx_with_opts(include_str!("kernels.cu"), options)
            .context("compile CUDA kernels with NVRTC")?;
        let module = ctx.load_module(ptx)?;
        let mut functions = HashMap::new();
        for name in [
            "embedding",
            "matvec",
            "rmsnorm",
            "rope",
            "add",
            "swiglu",
            "scores",
            "attention_softmax",
            "attention_values",
        ] {
            functions.insert(name, module.load_function(name)?);
        }
        let c = &model.config;
        let weights = Buffer::new(&ctx, model.weights.len())?;
        weights.upload(&model.weights)?;
        let cache_bytes = product(&[c.context, c.dim / c.heads, c.kv_heads, 4])?;
        let mut keys = Vec::new();
        let mut values = Vec::new();
        for _ in 0..c.layers {
            keys.push(VmmBuffer::reserve(ctx.clone(), cache_bytes, chunk_bytes)?);
            values.push(VmmBuffer::reserve(ctx.clone(), cache_bytes, chunk_bytes)?);
        }
        eprintln!(
            "CUDA device: {}; KV cache initially maps 0 physical bytes",
            ctx.name()?
        );
        Ok(Self {
            x: Buffer::new(&ctx, c.dim)?,
            xb: Buffer::new(&ctx, c.dim)?,
            residual: Buffer::new(&ctx, c.dim)?,
            q: Buffer::new(&ctx, c.dim)?,
            hidden: Buffer::new(&ctx, c.hidden_dim)?,
            gate: Buffer::new(&ctx, c.hidden_dim)?,
            scores: Buffer::new(&ctx, product(&[c.heads, c.context])?)?,
            logits: Buffer::new(&ctx, c.vocab_size)?,
            host_logits: vec![0.; c.vocab_size],
            ctx,
            module,
            functions,
            config: c.clone(),
            layout: model.layout.clone(),
            weights,
            keys,
            values,
            position: 0,
        })
    }

    pub fn cache_stats(&self) -> CacheStats {
        CacheStats {
            reserved_bytes: self
                .keys
                .iter()
                .chain(&self.values)
                .map(|b| b.reserved)
                .sum(),
            mapped_bytes: self.keys.iter().chain(&self.values).map(|b| b.mapped).sum(),
        }
    }
    pub fn position(&self) -> usize {
        self.position
    }

    /// Retain mappings for reuse by the next request (deferred reclamation).
    pub fn reset(&mut self) {
        self.position = 0;
    }

    /// Release physical memory while retaining virtual address reservations.
    pub fn reset_and_release(&mut self) -> Result<()> {
        self.ctx.synchronize()?;
        self.position = 0;
        for buffer in self.keys.iter_mut().chain(&mut self.values) {
            buffer.release()?;
        }
        Ok(())
    }

    fn launch(&self, name: &'static str, args: &[u64], blocks: usize) -> Result<()> {
        self.ctx.bind_to_thread()?;
        let blocks = u32::try_from(blocks).context("CUDA grid exceeds u32")?;
        let stream = self.ctx.default_stream();
        let mut builder = stream.launch_builder(&self.functions[name]);
        for arg in args {
            builder.arg(arg);
        }
        // SAFETY: this private method is called only with the matching kernel ABI.
        // All arguments are 64-bit pointers/integers. Device allocations outlive
        // queued work. VMM prefixes are mapped before use; default-stream order
        // provides dependencies between producer and consumer kernels.
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })?;
        }
        Ok(())
    }
    fn matvec(&self, out: u64, x: u64, w: usize, n: usize, rows: usize) -> Result<()> {
        self.launch("matvec", &[out, x, self.weight(w), n as u64], rows)
    }
    fn weight(&self, offset: usize) -> u64 {
        self.weights.ptr + offset as u64 * 4
    }
    fn norm(&self, out: u64, input: u64, w: usize) -> Result<()> {
        self.launch(
            "rmsnorm",
            &[out, input, self.weight(w), self.config.dim as u64],
            1,
        )
    }

    pub fn forward(&mut self, token: usize) -> Result<&[f32]> {
        ensure!(token < self.config.vocab_size, "token outside vocabulary");
        ensure!(self.position < self.config.context, "context limit reached");
        let c = &self.config;
        let d = c.dim;
        let head = d / c.heads;
        let kv = head * c.kv_heads;
        let pos = self.position;
        let length = pos + 1;
        // Allocate before scheduling any kernels. No mapping is changed while
        // in-flight work might use it; previous forward ends with a sync copy.
        for b in self.keys.iter_mut().chain(&mut self.values) {
            b.ensure_bytes(length * kv * 4)?;
        }
        self.launch(
            "embedding",
            &[
                self.x.ptr,
                self.weight(self.layout.embedding.start),
                token as u64,
                d as u64,
            ],
            d.div_ceil(256),
        )?;
        for layer in 0..c.layers {
            let l = &self.layout;
            let key = self.keys[layer].address;
            let value = self.values[layer].address;
            let current_key = key + (pos * kv * 4) as u64;
            let current_value = value + (pos * kv * 4) as u64;
            self.norm(self.xb.ptr, self.x.ptr, l.att_norm.start + layer * d)?;
            self.matvec(self.q.ptr, self.xb.ptr, l.q.start + layer * d * d, d, d)?;
            self.matvec(current_key, self.xb.ptr, l.k.start + layer * kv * d, d, kv)?;
            self.matvec(
                current_value,
                self.xb.ptr,
                l.v.start + layer * kv * d,
                d,
                kv,
            )?;
            self.launch(
                "rope",
                &[self.q.ptr, d as u64, head as u64, pos as u64],
                (d / 2).div_ceil(256),
            )?;
            self.launch(
                "rope",
                &[current_key, kv as u64, head as u64, pos as u64],
                (kv / 2).div_ceil(256),
            )?;
            let kv_mul = (c.heads / c.kv_heads) as u64;
            self.launch(
                "scores",
                &[
                    self.scores.ptr,
                    self.q.ptr,
                    key,
                    head as u64,
                    kv as u64,
                    kv_mul,
                    length as u64,
                ],
                c.heads * length,
            )?;
            self.launch(
                "attention_softmax",
                &[self.scores.ptr, length as u64],
                c.heads,
            )?;
            self.launch(
                "attention_values",
                &[
                    self.xb.ptr,
                    self.scores.ptr,
                    value,
                    d as u64,
                    head as u64,
                    kv as u64,
                    kv_mul,
                    length as u64,
                ],
                d.div_ceil(256),
            )?;
            self.matvec(
                self.residual.ptr,
                self.xb.ptr,
                l.o.start + layer * d * d,
                d,
                d,
            )?;
            self.launch(
                "add",
                &[self.x.ptr, self.residual.ptr, d as u64],
                d.div_ceil(256),
            )?;
            self.norm(self.xb.ptr, self.x.ptr, l.ffn_norm.start + layer * d)?;
            let f = d * c.hidden_dim;
            self.matvec(
                self.hidden.ptr,
                self.xb.ptr,
                l.up.start + layer * f,
                d,
                c.hidden_dim,
            )?;
            self.matvec(
                self.gate.ptr,
                self.xb.ptr,
                l.gate.start + layer * f,
                d,
                c.hidden_dim,
            )?;
            self.launch(
                "swiglu",
                &[self.hidden.ptr, self.gate.ptr, c.hidden_dim as u64],
                c.hidden_dim.div_ceil(256),
            )?;
            self.matvec(
                self.residual.ptr,
                self.hidden.ptr,
                l.down.start + layer * f,
                c.hidden_dim,
                d,
            )?;
            self.launch(
                "add",
                &[self.x.ptr, self.residual.ptr, d as u64],
                d.div_ceil(256),
            )?;
        }
        self.norm(self.xb.ptr, self.x.ptr, self.layout.norm.start)?;
        self.matvec(
            self.logits.ptr,
            self.xb.ptr,
            self.layout.classifier.start,
            d,
            c.vocab_size,
        )?;
        self.logits.download(&mut self.host_logits)?;
        ensure!(
            self.host_logits.iter().all(|v| v.is_finite()),
            "non-finite CUDA logits"
        );
        self.position += 1;
        Ok(&self.host_logits)
    }
}

impl Drop for CudaSession {
    fn drop(&mut self) {
        let _ = self.ctx.synchronize();
        // Keep the module alive until queued kernels complete.
        let _ = &self.module;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn compare(cpu: &[f32], gpu: &[f32]) {
        assert_eq!(cpu.len(), gpu.len());
        for (i, (&a, &b)) in cpu.iter().zip(gpu).enumerate() {
            assert!(
                (a - b).abs() <= 0.003 + 0.001 * a.abs(),
                "logit {i}: CPU={a}, CUDA={b}"
            );
        }
    }

    #[test]
    #[ignore = "requires a CUDA GPU and driver"]
    fn cuda_gqa_shared_and_unshared_weights() -> Result<()> {
        for shared in [true, false] {
            let vocab = if shared { 259 } else { -259 };
            let mut bytes: Vec<u8> = [8i32, 16, 2, 2, 1, vocab, 128]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            let sizes = [
                259 * 8,
                2 * 8,
                2 * 8 * 8,
                2 * 4 * 8,
                2 * 4 * 8,
                2 * 8 * 8,
                2 * 8,
                2 * 16 * 8,
                2 * 8 * 16,
                2 * 16 * 8,
                8,
                128 * 4,
                if shared { 0 } else { 259 * 8 },
            ];
            for (block, n) in sizes.into_iter().enumerate() {
                for i in 0..n {
                    let v = if [1, 6, 10].contains(&block) {
                        1.0
                    } else {
                        ((i + block * 7) as f32 * 0.37).sin() * 0.1
                    };
                    bytes.extend(v.to_le_bytes());
                }
            }
            let model = Model::read(bytes.as_slice())?;
            let mut cpu = model.session()?;
            let mut gpu = CudaSession::new(&model, 0, 64 * 1024)?;
            assert_eq!(gpu.cache_stats().mapped_bytes, 0);
            let first = gpu.forward(1)?.to_vec();
            compare(cpu.forward(1)?, &first);
            for token in [4, 128, 88, 33, 5, 258, 1] {
                compare(cpu.forward(token)?, gpu.forward(token)?);
            }
            let mapped = gpu.cache_stats().mapped_bytes;
            gpu.reset();
            assert_eq!(gpu.cache_stats().mapped_bytes, mapped);
            compare(&first, gpu.forward(1)?);
            gpu.reset_and_release()?;
            assert_eq!(gpu.cache_stats().mapped_bytes, 0);
            compare(&first, gpu.forward(1)?);
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires CUDA GPU and RUST_VATTENTION_TEST_MODEL checkpoint path"]
    fn cuda_real_checkpoint_matches_cpu() -> Result<()> {
        let path = std::env::var("RUST_VATTENTION_TEST_MODEL")
            .context("set RUST_VATTENTION_TEST_MODEL")?;
        let model = Model::load(path)?;
        let mut cpu = model.session()?;
        let mut gpu = CudaSession::new(&model, 0, 64 * 1024)?;
        let mut token = 1;
        for _ in 0..80.min(model.config.context) {
            let a = cpu.forward(token)?;
            let b = gpu.forward(token)?;
            compare(a, b);
            token = a
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0;
        }
        eprintln!("80-token CPU/CUDA comparison: {:?}", gpu.cache_stats());
        Ok(())
    }
}
