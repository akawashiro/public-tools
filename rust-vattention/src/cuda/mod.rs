//! GPU transformer and vAttention-style KV allocation using CUDA Driver VMM.
mod attention;
mod decode;
mod prefill;
mod sampling;
mod vmm;
mod weights;
use crate::{
    model::{Config, Layout, Model},
    product,
};
use anyhow::{ensure, Context, Result};
use cudarc::{
    cublas::CudaBlas,
    driver::{sys, CudaContext, CudaModule, CudaStream, LaunchConfig, PushKernelArg},
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
    bytes: usize,
}
impl Buffer {
    fn new(ctx: &Arc<CudaContext>, len: usize) -> Result<Self> {
        Self::new_bytes(ctx, product(&[len, 4])?)
    }
    fn new_bytes(ctx: &Arc<CudaContext>, bytes: usize) -> Result<Self> {
        ctx.bind_to_thread()?;
        let mut ptr = 0;
        // SAFETY: allocation output pointer is valid; this object owns the result.
        unsafe {
            check(sys::cuMemAlloc_v2(&mut ptr, bytes), "cuMemAlloc")?;
        }
        Ok(Self {
            ctx: ctx.clone(),
            ptr,
            bytes,
        })
    }
    #[cfg(test)]
    fn upload(&self, data: &[f32]) -> Result<()> {
        ensure!(data.len() * 4 == self.bytes, "upload length mismatch");
        self.ctx.bind_to_thread()?;
        // SAFETY: host source and device destination span exactly len floats;
        // synchronous transfer completes before the host borrow ends.
        unsafe {
            check(
                sys::cuMemcpyHtoD_v2(self.ptr, data.as_ptr().cast(), self.bytes),
                "cuMemcpyHtoD",
            )?;
        }
        // Pageable HtoD may return after host staging but before device DMA.
        // Device users on a nonblocking stream must observe completion.
        self.ctx.synchronize()?;
        Ok(())
    }
    fn download(&self, out: &mut [f32]) -> Result<()> {
        ensure!(out.len() * 4 == self.bytes, "download length mismatch");
        self.ctx.synchronize()?;
        // SAFETY: both allocations span len floats; preceding kernels completed.
        unsafe {
            check(
                sys::cuMemcpyDtoH_v2(out.as_mut_ptr().cast(), self.ptr, self.bytes),
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
    stream: Arc<CudaStream>,
    decode_graphs: HashMap<usize, decode::DecodeGraph>,
    greedy_graph: Option<decode::DecodeGraph>,
    pinned_token: decode::PinnedMemory,
    use_graphs: bool,
    prefill_limit: usize,
    module: Arc<CudaModule>,
    functions: HashMap<&'static str, cudarc::driver::CudaFunction>,
    blas: CudaBlas,
    prefill: Option<prefill::Scratch>,
    sampling: Option<sampling::Sampling>,
    attention: Option<attention::Attention>,
    tensor_attention: bool,
    mma_attention: bool,
    mma_limit: usize,
    config: Config,
    layout: Layout,
    weights: Buffer,
    weight_bytes: usize,
    keys: Vec<VmmBuffer>,
    values: Vec<VmmBuffer>,
    x: Buffer,
    xb: Buffer,
    residual: Buffer,
    q: Buffer,
    qkv: Buffer,
    gateup: Buffer,
    hidden: Buffer,
    gate: Buffer,
    scores: Buffer,
    attention_partials: Buffer,
    attention_segments: usize,
    logits: Buffer,
    argmax_values: Buffer,
    argmax_ids: Buffer,
    next_token: Buffer,
    host_logits: Vec<f32>,
    position: usize,
}

impl CudaSession {
    pub fn new(model: &Model, device: usize, chunk_bytes: usize) -> Result<Self> {
        Self::new_with_fp16(model, device, chunk_bytes, false)
    }

    /// FP16 stores weights at half size; activations, KV, and sums remain FP32.
    pub fn new_with_fp16(
        model: &Model,
        device: usize,
        chunk_bytes: usize,
        fp16: bool,
    ) -> Result<Self> {
        let prefill_limit = std::env::var("RUST_VATTENTION_PREFILL_BATCH")
            .map_or(Ok(4096), |v| v.parse::<usize>())?;
        ensure!(
            (1..=4096).contains(&prefill_limit),
            "RUST_VATTENTION_PREFILL_BATCH must be in 1..=4096"
        );
        let ctx = CudaContext::new(device).context("initialize CUDA device")?;
        let (major, minor) = ctx.compute_capability()?;
        let mut options = CompileOptions {
            fmad: Some(false),
            options: vec![format!("--gpu-architecture=compute_{major}{minor}")],
            ..Default::default()
        };
        if fp16 {
            options.options.push("-DWEIGHTS_FP16=1".into());
        }
        let ptx = compile_ptx_with_opts(
            format!(
                "{}\n{}\n{}\n{}\n{}\n{}",
                include_str!("kernels.cu"),
                include_str!("packed.cu"),
                include_str!("sampling.cu"),
                include_str!("attention.cu"),
                include_str!("split_attention.cu"),
                include_str!("mma_attention.cu")
            ),
            options,
        )
        .context("compile CUDA kernels with NVRTC")?;
        let module = ctx.load_module(ptx)?;
        let mut functions = HashMap::new();
        for name in [
            "embedding",
            "set_step",
            "argmax",
            "mma_causal_attention",
            "decode_attention_partial",
            "decode_attention_merge",
            "causal_probabilities",
            "attention_output_half",
            "set_sampling",
            "sampling_init",
            "sampling_sum",
            "sampling_keys",
            "radix_hist",
            "radix_offsets",
            "radix_scatter",
            "ordered_mass",
            "mass_prefix",
            "sample_final",
            "embedding_step",
            "qkv_transform",
            "swiglu_pair",
            "matvec",
            "rmsnorm",
            "head_rmsnorm",
            "rope",
            "add",
            "swiglu",
            "scores",
            "attention_softmax",
            "attention_values",
            "embedding_batch",
            "norm_batch",
            "add_norm",
            "to_half",
            "batch_attention",
        ] {
            functions.insert(name, module.load_function(name)?);
        }
        let c = &model.config;
        let weight_bytes = if fp16 { 2 } else { 4 };
        let (weights, layout) = weights::upload(&ctx, model, fp16)?;
        let cache_bytes = product(&[c.context, c.head_dim, c.kv_heads, 4])?;
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
        let stream = ctx.new_stream()?;
        Ok(Self {
            x: Buffer::new(&ctx, c.dim)?,
            xb: Buffer::new(&ctx, c.dim.max(c.heads * c.head_dim))?,
            residual: Buffer::new(&ctx, c.dim)?,
            q: Buffer::new(&ctx, c.heads * c.head_dim)?,
            qkv: Buffer::new(&ctx, (c.heads + 2 * c.kv_heads) * c.head_dim)?,
            gateup: Buffer::new(&ctx, 2 * c.hidden_dim)?,
            hidden: Buffer::new(&ctx, c.hidden_dim)?,
            gate: Buffer::new(&ctx, c.hidden_dim)?,
            scores: Buffer::new(
                &ctx,
                if c.head_dim > 256 {
                    product(&[c.heads, c.context])?
                } else {
                    1
                },
            )?,
            attention_partials: Buffer::new(
                &ctx,
                product(&[c.heads, c.context.div_ceil(256), c.head_dim + 2])?,
            )?,
            attention_segments: c.context.div_ceil(256),
            logits: Buffer::new(&ctx, c.vocab_size)?,
            argmax_values: Buffer::new(&ctx, c.vocab_size.div_ceil(1024))?,
            argmax_ids: Buffer::new_bytes(&ctx, c.vocab_size.div_ceil(1024) * 4)?,
            next_token: Buffer::new_bytes(&ctx, 4)?,
            host_logits: vec![0.; c.vocab_size],
            blas: CudaBlas::new(stream.clone())?,
            stream,
            decode_graphs: HashMap::new(),
            greedy_graph: None,
            pinned_token: decode::PinnedMemory::new(ctx.clone(), 4)?,
            use_graphs: std::env::var_os("RUST_VATTENTION_DISABLE_GRAPHS").is_none(),
            prefill: None,
            sampling: None,
            attention: None,
            mma_limit: if std::env::var("RUST_VATTENTION_PREFILL_ATTENTION").as_deref() == Ok("mma")
            {
                usize::MAX
            } else {
                256
            },
            mma_attention: fp16
                && major >= 8
                && c.head_dim == 128
                && !matches!(
                    std::env::var("RUST_VATTENTION_PREFILL_ATTENTION").as_deref(),
                    Ok("blas" | "online")
                ),
            tensor_attention: fp16
                && major >= 8
                && std::env::var("RUST_VATTENTION_PREFILL_ATTENTION").as_deref() != Ok("online"),
            prefill_limit,
            ctx,
            module,
            functions,
            config: c.clone(),
            layout,
            weights,
            weight_bytes,
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
        self.launch_shared(name, args, blocks, 0)
    }
    fn launch_shared(
        &self,
        name: &'static str,
        args: &[u64],
        blocks: usize,
        shared: usize,
    ) -> Result<()> {
        self.launch_threads(name, args, blocks, shared, 256)
    }
    fn launch_threads(
        &self,
        name: &'static str,
        args: &[u64],
        blocks: usize,
        shared: usize,
        threads: u32,
    ) -> Result<()> {
        self.ctx.bind_to_thread()?;
        let blocks = u32::try_from(blocks).context("CUDA grid exceeds u32")?;
        let stream = &self.stream;
        let mut builder = stream.launch_builder(&self.functions[name]);
        for arg in args {
            builder.arg(arg);
        }
        // SAFETY: this private method is called only with the matching kernel ABI.
        // All arguments are 64-bit pointers/integers. Device allocations outlive
        // queued work. VMM prefixes are mapped before use; private-stream order
        // provides dependencies between producer and consumer kernels.
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: u32::try_from(shared)?,
            })?;
        }
        Ok(())
    }
    fn matvec(&self, out: u64, x: u64, w: usize, n: usize, rows: usize) -> Result<()> {
        self.launch(
            "matvec",
            &[out, x, self.weight(w), n as u64, rows as u64],
            rows.div_ceil(8),
        )
    }
    fn weight(&self, offset: usize) -> u64 {
        self.weights.ptr + offset as u64 * self.weight_bytes as u64
    }
    fn norm(&self, out: u64, input: u64, w: usize) -> Result<()> {
        self.launch(
            "rmsnorm",
            &[
                out,
                input,
                self.weight(w),
                self.config.dim as u64,
                self.config.rms_norm_eps.to_bits() as u64,
            ],
            1,
        )
    }

    fn forward_legacy(&mut self, token: usize) -> Result<&[f32]> {
        ensure!(token < self.config.vocab_size, "token outside vocabulary");
        ensure!(self.position < self.config.context, "context limit reached");
        let c = &self.config;
        let d = c.dim;
        let head = c.head_dim;
        let qdim = c.heads * head;
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
            self.matvec(
                self.q.ptr,
                self.xb.ptr,
                l.q.start + layer * (qdim + 2 * kv) * d,
                d,
                qdim,
            )?;
            self.matvec(
                current_key,
                self.xb.ptr,
                l.k.start + layer * (qdim + 2 * kv) * d,
                d,
                kv,
            )?;
            self.matvec(
                current_value,
                self.xb.ptr,
                l.v.start + layer * (qdim + 2 * kv) * d,
                d,
                kv,
            )?;
            if let (Some(qn), Some(kn)) = (&l.q_norm, &l.k_norm) {
                self.launch(
                    "head_rmsnorm",
                    &[
                        self.q.ptr,
                        self.weight(qn.start + layer * head),
                        head as u64,
                        c.rms_norm_eps.to_bits() as u64,
                    ],
                    c.heads,
                )?;
                self.launch(
                    "head_rmsnorm",
                    &[
                        current_key,
                        self.weight(kn.start + layer * head),
                        head as u64,
                        c.rms_norm_eps.to_bits() as u64,
                    ],
                    c.kv_heads,
                )?;
            }
            self.launch(
                "rope",
                &[
                    self.q.ptr,
                    qdim as u64,
                    head as u64,
                    pos as u64,
                    c.rope_theta.to_bits() as u64,
                ],
                (qdim / 2).div_ceil(256),
            )?;
            self.launch(
                "rope",
                &[
                    current_key,
                    kv as u64,
                    head as u64,
                    pos as u64,
                    c.rope_theta.to_bits() as u64,
                ],
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
                    qdim as u64,
                    head as u64,
                    kv as u64,
                    kv_mul,
                    length as u64,
                ],
                qdim.div_ceil(256),
            )?;
            self.matvec(
                self.residual.ptr,
                self.xb.ptr,
                l.o.start + layer * d * qdim,
                qdim,
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
                l.up.start + layer * 3 * f,
                d,
                c.hidden_dim,
            )?;
            self.matvec(
                self.gate.ptr,
                self.xb.ptr,
                l.gate.start + layer * 3 * f,
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
                l.down.start + layer * 3 * f,
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
    #[test]
    #[ignore = "requires CUDA and cuBLAS"]
    fn chunked_prefill_preserves_vmm_and_logits() -> Result<()> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/qwen3-bf16-sharded");
        let mut model = crate::hub::load(
            path.to_str().unwrap(),
            "main",
            std::path::Path::new("unused"),
        )?
        .model;
        model.config.context = 600;
        let tokens: Vec<_> = (0..300).map(|i| (i * 7 + 3) % 16).collect();
        for fp16 in [false, true] {
            let mut cpu = model.session()?;
            let mut expected = Vec::new();
            for &token in &tokens {
                expected = cpu.forward(token)?.to_vec();
            }
            let mut gpu = CudaSession::new_with_fp16(&model, 0, 65536, fp16)?;
            gpu.prefill_limit = 256;
            gpu.prefill(&tokens[..5])?;
            gpu.forward(7)?;
            assert!(!gpu.decode_graphs.is_empty());
            gpu.reset_and_release()?;
            let addresses: Vec<_> = gpu
                .keys
                .iter()
                .chain(&gpu.values)
                .map(|b| b.address)
                .collect();
            assert_eq!(gpu.cache_stats().mapped_bytes, 0);
            compare(&expected, gpu.prefill(&tokens)?);
            assert_eq!(gpu.position(), 300);
            for token in [2, 7, 11, 1, 9] {
                compare(cpu.forward(token)?, gpu.forward(token)?);
            }
            assert_eq!(gpu.position(), 305);
            assert_eq!(
                addresses,
                gpu.keys
                    .iter()
                    .chain(&gpu.values)
                    .map(|b| b.address)
                    .collect::<Vec<_>>()
            );
            gpu.reset_and_release()?;
            assert_eq!(gpu.cache_stats().mapped_bytes, 0);
            compare(&expected, gpu.prefill(&tokens)?);
        }
        Ok(())
    }
    #[test]
    #[ignore = "requires CUDA, cuBLAS and RUST_VATTENTION_TEST_MODEL"]
    fn decode_graph_survives_physical_mapping_growth() -> Result<()> {
        let path = std::env::var("RUST_VATTENTION_TEST_MODEL")?;
        let mut model = Model::load(path)?;
        model.config.context = 4096;
        let mut graph = CudaSession::new_with_fp16(&model, 0, 65536, true)?;
        let mut eager = CudaSession::new_with_fp16(&model, 0, 65536, true)?;
        eager.use_graphs = false;
        graph.forward(1)?;
        assert!(!graph.decode_graphs.is_empty());
        let stride = model.config.head_dim * model.config.kv_heads * 4;
        let prefix = graph.keys[0].mapped / stride;
        ensure!(
            prefix + 8 < model.config.context,
            "test model cannot reach VMM boundary"
        );
        let tokens: Vec<_> = (0..prefix)
            .map(|i| (i * 17 + 3) % model.config.vocab_size)
            .collect();
        let addresses: Vec<_> = graph
            .keys
            .iter()
            .chain(&graph.values)
            .map(|b| b.address)
            .collect();
        for _ in 0..2 {
            graph.reset_and_release()?;
            eager.reset_and_release()?;
            compare(eager.prefill(&tokens)?, graph.prefill(&tokens)?);
            let mapped = graph.cache_stats().mapped_bytes;
            for token in [1, 7, 9, 2, 13, 17, 21, 23] {
                compare(eager.forward(token)?, graph.forward(token)?);
            }
            assert!(graph.cache_stats().mapped_bytes > mapped);
            assert_eq!(
                addresses,
                graph
                    .keys
                    .iter()
                    .chain(&graph.values)
                    .map(|b| b.address)
                    .collect::<Vec<_>>()
            );
            assert!(!graph.decode_graphs.is_empty());
        }
        Ok(())
    }
    #[test]
    #[ignore = "requires CUDA and RUST_VATTENTION_TEST_MODEL"]
    fn gpu_argmax_handles_ties_boundaries_and_nonfinite() -> Result<()> {
        let model = Model::load(std::env::var("RUST_VATTENTION_TEST_MODEL")?)?;
        ensure!(
            model.config.vocab_size > 1024,
            "test requires multiple argmax blocks"
        );
        let mut gpu = CudaSession::new(&model, 0, 65536)?;
        let mut values = vec![-9.; model.config.vocab_size];
        values[1023] = 2.;
        values[1024] = 2.;
        gpu.logits.upload(&values)?;
        assert_eq!(gpu.greedy_from_logits()?, 1023);
        values.fill(f32::MIN);
        gpu.logits.upload(&values)?;
        assert_eq!(gpu.greedy_from_logits()?, 0);
        values.fill(-0.0);
        values[1] = 0.0;
        gpu.logits.upload(&values)?;
        assert_eq!(gpu.greedy_from_logits()?, 0);
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            values.fill(-1.);
            values[2] = 100.;
            values[1024] = bad;
            gpu.logits.upload(&values)?;
            assert!(gpu.greedy_from_logits().is_err());
        }
        Ok(())
    }
    #[test]
    #[ignore = "requires SM80+ CUDA and RUST_VATTENTION_TEST_MODEL"]
    fn mma_attention_matches_dense_reference_from_vmm() -> Result<()> {
        let model = Model::load(std::env::var("RUST_VATTENTION_TEST_MODEL")?)?;
        let gpu = CudaSession::new(&model, 0, 65536)?;
        for (n, pos, heads, kv_heads) in [
            (1usize, 0usize, 2usize, 1usize),
            (17, 0, 2, 1),
            (63, 11, 4, 2),
            (128, 3, 2, 2),
            (257, 0, 2, 1),
        ] {
            let prefix = n + pos;
            let q: Vec<f32> = (0..n * heads * 128)
                .map(|i| ((i * 37 % 1031) as f32 - 515.) / 512.)
                .collect();
            let k: Vec<f32> = (0..prefix * kv_heads * 128)
                .map(|i| ((i * 53 % 2003) as f32 - 1001.) / 1024.)
                .collect();
            let v: Vec<f32> = (0..prefix * kv_heads * 128)
                .map(|i| ((i * 17 % 601) as f32 - 300.) / 128.)
                .collect();
            let qbuf = Buffer::new(&gpu.ctx, q.len())?;
            qbuf.upload(&q)?;
            let mut key = VmmBuffer::reserve(gpu.ctx.clone(), k.len() * 4, 65536)?;
            let mut value = VmmBuffer::reserve(gpu.ctx.clone(), v.len() * 4, 65536)?;
            key.ensure_bytes(k.len() * 4)?;
            value.ensure_bytes(v.len() * 4)?;
            // SAFETY: mapped prefixes exactly cover the CPU tensor uploads.
            unsafe {
                check(
                    sys::cuMemcpyHtoD_v2(key.address, k.as_ptr().cast(), k.len() * 4),
                    "upload attention K",
                )?;
                check(
                    sys::cuMemcpyHtoD_v2(value.address, v.as_ptr().cast(), v.len() * 4),
                    "upload attention V",
                )?;
            }
            gpu.ctx.synchronize()?;
            let output = Buffer::new_bytes(&gpu.ctx, q.len() * 2)?;
            gpu.launch_threads(
                "mma_causal_attention",
                &[
                    output.ptr,
                    qbuf.ptr,
                    key.address,
                    value.address,
                    n as u64,
                    heads as u64,
                    kv_heads as u64,
                    pos as u64,
                ],
                n.div_ceil(16) * heads,
                0,
                128,
            )?;
            gpu.stream.synchronize()?;
            let mut bits = vec![0u16; q.len()];
            // SAFETY: the kernel completed and output contains exactly q.len FP16 values.
            unsafe {
                check(
                    sys::cuMemcpyDtoH_v2(bits.as_mut_ptr().cast(), output.ptr, bits.len() * 2),
                    "read attention output",
                )?;
            }
            let actual: Vec<_> = bits
                .into_iter()
                .map(|b| half::f16::from_bits(b).to_f32())
                .collect();
            let mut maximum_error = 0f32;
            for t in 0..n {
                for h in 0..heads {
                    let kh = h / (heads / kv_heads);
                    let mut scores = vec![0f32; pos + t + 1];
                    for (j, s) in scores.iter_mut().enumerate() {
                        for d in 0..128 {
                            *s += q[(t * heads + h) * 128 + d] * k[(j * kv_heads + kh) * 128 + d];
                        }
                        *s /= 128f32.sqrt();
                    }
                    crate::softmax(&mut scores);
                    for d in 0..128 {
                        let expected: f32 = scores
                            .iter()
                            .enumerate()
                            .map(|(j, p)| p * v[(j * kv_heads + kh) * 128 + d])
                            .sum();
                        let a = actual[(t * heads + h) * 128 + d];
                        ensure!(a.is_finite(), "nonfinite MMA attention output");
                        maximum_error = maximum_error.max((a - expected).abs());
                    }
                }
            }
            eprintln!(
                "MMA n={n} pos={pos} heads={heads}/{kv_heads}: max abs error {maximum_error}"
            );
            ensure!(
                maximum_error < 0.003,
                "MMA attention mismatch: {maximum_error}"
            );
        }
        Ok(())
    }
}
