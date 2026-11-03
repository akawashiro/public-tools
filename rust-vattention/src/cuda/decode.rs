//! Length-bucketed decode graphs; token/position are dynamic device scalars.
//! VMM mapping changes stay outside capture/replay and all KV pointers are stable.
use super::CudaSession;
use anyhow::{ensure, Context, Result};
use cudarc::driver::{sys, CudaGraph};

pub(super) struct DecodeGraph(CudaGraph);
impl DecodeGraph {
    pub(super) fn new(graph: CudaGraph) -> Self {
        Self(graph)
    }
    pub(super) fn launch(&self) -> Result<()> {
        self.0.launch()?;
        Ok(())
    }
}
// SAFETY: CUDA graphs may move between host threads but must not be accessed
// concurrently. This graph is private, exclusively owned by CudaSession, whose
// execution requires &mut self. Server requests additionally hold a Mutex.
// Every operation binds the owning CUDA context; session Drop synchronizes
// before graph destruction, including when a worker thread has changed.
unsafe impl Send for DecodeGraph {}

// CUDA-owned pinned host memory keeps its address across worker-thread moves.
// The integer address is never exposed; reads occur only after stream completion.
pub(super) struct PinnedMemory {
    ctx: std::sync::Arc<cudarc::driver::CudaContext>,
    pub(super) address: usize,
}
impl PinnedMemory {
    pub(super) fn new(
        ctx: std::sync::Arc<cudarc::driver::CudaContext>,
        bytes: usize,
    ) -> Result<Self> {
        ctx.bind_to_thread()?;
        let mut address = std::ptr::null_mut();
        // SAFETY: valid output; exactly `bytes` bytes are owned until Drop.
        unsafe {
            super::check(
                sys::cuMemHostAlloc(&mut address, bytes, 0),
                "allocate pinned token",
            )?;
        }
        Ok(Self {
            ctx,
            address: address as usize,
        })
    }
}
impl Drop for PinnedMemory {
    fn drop(&mut self) {
        if self.ctx.synchronize().is_ok() {
            // SAFETY: no DMA or kernels can still use this pinned allocation.
            unsafe {
                let _ = sys::cuMemFreeHost(self.address as *mut _);
            }
        }
    }
}

impl CudaSession {
    pub fn forward(&mut self, token: usize) -> Result<&[f32]> {
        self.decode(token)?;
        self.logits.download(&mut self.host_logits)?;
        ensure!(
            self.host_logits.iter().all(|v| v.is_finite()),
            "non-finite CUDA logits"
        );
        Ok(&self.host_logits)
    }
    /// Greedy decoding without transferring the vocabulary-sized logit array.
    pub fn forward_greedy(&mut self, token: usize) -> Result<usize> {
        self.decode(token)?;
        self.greedy_from_logits()
    }
    pub(super) fn decode(&mut self, token: usize) -> Result<()> {
        if self.config.head_dim > 256 {
            self.forward_legacy(token)?;
            return Ok(());
        }
        ensure!(token < self.config.vocab_size, "token outside vocabulary");
        ensure!(self.position < self.config.context, "context limit reached");
        let bytes = (self.position + 1) * self.config.head_dim * self.config.kv_heads * 4;
        for b in self.keys.iter_mut().chain(&mut self.values) {
            b.ensure_bytes(bytes)?;
        }
        self.launch("set_step", &[token as u64, self.position as u64], 1)?;
        let segments = (self.position + 1).div_ceil(256).next_power_of_two();
        if self.use_graphs {
            if !self.decode_graphs.contains_key(&segments) {
                self.stream
                    .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
                let result = self.decode_body(segments);
                let graph = self.stream.end_capture(sys::CUgraphInstantiate_flags(0));
                result?;
                self.decode_graphs
                    .insert(segments, DecodeGraph(graph?.context("empty decode graph")?));
            }
            self.decode_graphs[&segments].0.launch()?;
        } else {
            self.decode_body(segments)?;
        }
        self.position += 1;
        Ok(())
    }
    pub(super) fn greedy_from_logits(&mut self) -> Result<usize> {
        if self.use_graphs {
            if self.greedy_graph.is_none() {
                self.stream
                    .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
                let result = self.greedy_body();
                let graph = self.stream.end_capture(sys::CUgraphInstantiate_flags(0));
                result?;
                self.greedy_graph = Some(DecodeGraph(graph?.context("empty greedy graph")?));
            }
            self.greedy_graph.as_ref().unwrap().0.launch()?;
        } else {
            self.greedy_body()?;
        }
        self.finish_sample()
    }
    pub(super) fn finish_sample(&self) -> Result<usize> {
        self.stream.synchronize()?;
        // SAFETY: the stream's final DMA completed into the pinned allocation.
        let token = unsafe { (self.pinned_token.address as *const u32).read_volatile() };
        ensure!(
            (token as usize) < self.config.vocab_size,
            "non-finite CUDA logits"
        );
        Ok(token as usize)
    }
    fn greedy_body(&self) -> Result<()> {
        self.argmax_device()?;
        self.copy_token_async()
    }
    pub(super) fn argmax_device(&self) -> Result<()> {
        let count = self.config.vocab_size;
        let blocks = count.div_ceil(1024);
        self.launch(
            "argmax",
            &[
                self.logits.ptr,
                0,
                count as u64,
                1024,
                self.argmax_values.ptr,
                self.argmax_ids.ptr,
            ],
            blocks,
        )?;
        self.launch(
            "argmax",
            &[
                self.argmax_values.ptr,
                self.argmax_ids.ptr,
                blocks as u64,
                blocks as u64,
                self.residual.ptr,
                self.next_token.ptr,
            ],
            1,
        )?;
        Ok(())
    }
    pub(super) fn copy_token_async(&self) -> Result<()> {
        // SAFETY: source and pinned destination contain one u32, live for the
        // entire session, and the copy is ordered after argmax on the same stream.
        unsafe {
            super::check(
                sys::cuMemcpyDtoHAsync_v2(
                    self.pinned_token.address as *mut _,
                    self.next_token.ptr,
                    4,
                    self.stream.cu_stream(),
                ),
                "schedule greedy token download",
            )?;
        }
        Ok(())
    }

    fn decode_body(&self, segments: usize) -> Result<()> {
        let c = &self.config;
        let l = &self.layout;
        let d = c.dim;
        let q = c.heads * c.head_dim;
        let kv = c.kv_heads * c.head_dim;
        self.launch(
            "embedding_step",
            &[self.x.ptr, self.weight(l.embedding.start), d as u64],
            d.div_ceil(256),
        )?;
        for layer in 0..c.layers {
            let key = self.keys[layer].address;
            let value = self.values[layer].address;
            if layer == 0 {
                self.norm(self.xb.ptr, self.x.ptr, l.att_norm.start)?;
            }
            self.matvec(
                self.qkv.ptr,
                self.xb.ptr,
                l.q.start + layer * (q + 2 * kv) * d,
                d,
                q + 2 * kv,
            )?;
            let qn = l
                .q_norm
                .as_ref()
                .map_or(0, |r| self.weight(r.start + layer * c.head_dim));
            let kn = l
                .k_norm
                .as_ref()
                .map_or(0, |r| self.weight(r.start + layer * c.head_dim));
            self.launch(
                "qkv_transform",
                &[
                    self.q.ptr,
                    self.qkv.ptr,
                    l.qkv_bias
                        .as_ref()
                        .map_or(0, |r| self.weight(r.start + layer * (q + 2 * kv))),
                    key,
                    value,
                    qn,
                    kn,
                    c.head_dim as u64,
                    c.heads as u64,
                    c.kv_heads as u64,
                    c.rms_norm_eps.to_bits() as u64,
                    c.rope_theta.to_bits() as u64,
                    u64::MAX,
                ],
                c.heads + 2 * c.kv_heads,
            )?;
            if segments == 1 {
                self.launch_shared(
                    "batch_attention",
                    &[
                        self.xb.ptr,
                        self.q.ptr,
                        key,
                        value,
                        c.head_dim as u64,
                        c.heads as u64,
                        c.kv_heads as u64,
                        u64::MAX,
                    ],
                    c.heads,
                    256 * 4,
                )?;
            } else {
                self.launch(
                    "decode_attention_partial",
                    &[
                        self.attention_partials.ptr,
                        self.q.ptr,
                        key,
                        value,
                        c.head_dim as u64,
                        c.heads as u64,
                        c.kv_heads as u64,
                        self.attention_segments as u64,
                    ],
                    segments * c.heads,
                )?;
                self.launch_shared(
                    "decode_attention_merge",
                    &[
                        self.xb.ptr,
                        self.attention_partials.ptr,
                        c.head_dim as u64,
                        self.attention_segments as u64,
                    ],
                    c.heads,
                    segments * 4,
                )?;
            }
            self.matvec(
                self.residual.ptr,
                self.xb.ptr,
                l.o.start + layer * d * q,
                q,
                d,
            )?;
            self.launch(
                "add_norm",
                &[
                    self.xb.ptr,
                    self.x.ptr,
                    self.residual.ptr,
                    self.weight(l.ffn_norm.start + layer * d),
                    d as u64,
                    c.rms_norm_eps.to_bits() as u64,
                ],
                1,
            )?;
            let f = d * c.hidden_dim;
            self.matvec(
                self.gateup.ptr,
                self.xb.ptr,
                l.up.start + layer * 3 * f,
                d,
                2 * c.hidden_dim,
            )?;
            self.launch(
                "swiglu_pair",
                &[self.hidden.ptr, self.gateup.ptr, c.hidden_dim as u64, 1],
                c.hidden_dim.div_ceil(256),
            )?;
            self.matvec(
                self.residual.ptr,
                self.hidden.ptr,
                l.down.start + layer * 3 * f,
                c.hidden_dim,
                d,
            )?;
            let norm = if layer + 1 < c.layers {
                l.att_norm.start + (layer + 1) * d
            } else {
                l.norm.start
            };
            self.launch(
                "add_norm",
                &[
                    self.xb.ptr,
                    self.x.ptr,
                    self.residual.ptr,
                    self.weight(norm),
                    d as u64,
                    c.rms_norm_eps.to_bits() as u64,
                ],
                1,
            )?;
        }
        self.matvec(
            self.logits.ptr,
            self.xb.ptr,
            l.classifier.start,
            d,
            c.vocab_size,
        )
    }
}
