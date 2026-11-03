//! Batched prefill writes straight into each layer's stable VMM K/V addresses.
use super::{Buffer, CudaSession};
use crate::product;
use anyhow::{ensure, Result};
use cudarc::cublas::sys as blas;

pub(super) struct Scratch {
    capacity: usize,
    ids: Buffer,
    pinned_ids: super::decode::PinnedMemory,
    x: Buffer,
    xb: Buffer,
    residual: Buffer,
    q: Buffer,
    qkv: Buffer,
    hidden: Buffer,
    gateup: Buffer,
    half: Buffer,
}

impl Scratch {
    pub(super) fn capacity(&self) -> usize {
        self.capacity
    }
}

impl CudaSession {
    fn allocate_prefill(&mut self, requested: usize) -> Result<()> {
        let c = &self.config;
        let q = c.head_dim * c.heads;
        let mut free = 0usize;
        let mut total = 0usize;
        // SAFETY: valid outputs; caller has bound and synchronized this context.
        unsafe {
            super::check(
                cudarc::driver::sys::cuMemGetInfo_v2(&mut free, &mut total),
                "query prefill memory budget",
            )?;
        }
        let per_token = product(&[
            2 * c.dim + c.dim.max(q) + 2 * q + 2 * c.kv_heads * c.head_dim + 3 * c.hidden_dim,
            4,
        ])? + product(&[c.dim.max(q).max(c.hidden_dim), 2])?
            + 8;
        let n = requested
            .next_power_of_two()
            .min(self.prefill_limit)
            .min(c.context)
            .min((free / 4 / per_token).max(1));
        if self.prefill.as_ref().is_some_and(|p| p.capacity >= n) {
            return Ok(());
        }
        let matrix = |dim| Buffer::new(&self.ctx, product(&[n, dim])?);
        self.prefill = Some(Scratch {
            capacity: n,
            ids: Buffer::new_bytes(&self.ctx, product(&[n, 8])?)?,
            pinned_ids: super::decode::PinnedMemory::new(self.ctx.clone(), n * 8)?,
            x: matrix(c.dim)?,
            xb: matrix(c.dim.max(q))?,
            residual: matrix(c.dim)?,
            q: matrix(q)?,
            qkv: matrix(q + 2 * c.kv_heads * c.head_dim)?,
            hidden: matrix(c.hidden_dim)?,
            gateup: matrix(2 * c.hidden_dim)?,
            half: Buffer::new_bytes(&self.ctx, product(&[n, c.dim.max(q).max(c.hidden_dim), 2])?)?,
        });
        Ok(())
    }

    /// Consume an entire prompt with matrix-matrix operations and return only
    /// the final token's logits. Supports appending to an existing session.
    pub fn prefill(&mut self, tokens: &[usize]) -> Result<&[f32]> {
        self.prefill_compute(tokens)?;
        self.logits.download(&mut self.host_logits)?;
        ensure!(
            self.host_logits.iter().all(|v| v.is_finite()),
            "non-finite prefill logits"
        );
        Ok(&self.host_logits)
    }
    /// Return the first greedy continuation token with a four-byte transfer.
    pub fn prefill_greedy(&mut self, tokens: &[usize]) -> Result<usize> {
        self.prefill_compute(tokens)?;
        self.greedy_from_logits()
    }
    pub(super) fn prefill_compute(&mut self, tokens: &[usize]) -> Result<()> {
        ensure!(!tokens.is_empty(), "prefill requires at least one token");
        ensure!(
            tokens.iter().all(|&id| id < self.config.vocab_size),
            "token outside vocabulary"
        );
        ensure!(
            tokens.len() <= self.config.context - self.position,
            "context limit reached"
        );
        if self.config.head_dim > 256 {
            for &token in tokens {
                self.forward(token)?;
            }
            return Ok(());
        }
        // All mapping changes precede kernels, including all chunks of a long
        // prompt. No queued work may dereference an address while it is mapped.
        self.ctx.synchronize()?;
        let bytes = product(&[
            self.position + tokens.len(),
            self.config.head_dim,
            self.config.kv_heads,
            4,
        ])?;
        for b in self.keys.iter_mut().chain(&mut self.values) {
            b.ensure_bytes(bytes)?;
        }
        self.allocate_prefill(tokens.len())?;
        self.prepare_attention(tokens.len())?;
        let capacity = self.prefill.as_ref().unwrap().capacity;
        for chunk in tokens.chunks(capacity) {
            // The host ID staging vector is reused; drain the private stream
            // before the synchronous upload overwrites its device buffer.
            self.stream.synchronize()?;
            self.prefill_chunk(chunk)?;
            self.position += chunk.len();
        }
        let p = self.prefill.as_ref().unwrap();
        let last = (tokens.len() - 1) % capacity;
        self.norm(
            self.xb.ptr,
            p.x.ptr + (last * self.config.dim * 4) as u64,
            self.layout.norm.start,
        )?;
        self.matvec(
            self.logits.ptr,
            self.xb.ptr,
            self.layout.classifier.start,
            self.config.dim,
            self.config.vocab_size,
        )?;
        Ok(())
    }

    fn prefill_chunk(&self, tokens: &[usize]) -> Result<()> {
        let p = self.prefill.as_ref().unwrap();
        let c = &self.config;
        let l = &self.layout;
        let n = tokens.len();
        let d = c.dim;
        let q = c.heads * c.head_dim;
        let kv = c.kv_heads * c.head_dim;
        let ids: Vec<u64> = tokens.iter().map(|&id| id as u64).collect();
        // SAFETY: the private stream was drained before this chunk, so its
        // pinned staging allocation is no longer in use. The async copy and
        // embedding kernel share that stream, establishing device-side order.
        unsafe {
            std::ptr::copy_nonoverlapping(ids.as_ptr(), p.pinned_ids.address as *mut u64, n);
            super::check(
                cudarc::driver::sys::cuMemcpyHtoDAsync_v2(
                    p.ids.ptr,
                    p.pinned_ids.address as *const _,
                    n * 8,
                    self.stream.cu_stream(),
                ),
                "upload prompt",
            )?;
        }
        self.launch(
            "embedding_batch",
            &[
                p.x.ptr,
                self.weight(l.embedding.start),
                p.ids.ptr,
                d as u64,
                n as u64,
            ],
            (n * d).div_ceil(256),
        )?;
        for layer in 0..c.layers {
            if layer == 0 {
                self.batch_norm(p.xb.ptr, p.x.ptr, l.att_norm.start, n)?;
            }
            self.gemm(
                p.qkv.ptr,
                p.xb.ptr,
                l.q.start + layer * (q + 2 * kv) * d,
                d,
                q + 2 * kv,
                n,
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
                    p.q.ptr,
                    p.qkv.ptr,
                    l.qkv_bias
                        .as_ref()
                        .map_or(0, |r| self.weight(r.start + layer * (q + 2 * kv))),
                    self.keys[layer].address,
                    self.values[layer].address,
                    qn,
                    kn,
                    c.head_dim as u64,
                    c.heads as u64,
                    c.kv_heads as u64,
                    c.rms_norm_eps.to_bits() as u64,
                    c.rope_theta.to_bits() as u64,
                    self.position as u64,
                ],
                n * (c.heads + 2 * c.kv_heads),
            )?;
            if self.tensor_attention_prefill(layer, n, p.q.ptr, p.qkv.ptr, p.half.ptr)? {
                self.gemm_prepared(
                    p.residual.ptr,
                    p.half.ptr,
                    l.o.start + layer * d * q,
                    q,
                    d,
                    n,
                    true,
                )?;
            } else {
                self.launch_shared(
                    "batch_attention",
                    &[
                        p.xb.ptr,
                        p.q.ptr,
                        self.keys[layer].address,
                        self.values[layer].address,
                        c.head_dim as u64,
                        c.heads as u64,
                        c.kv_heads as u64,
                        self.position as u64,
                    ],
                    n * c.heads,
                    256 * 4,
                )?;
                self.gemm(p.residual.ptr, p.xb.ptr, l.o.start + layer * d * q, q, d, n)?;
            }
            self.launch(
                "add_norm",
                &[
                    p.xb.ptr,
                    p.x.ptr,
                    p.residual.ptr,
                    self.weight(l.ffn_norm.start + layer * d),
                    d as u64,
                    c.rms_norm_eps.to_bits() as u64,
                ],
                n,
            )?;
            let f = d * c.hidden_dim;
            self.gemm(
                p.gateup.ptr,
                p.xb.ptr,
                l.up.start + layer * 3 * f,
                d,
                2 * c.hidden_dim,
                n,
            )?;
            self.launch(
                "swiglu_pair",
                &[p.hidden.ptr, p.gateup.ptr, c.hidden_dim as u64, n as u64],
                (n * c.hidden_dim).div_ceil(256),
            )?;
            self.gemm(
                p.residual.ptr,
                p.hidden.ptr,
                l.down.start + layer * 3 * f,
                c.hidden_dim,
                d,
                n,
            )?;
            if layer + 1 < c.layers {
                self.launch(
                    "add_norm",
                    &[
                        p.xb.ptr,
                        p.x.ptr,
                        p.residual.ptr,
                        self.weight(l.att_norm.start + (layer + 1) * d),
                        d as u64,
                        c.rms_norm_eps.to_bits() as u64,
                    ],
                    n,
                )?;
            } else {
                self.launch(
                    "add",
                    &[p.x.ptr, p.residual.ptr, (n * d) as u64],
                    (n * d).div_ceil(256),
                )?;
            }
        }
        Ok(())
    }

    fn batch_norm(&self, out: u64, input: u64, weight: usize, n: usize) -> Result<()> {
        self.launch(
            "norm_batch",
            &[
                out,
                input,
                self.weight(weight),
                self.config.dim as u64,
                self.config.rms_norm_eps.to_bits() as u64,
            ],
            n,
        )
    }

    fn gemm(
        &self,
        out: u64,
        input: u64,
        weight: usize,
        columns: usize,
        rows: usize,
        tokens: usize,
    ) -> Result<()> {
        let (kind, b) = if self.weight_bytes == 2 {
            let half = self.prefill.as_ref().unwrap().half.ptr;
            self.launch(
                "to_half",
                &[half, input, (columns * tokens) as u64],
                (columns * tokens).div_ceil(256),
            )?;
            (blas::cudaDataType_t::CUDA_R_16F, half)
        } else {
            (blas::cudaDataType_t::CUDA_R_32F, input)
        };
        self.gemm_prepared(
            out,
            b,
            weight,
            columns,
            rows,
            tokens,
            kind == blas::cudaDataType_t::CUDA_R_16F,
        )
    }
    #[allow(clippy::too_many_arguments)]
    fn gemm_prepared(
        &self,
        out: u64,
        b: u64,
        weight: usize,
        columns: usize,
        rows: usize,
        tokens: usize,
        half: bool,
    ) -> Result<()> {
        let kind = if half {
            blas::cudaDataType_t::CUDA_R_16F
        } else {
            blas::cudaDataType_t::CUDA_R_32F
        };
        let alpha = 1f32;
        let beta = 0f32;
        // Row-major weights [rows, columns] are a transposed column-major
        // matrix. Token-major activations are column-major [columns, tokens].
        // SAFETY: private call sites supply disjoint, correctly sized buffers;
        // all allocations stay alive through synchronization on this stream.
        unsafe {
            blas::cublasGemmEx(
                *self.blas.handle(),
                blas::cublasOperation_t::CUBLAS_OP_T,
                blas::cublasOperation_t::CUBLAS_OP_N,
                i32::try_from(rows)?,
                i32::try_from(tokens)?,
                i32::try_from(columns)?,
                (&alpha as *const f32).cast(),
                self.weight(weight) as *const _,
                kind,
                i32::try_from(columns)?,
                b as *const _,
                kind,
                i32::try_from(columns)?,
                (&beta as *const f32).cast(),
                out as *mut _,
                blas::cudaDataType_t::CUDA_R_32F,
                i32::try_from(rows)?,
                blas::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                blas::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            )
            .result()?;
        }
        Ok(())
    }
}
