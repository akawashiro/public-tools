//! Batched Tensor Core attention with direct strided reads of FP32 VMM K/V.
//! No K/V packing/copy and no application block table.
use super::{Buffer, CudaSession};
use crate::product;
use anyhow::Result;
use cudarc::cublas::sys as blas;

pub(super) struct Attention {
    queries: usize,
    keys: usize,
    scores: Buffer,
    probs: Buffer,
}
#[derive(Clone, Copy)]
struct Matrix {
    ptr: u64,
    leading: usize,
    stride: usize,
}
impl CudaSession {
    pub(super) fn prepare_attention(&mut self, count: usize) -> Result<()> {
        if (self.mma_attention && self.position + count <= self.mma_limit)
            || !self.tensor_attention
            || count < 16
            || self.config.heads / self.config.kv_heads > 8
            || self.position >= 16384
        {
            return Ok(());
        }
        let keys = (self.position + count)
            .next_power_of_two()
            .min(16384)
            .min(self.config.context);
        let mut queries = self.prefill.as_ref().unwrap().capacity().min(128);
        if self
            .attention
            .as_ref()
            .is_some_and(|a| a.queries >= queries && a.keys >= keys)
        {
            return Ok(());
        }
        let mut free = 0usize;
        let mut total = 0usize;
        // SAFETY: valid outputs; prefill synchronized before allocating scratch.
        unsafe {
            super::check(
                cudarc::driver::sys::cuMemGetInfo_v2(&mut free, &mut total),
                "query attention memory",
            )?;
        }
        while queries >= 16 && product(&[self.config.heads, queries, keys, 8])? > free / 3 {
            queries /= 2;
        }
        if queries < 16 {
            return Ok(());
        }
        let size = product(&[self.config.heads, queries, keys])?;
        self.attention = Some(Attention {
            queries,
            keys,
            scores: Buffer::new(&self.ctx, size)?,
            probs: Buffer::new(&self.ctx, size)?,
        });
        Ok(())
    }
    pub(super) fn tensor_attention_prefill(
        &self,
        layer: usize,
        n: usize,
        q: u64,
        head_output: u64,
        half_output: u64,
    ) -> Result<bool> {
        let c = &self.config;
        let length = self.position + n;
        if self.mma_attention && length <= self.mma_limit && n >= 16 {
            self.launch_threads(
                "mma_causal_attention",
                &[
                    half_output,
                    q,
                    self.keys[layer].address,
                    self.values[layer].address,
                    n as u64,
                    c.heads as u64,
                    c.kv_heads as u64,
                    self.position as u64,
                ],
                n.div_ceil(16) * c.heads,
                0,
                128,
            )?;
            return Ok(true);
        }
        let Some(a) = &self.attention else {
            return Ok(false);
        };
        if n < 16 || length > a.keys {
            return Ok(false);
        }
        let groups = c.heads / c.kv_heads;
        let head = c.head_dim;
        let qdim = c.heads * head;
        let kv = c.kv_heads * head;
        for start in (0..n).step_by(a.queries) {
            let count = a.queries.min(n - start);
            for group in 0..groups {
                self.attention_mm(
                    [length, count, head],
                    c.kv_heads,
                    (head as f32).sqrt().recip(),
                    true,
                    Matrix {
                        ptr: self.keys[layer].address,
                        leading: kv,
                        stride: head,
                    },
                    Matrix {
                        ptr: q + ((group * head + start * qdim) * 4) as u64,
                        leading: qdim,
                        stride: groups * head,
                    },
                    Matrix {
                        ptr: a.scores.ptr + (group * a.queries * length * 4) as u64,
                        leading: length,
                        stride: groups * a.queries * length,
                    },
                )?;
            }
            self.launch(
                "causal_probabilities",
                &[
                    a.scores.ptr,
                    a.probs.ptr,
                    length as u64,
                    a.queries as u64,
                    count as u64,
                    start as u64,
                    self.position as u64,
                ],
                c.heads * count,
            )?;
            for group in 0..groups {
                self.attention_mm(
                    [head, count, length],
                    c.kv_heads,
                    1.,
                    false,
                    Matrix {
                        ptr: self.values[layer].address,
                        leading: kv,
                        stride: head,
                    },
                    Matrix {
                        ptr: a.probs.ptr + (group * a.queries * length * 4) as u64,
                        leading: length,
                        stride: groups * a.queries * length,
                    },
                    Matrix {
                        ptr: head_output + ((group * n + start) * head * 4) as u64,
                        leading: head,
                        stride: groups * n * head,
                    },
                )?;
            }
        }
        self.launch(
            "attention_output_half",
            &[
                half_output,
                head_output,
                head as u64,
                c.heads as u64,
                n as u64,
            ],
            (n * qdim).div_ceil(256),
        )?;
        Ok(true)
    }
    #[allow(clippy::too_many_arguments)]
    fn attention_mm(
        &self,
        dims: [usize; 3],
        batches: usize,
        alpha: f32,
        transpose_a: bool,
        a: Matrix,
        b: Matrix,
        out: Matrix,
    ) -> Result<()> {
        let beta = 0f32;
        // SAFETY: matrices are subviews of live, mapped VMM prefixes or owned
        // scratch. Input batches may interleave, but output matrices are disjoint.
        // All calls and consumers are ordered on the session's private stream.
        unsafe {
            blas::cublasGemmStridedBatchedEx(
                *self.blas.handle(),
                if transpose_a {
                    blas::cublasOperation_t::CUBLAS_OP_T
                } else {
                    blas::cublasOperation_t::CUBLAS_OP_N
                },
                blas::cublasOperation_t::CUBLAS_OP_N,
                i32::try_from(dims[0])?,
                i32::try_from(dims[1])?,
                i32::try_from(dims[2])?,
                (&alpha as *const f32).cast(),
                a.ptr as *const _,
                blas::cudaDataType_t::CUDA_R_32F,
                i32::try_from(a.leading)?,
                i64::try_from(a.stride)?,
                b.ptr as *const _,
                blas::cudaDataType_t::CUDA_R_32F,
                i32::try_from(b.leading)?,
                i64::try_from(b.stride)?,
                (&beta as *const f32).cast(),
                out.ptr as *mut _,
                blas::cudaDataType_t::CUDA_R_32F,
                i32::try_from(out.leading)?,
                i64::try_from(out.stride)?,
                i32::try_from(batches)?,
                blas::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32,
                blas::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            )
            .result()?;
        }
        Ok(())
    }
}
