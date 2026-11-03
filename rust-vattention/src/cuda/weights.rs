//! GPU-only packing: Q/K/V and gate/up projections become single GEMMs/GEMVs.
//! CPU checkpoints and their public layout are unchanged; no duplicate weights.
use super::{check, Buffer};
use crate::{
    model::{Layout, Model},
    product,
};
use anyhow::{ensure, Result};
use cudarc::driver::{sys, CudaContext};
use std::{ops::Range, sync::Arc};

pub(super) fn upload(
    ctx: &Arc<CudaContext>,
    model: &Model,
    fp16: bool,
) -> Result<(Buffer, Layout)> {
    let c = &model.config;
    let l = &model.layout;
    ensure!(
        l.q.end == l.k.start && l.k.end == l.v.start,
        "QKV weight regions must be adjacent"
    );
    ensure!(
        l.up.end == l.down.start && l.down.end == l.gate.start,
        "MLP weight regions must be adjacent"
    );
    let bytes = if fp16 { 2 } else { 4 };
    let out = Buffer::new_bytes(ctx, product(&[model.weights.len(), bytes])?)?;
    let copy = |source: Range<usize>, destination: usize| -> Result<()> {
        for (chunk_index, chunk) in model.weights[source].chunks(1_048_576).enumerate() {
            let address = out.ptr + ((destination + chunk_index * 1_048_576) * bytes) as u64;
            // Pageable copies may stage asynchronously on the legacy stream.
            // They are ordered there and all finish before upload returns.
            if fp16 {
                let half: Vec<_> = chunk.iter().map(|&v| half::f16::from_f32(v)).collect();
                ensure!(
                    half.iter().all(|v| v.is_finite()),
                    "weights overflow FP16; use --weight-dtype fp32"
                );
                // SAFETY: source is valid through host staging and destination
                // lies inside the exclusively owned weight allocation.
                unsafe {
                    check(
                        sys::cuMemcpyHtoD_v2(address, half.as_ptr().cast(), half.len() * 2),
                        "upload packed weights",
                    )?;
                }
            } else {
                // SAFETY: same bounds and lifetime as above, FP32 storage.
                unsafe {
                    check(
                        sys::cuMemcpyHtoD_v2(address, chunk.as_ptr().cast(), chunk.len() * 4),
                        "upload packed weights",
                    )?;
                }
            }
        }
        Ok(())
    };
    for range in [
        0..l.q.start,
        l.v.end..l.up.start,
        l.gate.end..model.weights.len(),
    ] {
        copy(range.clone(), range.start)?;
    }
    let d = c.dim;
    let q = c.heads * c.head_dim;
    let kv = c.kv_heads * c.head_dim;
    let qkv_stride = (q + 2 * kv) * d;
    let f = d * c.hidden_dim;
    for layer in 0..c.layers {
        for (src, size, offset) in [
            (&l.q, q * d, 0),
            (&l.k, kv * d, q * d),
            (&l.v, kv * d, (q + kv) * d),
        ] {
            let start = src.start + layer * size;
            copy(start..start + size, l.q.start + layer * qkv_stride + offset)?;
        }
        for (src, offset) in [(&l.up, 0), (&l.gate, f), (&l.down, 2 * f)] {
            let start = src.start + layer * f;
            copy(start..start + f, l.up.start + layer * 3 * f + offset)?;
        }
    }
    ctx.synchronize()?;
    let mut packed = l.clone();
    packed.q = l.q.start..l.q.start + q * d;
    packed.k = packed.q.end..packed.q.end + kv * d;
    packed.v = packed.k.end..packed.k.end + kv * d;
    packed.up = l.up.start..l.up.start + f;
    packed.gate = packed.up.end..packed.up.end + f;
    packed.down = packed.gate.end..packed.gate.end + f;
    Ok((out, packed))
}
