//! Experimental, ahead-of-time cuTile cubin loaded by the existing CUDA driver.
//! No additional CUDA context, allocator, stream, or KV copy is introduced.
use super::{Config, CudaContext, CudaSession, LaunchConfig, PushKernelArg};
use anyhow::{ensure, Context, Result};
use cudarc::{driver::CudaFunction, nvrtc::Ptx};
use serde::Deserialize;
use std::{
    path::{Component, Path},
    sync::Arc,
};

#[derive(Deserialize)]
struct Manifest {
    abi_version: u32,
    arch: String,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    query_tile: usize,
    key_tile: usize,
    cubin: String,
    entry: String,
}
pub(super) struct Attention {
    function: CudaFunction,
    query_tile: usize,
}
impl Attention {
    pub(super) fn load(ctx: &Arc<CudaContext>, c: &Config, directory: &Path) -> Result<Self> {
        let (major, minor) = ctx.compute_capability()?;
        let arch = format!("sm_{major}{minor}");
        let file = directory.join(format!(
            "attention-h{}-kv{}-d{}-{arch}.json",
            c.heads, c.kv_heads, c.head_dim
        ));
        let m: Manifest = serde_json::from_slice(&std::fs::read(&file).with_context(|| {
            format!(
                "load cuTile manifest {}; compile this model's head layout first",
                file.display()
            )
        })?)?;
        ensure!(
            m.abi_version == 1
                && m.arch == arch
                && m.heads == c.heads
                && m.kv_heads == c.kv_heads
                && m.head_dim == c.head_dim,
            "cuTile manifest/model mismatch"
        );
        ensure!(
            [16, 32, 64, 128].contains(&m.query_tile) && [32, 64, 128, 256].contains(&m.key_tile),
            "invalid cuTile tile dimensions"
        );
        ensure!(
            m.entry == "vmm_attention_entry"
                && Path::new(&m.cubin).components().count() == 1
                && matches!(
                    Path::new(&m.cubin).components().next(),
                    Some(Component::Normal(_))
                ),
            "invalid cuTile artifact path/entry"
        );
        let module = ctx.load_module(Ptx::from_binary(std::fs::read(directory.join(m.cubin))?))?;
        let function = module.load_function(&m.entry)?;
        eprintln!(
            "cuTile attention: {arch}, {}x{} tiles; direct FP32 VMM K/V",
            m.query_tile, m.key_tile
        );
        Ok(Self {
            function,
            query_tile: m.query_tile,
        })
    }
}
enum Arg {
    Pointer(u64),
    Int(i32),
    Float(f32),
}
fn tensor(
    args: &mut Vec<Arg>,
    ptr: u64,
    shape: [i32; 3],
    strides: [i32; 3],
    partition: Option<[i32; 3]>,
) {
    args.push(Arg::Pointer(ptr));
    args.extend(shape.into_iter().map(Arg::Int));
    args.extend(strides.into_iter().map(Arg::Int));
    if let Some(p) = partition {
        args.extend(p.into_iter().map(Arg::Int));
        args.extend(strides.into_iter().map(Arg::Int));
    }
}
impl CudaSession {
    pub(super) fn cutile_prefill(
        &self,
        layer: usize,
        n: usize,
        q: u64,
        output: u64,
    ) -> Result<bool> {
        let Some(a) = &self.cutile_attention else {
            return Ok(false);
        };
        let c = &self.config;
        let cv = |v| i32::try_from(v).context("cuTile dimension exceeds i32");
        let (h, kh, d, n, len, pos, bm) = (
            cv(c.heads)?,
            cv(c.kv_heads)?,
            cv(c.head_dim)?,
            cv(n)?,
            cv(self.position + n)?,
            cv(self.position)?,
            cv(a.query_tile)?,
        );
        let qstride = [d, h.checked_mul(d).context("cuTile Q stride overflow")?, 1];
        let kvstride = [
            d,
            kh.checked_mul(d).context("cuTile KV stride overflow")?,
            1,
        ];
        let mut args = Vec::with_capacity(37);
        tensor(&mut args, output, [h, n, d], qstride, Some([1, bm, d]));
        tensor(&mut args, q, [h, n, d], qstride, None);
        tensor(
            &mut args,
            self.keys[layer].address,
            [kh, len, d],
            kvstride,
            None,
        );
        tensor(
            &mut args,
            self.values[layer].address,
            [kh, len, d],
            kvstride,
            None,
        );
        args.push(Arg::Float((d as f32).sqrt().recip()));
        args.push(Arg::Int(pos));
        args.push(Arg::Int(cv(c.sliding_window.unwrap_or(0))?));
        self.ctx.bind_to_thread()?;
        let mut builder = self.stream.launch_builder(&a.function);
        for arg in &args {
            match arg {
                Arg::Pointer(v) => {
                    builder.arg(v);
                }
                Arg::Int(v) => {
                    builder.arg(v);
                }
                Arg::Float(v) => {
                    builder.arg(v);
                }
            }
        }
        // SAFETY: the pinned compiler ABI is ptr + rank-3 dims/strides, with
        // partition dims/strides on the mutable output, then f32/i32 scalars.
        // Host validates exact model strides, positive shapes, disjoint output
        // partitions and a grid of (heads, ceil(n/BM), 1). K/V prefixes are mapped
        // before prefill; all producers and consumers share this private stream.
        // FP16 output scratch and FP32 Q/K/V allocations outlive queued work.
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (h as u32, (n as u32).div_ceil(bm as u32), 1),
                block_dim: (1, 1, 1),
                shared_mem_bytes: 0,
            })?;
        }
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        cuda::{check, Buffer},
        hub,
    };
    use cudarc::driver::sys;
    fn session() -> Result<CudaSession> {
        let mut loaded = hub::load(
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/mistral-window-f16"
            ),
            "main",
            Path::new("unused"),
        )?;
        loaded.model.config.context = 8192;
        let mut gpu = CudaSession::new_with_fp16(&loaded.model, 0, 65536, true)?;
        let dir = std::env::var_os("RUST_VATTENTION_CUTILE_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| "target/cutile".into());
        gpu.cutile_attention = Some(Attention::load(&gpu.ctx, &gpu.config, &dir)?);
        Ok(gpu)
    }
    #[test]
    #[ignore = "requires CUDA and compiled cuTile artifacts for heads=2, kv_heads=1, head_dim=128"]
    fn direct_vmm_attention_matches_dense_reference() -> Result<()> {
        let mut gpu = session()?;
        let addresses = (gpu.keys[0].address, gpu.values[0].address);
        for (n, pos, window) in [
            (17usize, 0usize, None),
            (63, 11, None),
            (128, 3, Some(5)),
            (273, 0, None),
            (31, 257, Some(17)),
            (17, 4093, Some(5)),
        ] {
            gpu.reset_and_release()?;
            gpu.position = pos;
            gpu.config.sliding_window = window;
            let length = pos + n;
            let q: Vec<f32> = (0..n * 2 * 128)
                .map(|i| ((i * 37 % 1031) as f32 - 515.) / 512.)
                .collect();
            let k: Vec<f32> = (0..length * 128)
                .map(|i| ((i * 53 % 2003) as f32 - 1001.) / 1024.)
                .collect();
            let v: Vec<f32> = (0..length * 128)
                .map(|i| ((i * 17 % 601) as f32 - 300.) / 128.)
                .collect();
            let qb = Buffer::new(&gpu.ctx, q.len())?;
            qb.upload(&q)?;
            gpu.keys[0].ensure_bytes(k.len() * 4)?;
            gpu.values[0].ensure_bytes(v.len() * 4)?;
            assert_eq!(addresses, (gpu.keys[0].address, gpu.values[0].address));
            // SAFETY: the mapped prefixes cover these uploads; context sync below
            // orders pageable staging before the private stream reads the cache.
            unsafe {
                check(
                    sys::cuMemcpyHtoD_v2(gpu.keys[0].address, k.as_ptr().cast(), k.len() * 4),
                    "upload cuTile reference K",
                )?;
                check(
                    sys::cuMemcpyHtoD_v2(gpu.values[0].address, v.as_ptr().cast(), v.len() * 4),
                    "upload cuTile reference V",
                )?;
            }
            gpu.ctx.synchronize()?;
            let out = Buffer::new_bytes(&gpu.ctx, q.len() * 2)?;
            assert!(gpu.cutile_prefill(0, n, qb.ptr, out.ptr)?);
            gpu.stream.synchronize()?;
            let mut bits = vec![0u16; q.len()];
            // SAFETY: output completed, and the host destination spans all FP16 elements.
            unsafe {
                check(
                    sys::cuMemcpyDtoH_v2(bits.as_mut_ptr().cast(), out.ptr, bits.len() * 2),
                    "read cuTile attention",
                )?;
            }
            let mut max_error = 0f64;
            for t in 0..n {
                for h in 0..2 {
                    let end = pos + t + 1;
                    let first = window.map_or(0, |w| end.saturating_sub(w));
                    let mut scores: Vec<f64> = (first..end)
                        .map(|j| {
                            (0..128)
                                .map(|d| q[(t * 2 + h) * 128 + d] as f64 * k[j * 128 + d] as f64)
                                .sum::<f64>()
                                / 128f64.sqrt()
                        })
                        .collect();
                    let m = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    for p in &mut scores {
                        *p = (*p - m).exp();
                    }
                    let sum: f64 = scores.iter().sum();
                    for d in 0..128 {
                        let expected: f64 = scores
                            .iter()
                            .enumerate()
                            .map(|(j, p)| p / sum * v[(first + j) * 128 + d] as f64)
                            .sum();
                        let actual = half::f16::from_bits(bits[(t * 2 + h) * 128 + d]).to_f32();
                        ensure!(actual.is_finite(), "nonfinite cuTile output");
                        max_error = max_error.max((actual as f64 - expected).abs());
                    }
                }
            }
            eprintln!("cuTile n={n} pos={pos} window={window:?} max_error={max_error}");
            ensure!(
                max_error < 0.003,
                "cuTile dense reference mismatch: {max_error}"
            );
        }
        Ok(())
    }
    #[test]
    #[ignore = "requires CUDA and compiled cuTile artifacts for the Mistral fixture"]
    fn prefill_append_chunking_and_decode_match_transformers() -> Result<()> {
        let mut gpu = session()?;
        let data: serde_json::Value = serde_json::from_slice(&std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/mistral-window-f16/expected.json"
        ))?)?;
        let tokens: Vec<usize> = data["tokens"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as usize)
            .collect();
        let compare = |actual: &[f32], position: usize| -> Result<()> {
            let expected = data["logits"][position].as_array().unwrap();
            let error = actual
                .iter()
                .zip(expected)
                .map(|(a, b)| (*a - b.as_f64().unwrap() as f32).abs())
                .fold(0f32, f32::max);
            ensure!(
                error < 0.0005,
                "cuTile model mismatch at {position}: {error}"
            );
            Ok(())
        };
        for n in [17, 128, 256, 257, 273] {
            gpu.reset_and_release()?;
            compare(gpu.prefill(&tokens[..n])?, n - 1)?;
            for (i, &token) in tokens.iter().enumerate().skip(n).take(3) {
                compare(gpu.forward(token)?, i)?;
            }
        }
        gpu.reset_and_release()?;
        gpu.prefill(&tokens[..17])?;
        compare(gpu.prefill(&tokens[17..])?, tokens.len() - 1)?;
        gpu.reset_and_release()?;
        gpu.prefill_limit = 64;
        compare(gpu.prefill(&tokens)?, tokens.len() - 1)?;
        Ok(())
    }
}
