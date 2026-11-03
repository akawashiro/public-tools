use anyhow::{ensure, Context, Result};
use sha2::{Digest, Sha256};
mod attention;
use clap::Parser;
use cutile::compile_api::KernelCompiler;
use cutile::cutile_compiler::cuda_tile_runtime_utils::{run_tileiras, TileirasOptions};
use std::path::PathBuf;

#[cutile::module]
mod kernels {
    use cutile::core::*;
    #[cutile::entry()]
    fn probe(out: &mut Tensor<f32, { [64] }>) {
        let value: Tile<f32, { [64] }> = constant(7.0f32, shape![64]);
        out.store(value);
    }
}
#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "sm_120")]
    arch: String,
    #[arg(long, default_value = "target/cutile")]
    output: PathBuf,
    #[arg(long)]
    probe: bool,
    #[arg(long, default_value="tf32", value_parser=["tf32","f16"])]
    mma_type: String,
    #[arg(long, default_value_t = 16)]
    heads: i32,
    #[arg(long, default_value_t = 8)]
    kv_heads: i32,
    #[arg(long, default_value_t = 128)]
    head_dim: i32,
    #[arg(long, default_value_t = 32)]
    query_tile: i32,
    #[arg(long, default_value_t = 64)]
    key_tile: i32,
}
fn main() -> Result<()> {
    let a = Args::parse();
    std::fs::create_dir_all(&a.output)?;
    let (stem, compiled) = if a.probe {
        let compiled = KernelCompiler::new(kernels::__module_ast_self, "kernels", "probe")
            .strides(&[("out", &[1])])
            .target(&a.arch)
            .compile()?;
        ("probe".to_owned(), compiled)
    } else {
        ensure!(
            a.heads > 0 && a.kv_heads > 0 && a.heads % a.kv_heads == 0,
            "invalid GQA heads"
        );
        ensure!(
            [32, 64, 128, 256].contains(&a.head_dim),
            "unsupported head dimension"
        );
        ensure!(
            [16, 32, 64, 128].contains(&a.query_tile) && [32, 64, 128, 256].contains(&a.key_tile),
            "invalid tiles"
        );
        let qstride = [
            a.head_dim,
            a.heads
                .checked_mul(a.head_dim)
                .context("Q stride exceeds i32")?,
            1,
        ];
        let kvstride = [
            a.head_dim,
            a.kv_heads
                .checked_mul(a.head_dim)
                .context("KV stride exceeds i32")?,
            1,
        ];
        let stem = format!(
            "attention-h{}-kv{}-d{}-{}",
            a.heads, a.kv_heads, a.head_dim, a.arch
        );
        let compiled = KernelCompiler::new(
            attention::vattention_tiles::__module_ast_self,
            "vattention_tiles",
            "vmm_attention",
        )
        .generics(vec![
            a.mma_type.clone(),
            a.query_tile.to_string(),
            a.key_tile.to_string(),
            a.head_dim.to_string(),
        ])
        .strides(&[
            ("out", &qstride),
            ("q", &qstride),
            ("k", &kvstride),
            ("v", &kvstride),
        ])
        .target(&a.arch)
        .compile()?;
        (stem, compiled)
    };
    let base = a.output.join(&stem);
    std::fs::write(base.with_extension("tileir"), compiled.ir_text())?;
    println!("launch checks: {:?}", compiled.launch_checks());
    let cubin = run_tileiras(&compiled.bytecode()?, &a.arch, &TileirasOptions::default())?;
    std::fs::write(base.with_extension("cubin"), &cubin)?;
    if !a.probe {
        std::fs::write(
            base.with_extension("json"),
            serde_json::to_vec_pretty(&serde_json::json!({
                "abi_version":1, "cubin_sha256":format!("{:x}",Sha256::digest(&cubin)),
                "kernel_source_sha256":format!("{:x}",Sha256::digest(include_bytes!("attention.rs"))),
                "launch_checks":format!("{:?}",compiled.launch_checks()), "entry":"vmm_attention_entry", "arch":a.arch,
                "heads":a.heads,"kv_heads":a.kv_heads,"head_dim":a.head_dim,
                "query_tile":a.query_tile,"key_tile":a.key_tile,
                "cubin":format!("{stem}.cubin"),"cutile_revision":"cced2a35bcc557b30f27209ff3887b4eb728cb63",
                "mma_type":a.mma_type,"precision":"fp32 VMM cache; fp32 accumulation; fp16 output"
            }))?,
        )?;
    }
    println!("wrote {} cubin bytes to {}", cubin.len(), base.display());
    Ok(())
}
