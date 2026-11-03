//! Stage timing without HTTP. Run separately from API benchmark measurements.
use anyhow::Result;
use clap::Parser;
use rust_vattention::{cuda::CudaSession, hub};
use std::{path::PathBuf, time::Instant};
#[derive(Parser)]
struct Args {
    model: String,
    #[arg(long, default_value_t = 128)]
    tokens: usize,
    #[arg(long, default_value_t = 64)]
    steps: usize,
    #[arg(long, default_value_t = 3)]
    repeats: usize,
    #[arg(long)]
    greedy: bool,
}
fn main() -> Result<()> {
    let a = Args::parse();
    let mut loaded = hub::load(&a.model, "main", &PathBuf::from("models/hub"))?;
    loaded
        .model
        .limit_context((a.tokens + a.steps + 1).max(512))?;
    let mut gpu = CudaSession::new_with_fp16(&loaded.model, 0, 65536, true)?;
    let ids: Vec<_> = (0..a.tokens)
        .map(|i| (i * 773 + 17) % loaded.model.config().vocab_size)
        .collect();
    if a.greedy {
        gpu.prefill_greedy(&ids)?;
        gpu.forward_greedy(1)?;
    } else {
        gpu.prefill(&ids)?;
        gpu.forward(1)?;
    }
    for i in 0..a.repeats {
        gpu.reset();
        let start = Instant::now();
        if a.greedy {
            gpu.prefill_greedy(&ids)?;
        } else {
            gpu.prefill(&ids)?;
        }
        let prefill = start.elapsed().as_secs_f64();
        let start = Instant::now();
        for j in 0..a.steps {
            let token = (j * 97 + 21) % loaded.model.config().vocab_size;
            if a.greedy {
                gpu.forward_greedy(token)?;
            } else {
                gpu.forward(token)?;
            }
        }
        let decode = start.elapsed().as_secs_f64();
        println!(
            "run={i} prefill_ms={:.3} decode_ms_per_token={:.3} decode_tok_s={:.2}",
            prefill * 1000.,
            decode * 1000. / a.steps as f64,
            a.steps as f64 / decode
        );
    }
    eprintln!("cache={:?}", gpu.cache_stats());
    Ok(())
}
