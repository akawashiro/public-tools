use anyhow::{ensure, Result};
use clap::{Parser, ValueEnum};
use rust_vattention::{
    cuda::CudaSession,
    model::{Model, Session},
    sampler::Sampler,
    tokenizer::Tokenizer,
};
use std::{
    io::{self, Write},
    path::PathBuf,
    time::Instant,
};

#[derive(Parser)]
#[command(
    version,
    about = "CUDA Llama inference with vAttention-style KV memory"
)]
struct Args {
    #[arg(long, value_enum, default_value_t = Backend::Cuda)]
    backend: Backend,
    #[arg(long, default_value_t = 0)]
    device: usize,
    /// Requested mapping chunk size; rounded up to CUDA VMM granularity.
    #[arg(long, default_value_t = 64)]
    kv_chunk_kib: usize,
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    tokenizer: PathBuf,
    #[arg(long, default_value = "Once upon a time")]
    prompt: String,
    /// Maximum new tokens, excluding prompt tokens.
    #[arg(long, default_value_t = 128)]
    max_tokens: usize,
    /// Zero selects greedy decoding.
    #[arg(long, default_value_t = 0.8)]
    temperature: f32,
    #[arg(long, default_value_t = 0.9)]
    top_p: f32,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long, default_value_t = 4)]
    threads: usize,
}

#[derive(Clone, Copy, ValueEnum)]
enum Backend {
    Cuda,
    Cpu,
}

enum Engine<'a> {
    Cuda(Box<CudaSession>),
    Cpu(Box<Session<'a>>),
}
impl Engine<'_> {
    fn forward(&mut self, token: usize) -> Result<&[f32]> {
        match self {
            Self::Cuda(s) => s.forward(token),
            Self::Cpu(s) => s.forward(token),
        }
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(args.threads > 0, "threads must be positive");
    let mut sampler = Sampler::new(args.temperature, args.top_p, args.seed)?;
    rayon::ThreadPoolBuilder::new()
        .num_threads(args.threads)
        .build_global()?;
    let start = Instant::now();
    let model = Model::load(&args.model)?;
    let tokenizer = Tokenizer::load(&args.tokenizer, model.config().vocab_size)?;
    let prompt = tokenizer.encode(&args.prompt);
    ensure!(
        prompt.len() <= model.config().context,
        "prompt has {} tokens; context limit is {}",
        prompt.len(),
        model.config().context
    );
    let budget = args
        .max_tokens
        .min(model.config().context - prompt.len() + 1);
    if budget < args.max_tokens {
        eprintln!("generation limited to {budget} tokens by context capacity");
    }
    eprintln!(
        "loaded {:?} in {:.2}s",
        model.config(),
        start.elapsed().as_secs_f64()
    );
    let chunk_bytes = args
        .kv_chunk_kib
        .checked_mul(1024)
        .ok_or_else(|| anyhow::anyhow!("KV chunk size overflow"))?;
    ensure!(chunk_bytes > 0, "KV chunk must be positive");
    let mut session = match args.backend {
        Backend::Cuda => Engine::Cuda(Box::new(CudaSession::new(
            &model,
            args.device,
            chunk_bytes,
        )?)),
        Backend::Cpu => Engine::Cpu(Box::new(model.session()?)),
    };
    let mut stdout = io::stdout().lock();
    stdout.write_all(args.prompt.as_bytes())?;
    stdout.flush()?;
    let prefill = Instant::now();
    if args.max_tokens == 0 {
        stdout.write_all(b"\n")?;
        return Ok(());
    }
    for &token in &prompt[..prompt.len() - 1] {
        session.forward(token)?;
    }
    eprintln!(
        "prefill: {} tokens in {:.3}s",
        prompt.len() - 1,
        prefill.elapsed().as_secs_f64()
    );
    let decode_start = Instant::now();
    let mut token = *prompt.last().unwrap();
    let mut generated = 0;
    for _ in 0..budget {
        let next = sampler.sample(session.forward(token)?)?;
        if next == 1 || next == 2 {
            break;
        }
        stdout.write_all(&tokenizer.decode(token, next)?)?;
        stdout.flush()?;
        token = next;
        generated += 1;
    }
    stdout.write_all(b"\n")?;
    if let Engine::Cuda(s) = &session {
        eprintln!("KV cache: {:?}", s.cache_stats());
    }
    eprintln!(
        "generated {generated} tokens in {:.3}s ({:.2} tok/s)",
        decode_start.elapsed().as_secs_f64(),
        generated as f64 / decode_start.elapsed().as_secs_f64()
    );
    Ok(())
}
