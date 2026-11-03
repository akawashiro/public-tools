use anyhow::{ensure, Result};
use clap::{Parser, ValueEnum};
use rust_vattention::{
    cuda::CudaSession,
    hub,
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
    /// Local llama2.c file, Hugging Face model ID, or local HF snapshot directory.
    #[arg(long)]
    model: String,
    /// Required only for local llama2.c checkpoints.
    #[arg(long)]
    tokenizer: Option<PathBuf>,
    #[arg(long, default_value = "main")]
    revision: String,
    #[arg(long, default_value = "models/hub")]
    cache_dir: PathBuf,
    /// GPU weight storage; sums, activations and KV remain FP32.
    #[arg(long, value_enum, default_value_t = WeightDtype::Fp32)]
    weight_dtype: WeightDtype,
    /// Limit the model context (recommended for large default contexts).
    #[arg(long)]
    context_length: Option<usize>,
    /// Run an OpenAI-compatible HTTP server.
    #[arg(long)]
    serve: bool,
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    #[arg(long, default_value_t = 8000)]
    port: u16,
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
enum WeightDtype {
    Fp32,
    Fp16,
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
    fn prefill_sample(&mut self, tokens: &[usize], sampler: &mut Sampler) -> Result<usize> {
        match self {
            Self::Cuda(s) => s.prefill_sample(tokens, sampler),
            Self::Cpu(s) => {
                for &token in &tokens[..tokens.len() - 1] {
                    s.forward(token)?;
                }
                sampler.sample(s.forward(*tokens.last().unwrap())?)
            }
        }
    }
    fn forward_sample(&mut self, token: usize, sampler: &mut Sampler) -> Result<usize> {
        match self {
            Self::Cuda(s) => s.forward_sample(token, sampler),
            Self::Cpu(s) => sampler.sample(s.forward(token)?),
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(args.threads > 0, "threads must be positive");
    let mut sampler = Sampler::new(args.temperature, args.top_p, args.seed)?;
    rayon::ThreadPoolBuilder::new()
        .num_threads(args.threads)
        .build_global()?;
    let chunk_bytes = args
        .kv_chunk_kib
        .checked_mul(1024)
        .ok_or_else(|| anyhow::anyhow!("KV chunk size overflow"))?;
    ensure!(chunk_bytes > 0, "KV chunk must be positive");
    let start = Instant::now();
    let (mut model, tokenizer) = if std::path::Path::new(&args.model).is_file() {
        let model = Model::load(&args.model)?;
        let tokenizer = Tokenizer::load(
            args.tokenizer.as_ref().ok_or_else(|| {
                anyhow::anyhow!("--tokenizer is required for a llama2.c checkpoint")
            })?,
            model.config().vocab_size,
        )?;
        (
            model,
            rust_vattention::text::TextTokenizer::Legacy(tokenizer),
        )
    } else {
        ensure!(
            args.tokenizer.is_none(),
            "--tokenizer is only used with llama2.c files"
        );
        let loaded = hub::load(&args.model, &args.revision, &args.cache_dir)?;
        let tokenizer = rust_vattention::text::TextTokenizer::HuggingFace {
            tokenizer: Box::new(loaded.tokenizer),
            bos: loaded.bos,
            eos: loaded.eos,
            chat_template: loaded.chat_template,
        };
        (loaded.model, tokenizer)
    };
    if let Some(context) = args.context_length {
        model.limit_context(context)?;
    }
    if args.serve {
        return rust_vattention::server::serve(
            model,
            tokenizer,
            args.model,
            args.host,
            args.port,
            matches!(args.backend, Backend::Cuda),
            args.device,
            chunk_bytes,
            matches!(args.weight_dtype, WeightDtype::Fp16),
        )
        .await;
    }
    let prompt = tokenizer.encode(&args.prompt)?;
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
    let mut session = match args.backend {
        Backend::Cuda => Engine::Cuda(Box::new(CudaSession::new_with_fp16(
            &model,
            args.device,
            chunk_bytes,
            matches!(args.weight_dtype, WeightDtype::Fp16),
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
    let mut next = session.prefill_sample(&prompt, &mut sampler)?;
    eprintln!(
        "prefill: {} tokens in {:.3}s",
        prompt.len(),
        prefill.elapsed().as_secs_f64()
    );
    let decode_start = Instant::now();
    let mut token = *prompt.last().unwrap();
    let mut generated = 0;
    let mut decoder = tokenizer.decoder(&prompt)?;
    for step in 0..budget {
        if tokenizer.is_eos(next) {
            break;
        }
        stdout.write_all(&decoder.decode(token, next)?)?;
        stdout.flush()?;
        token = next;
        generated += 1;
        if step + 1 < budget {
            next = session.forward_sample(token, &mut sampler)?;
        }
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
