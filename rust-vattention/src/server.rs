//! Single-request OpenAI-compatible serving with SSE and client cancellation.
use crate::{cuda::CudaSession, model::Model, sampler::Sampler, text::TextTokenizer};
use anyhow::{ensure, Context, Result};
use axum::{
    extract::State,
    http::StatusCode,
    response::{
        sse::{Event, KeepAlive},
        IntoResponse, Response, Sse,
    },
    routing::{get, post},
    serve::ListenerExt,
    Json, Router,
};
use serde::Deserialize;
use serde_json::{json, Value};
use std::{
    convert::Infallible,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex,
    },
    time::{SystemTime, UNIX_EPOCH},
};
use tokio::sync::{mpsc, oneshot};
use tokio_stream::wrappers::ReceiverStream;

struct Runtime {
    model: Model,
    tokenizer: TextTokenizer,
    cuda: Option<CudaSession>,
}
struct App {
    id: String,
    runtime: Mutex<Runtime>,
}
static NEXT_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Deserialize)]
struct Request {
    model: String,
    prompt: Option<Value>,
    messages: Option<Vec<Value>>,
    max_tokens: Option<usize>,
    max_completion_tokens: Option<usize>,
    #[serde(default)]
    stream: bool,
    stream_options: Option<Value>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    seed: Option<u64>,
    n: Option<usize>,
    stop: Option<Value>,
    #[serde(default)]
    ignore_eos: bool,
    #[serde(flatten)]
    extra: std::collections::HashMap<String, Value>,
}
struct Prepared {
    tokens: Vec<usize>,
    max_tokens: usize,
    sampler: Sampler,
    stops: Vec<String>,
}

fn prepare(app: &App, r: &Request, chat: bool) -> Result<Prepared> {
    ensure!(
        r.model == app.id,
        "unknown model {}; expected {}",
        r.model,
        app.id
    );
    ensure!(r.n.unwrap_or(1) == 1, "only n=1 is supported");
    for key in [
        "tools",
        "tool_choice",
        "response_format",
        "logprobs",
        "prompt_logprobs",
        "suffix",
        "logit_bias",
        "best_of",
    ] {
        ensure!(
            r.extra.get(key).is_none_or(Value::is_null),
            "{key} is unsupported"
        );
    }
    for key in ["min_p", "presence_penalty", "frequency_penalty"] {
        ensure!(
            r.extra.get(key).is_none_or(|v| v.as_f64() == Some(0.)),
            "{key} is unsupported"
        );
    }
    for (key, neutral) in [("repetition_penalty", 1.), ("top_k", -1.)] {
        ensure!(
            r.extra.get(key).is_none_or(|v| v.as_f64() == Some(neutral)),
            "{key} is unsupported"
        );
    }
    ensure!(
        r.extra.get("echo").is_none_or(|v| v == false),
        "echo is unsupported"
    );
    let runtime = app
        .runtime
        .lock()
        .map_err(|_| anyhow::anyhow!("inference worker unavailable"))?;
    let tokens = if chat {
        runtime
            .tokenizer
            .chat(r.messages.as_ref().context("messages is required")?)?
    } else {
        match r.prompt.as_ref().context("prompt is required")? {
            Value::String(text) => runtime.tokenizer.encode(text)?,
            Value::Array(ids) => ids
                .iter()
                .map(|v| {
                    usize::try_from(
                        v.as_u64()
                            .context("prompt must be a string or array of token IDs")?,
                    )
                    .map_err(Into::into)
                })
                .collect::<Result<Vec<_>>>()?,
            _ => anyhow::bail!("prompt must be a string or array of token IDs"),
        }
    };
    let c = runtime.model.config();
    ensure!(
        !tokens.is_empty() && tokens.len() <= c.context,
        "prompt must contain 1..={} tokens",
        c.context
    );
    ensure!(
        tokens.iter().all(|&id| id < c.vocab_size),
        "prompt token outside vocabulary"
    );
    let max_tokens = r.max_completion_tokens.or(r.max_tokens).unwrap_or(16);
    ensure!(
        max_tokens <= c.context - tokens.len() + 1,
        "prompt plus generation exceeds context capacity"
    );
    let stops = match &r.stop {
        None | Some(Value::Null) => vec![],
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(a)) => a
            .iter()
            .map(|v| {
                v.as_str()
                    .map(str::to_owned)
                    .context("stop must contain strings")
            })
            .collect::<Result<_>>()?,
        _ => anyhow::bail!("stop must be a string or array of strings"),
    };
    ensure!(
        stops.iter().all(|s| !s.is_empty()),
        "stop strings must not be empty"
    );
    Ok(Prepared {
        tokens,
        max_tokens,
        sampler: Sampler::new(
            r.temperature.unwrap_or(1.),
            r.top_p.unwrap_or(1.),
            r.seed.unwrap_or(42),
        )?,
        stops,
    })
}

fn api_error(status: StatusCode, message: impl ToString) -> Response {
    (status, Json(json!({"error": {"message": message.to_string(), "type": "invalid_request_error", "param": null, "code": null}}))).into_response()
}
async fn completions(State(app): State<Arc<App>>, Json(r): Json<Request>) -> Response {
    handle(app, r, false).await
}
async fn chat(State(app): State<Arc<App>>, Json(r): Json<Request>) -> Response {
    handle(app, r, true).await
}

async fn handle(app: Arc<App>, r: Request, chat: bool) -> Response {
    // Tokenization can wait behind inference, so keep it off Tokio's worker threads.
    let a = app.clone();
    let prep = tokio::task::spawn_blocking(move || {
        let p = prepare(&a, &r, chat);
        (r, p)
    })
    .await;
    let (r, p) = match prep {
        Ok((r, Ok(p))) => (r, p),
        Ok((_, Err(e))) => return api_error(StatusCode::BAD_REQUEST, e),
        Err(e) => return api_error(StatusCode::INTERNAL_SERVER_ERROR, e),
    };
    let id = format!(
        "{}-{}",
        if chat { "chatcmpl" } else { "cmpl" },
        NEXT_ID.fetch_add(1, Ordering::Relaxed)
    );
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let (tx, rx) = mpsc::channel::<Result<Event, Infallible>>(32);
    let (done_tx, done_rx) = oneshot::channel();
    let stream = r.stream;
    tokio::task::spawn_blocking(move || {
        let result = generate(&app, r, p, chat, &id, created, &tx);
        if stream {
            if let Err(e) = &result {
                let _ = tx.blocking_send(Ok(Event::default().data(
                    json!({"error":{"message":e.to_string(),"type":"server_error"}}).to_string(),
                )));
            }
            let _ = tx.blocking_send(Ok(Event::default().data("[DONE]")));
        }
        let _ = done_tx.send(result);
    });
    if stream {
        Sse::new(ReceiverStream::new(rx))
            .keep_alive(KeepAlive::default())
            .into_response()
    } else {
        // Keep the receiver alive so cancellation detection also works for nonstreaming calls.
        let result = done_rx.await;
        drop(rx);
        match result {
            Ok(Ok(v)) => Json(v).into_response(),
            Ok(Err(e)) => api_error(StatusCode::INTERNAL_SERVER_ERROR, e),
            Err(e) => api_error(StatusCode::INTERNAL_SERVER_ERROR, e),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn generate(
    app: &App,
    r: Request,
    mut p: Prepared,
    chat: bool,
    id: &str,
    created: u64,
    tx: &mpsc::Sender<Result<Event, Infallible>>,
) -> Result<Value> {
    let mut runtime = app
        .runtime
        .lock()
        .map_err(|_| anyhow::anyhow!("inference worker unavailable"))?;
    let Runtime {
        model,
        tokenizer,
        cuda,
    } = &mut *runtime;
    let mut cpu = if cuda.is_none() {
        Some(model.session()?)
    } else {
        None
    };
    if let Some(s) = cuda {
        s.reset();
    }
    let mut next = if p.max_tokens == 0 {
        0
    } else if let Some(s) = cuda {
        s.prefill_sample(&p.tokens, &mut p.sampler)?
    } else {
        let s = cpu.as_mut().unwrap();
        for &token in &p.tokens[..p.tokens.len() - 1] {
            s.forward(token)?;
        }
        p.sampler.sample(s.forward(*p.tokens.last().unwrap())?)?
    };
    let send = |choice: Value, usage: Option<Value>| -> Result<()> {
        let object = if chat {
            "chat.completion.chunk"
        } else {
            "text_completion"
        };
        let mut v =
            json!({"id":id,"object":object,"created":created,"model":app.id,"choices":[choice]});
        if r.stream_options
            .as_ref()
            .is_some_and(|v| v["include_usage"] == true)
        {
            v["usage"] = usage.unwrap_or(Value::Null);
        }
        tx.blocking_send(Ok(Event::default().data(v.to_string())))
            .map_err(|_| anyhow::anyhow!("client disconnected"))
    };
    let mut token = *p.tokens.last().unwrap();
    let mut decoder = tokenizer.decoder(&p.tokens)?;
    let mut text = String::new();
    let mut pending = Vec::<u8>::new();
    let mut sent = 0;
    let mut count = 0;
    let mut reason = "length";
    for step in 0..p.max_tokens {
        if tx.is_closed() {
            anyhow::bail!("client disconnected");
        }
        count += 1;
        if !r.ignore_eos && tokenizer.is_eos(next) {
            reason = "stop";
            break;
        }
        pending.extend(decoder.decode(token, next)?);
        token = next;
        // Legacy byte fallback may emit partial UTF-8; wait for a complete character.
        if let Ok(chunk) = std::str::from_utf8(&pending) {
            text.push_str(chunk);
            pending.clear();
        }
        if let Some(pos) = p.stops.iter().filter_map(|stop| text.find(stop)).min() {
            text.truncate(pos);
            reason = "stop";
            break;
        }
        let hold = p
            .stops
            .iter()
            .flat_map(|stop| {
                (1..stop.len())
                    .filter(|&i| stop.is_char_boundary(i))
                    .filter(|&i| text.ends_with(&stop[..i]))
            })
            .max()
            .unwrap_or(0);
        let end = text.len() - hold;
        if r.stream && end > sent {
            let delta = &text[sent..end];
            let choice = if chat {
                json!({"index":0,"delta":{"role":"assistant","content":delta},"finish_reason":null})
            } else {
                json!({"index":0,"text":delta,"logprobs":null,"finish_reason":null})
            };
            send(choice, None)?;
            sent = end;
        }
        if step + 1 < p.max_tokens {
            next = if let Some(s) = cuda {
                s.forward_sample(token, &mut p.sampler)?
            } else {
                p.sampler.sample(cpu.as_mut().unwrap().forward(token)?)?
            };
        }
    }
    if !pending.is_empty() {
        text.push_str(&String::from_utf8_lossy(&pending));
    }
    let usage = json!({"prompt_tokens":p.tokens.len(),"completion_tokens":count,"total_tokens":p.tokens.len()+count});
    if r.stream {
        let tail = &text[sent..];
        let choice = if chat {
            json!({"index":0,"delta":{"content":tail},"finish_reason":reason})
        } else {
            json!({"index":0,"text":tail,"logprobs":null,"finish_reason":reason})
        };
        send(choice, None)?;
        if r.stream_options
            .as_ref()
            .is_some_and(|v| v["include_usage"] == true)
        {
            let object = if chat {
                "chat.completion.chunk"
            } else {
                "text_completion"
            };
            let v = json!({"id":id,"object":object,"created":created,"model":app.id,"choices":[],"usage":usage});
            tx.blocking_send(Ok(Event::default().data(v.to_string())))
                .map_err(|_| anyhow::anyhow!("client disconnected"))?;
        }
    }
    let choice = if chat {
        json!({"index":0,"message":{"role":"assistant","content":text},"finish_reason":reason})
    } else {
        json!({"index":0,"text":text,"logprobs":null,"finish_reason":reason})
    };
    Ok(
        json!({"id":id,"object":if chat {"chat.completion"} else {"text_completion"},"created":created,"model":app.id,"choices":[choice],"usage":usage}),
    )
}

#[allow(clippy::too_many_arguments)]
pub async fn serve(
    model: Model,
    tokenizer: TextTokenizer,
    id: String,
    host: String,
    port: u16,
    gpu: bool,
    device: usize,
    chunk: usize,
    fp16: bool,
) -> Result<()> {
    ensure!(chunk > 0, "KV chunk must be positive");
    let cuda = if gpu {
        Some(CudaSession::new_with_fp16(&model, device, chunk, fp16)?)
    } else {
        None
    };
    let app = Arc::new(App {
        id,
        runtime: Mutex::new(Runtime {
            model,
            tokenizer,
            cuda,
        }),
    });
    let router = Router::new()
        .route("/health", get(|| async { "ok" }))
        .route("/v1/models", get({ let app = app.clone(); move || { let app = app.clone(); async move { Json(json!({"object":"list","data":[{"id":app.id,"object":"model","created":0,"owned_by":"local"}]})) } } }))
        .route("/v1/completions", post(completions))
        .route("/v1/chat/completions", post(chat))
        .with_state(app);
    let listener = tokio::net::TcpListener::bind((host.as_str(), port)).await?;
    eprintln!("OpenAI-compatible API: http://{host}:{port}/v1 (requests execute sequentially)");
    // SSE consists of small writes; avoid Nagle/delayed-ACK stalls between
    // response headers and the first token, and between subsequent tokens.
    let listener = listener.tap_io(|stream| {
        if let Err(error) = stream.set_nodelay(true) {
            eprintln!("TCP_NODELAY: {error}");
        }
    });
    axum::serve(listener, router).await?;
    Ok(())
}
