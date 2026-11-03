use crate::tokenizer::Tokenizer;
use anyhow::{ensure, Context, Result};
use serde_json::Value;

pub enum TextTokenizer {
    Legacy(Tokenizer),
    HuggingFace {
        tokenizer: Box<tokenizers::Tokenizer>,
        bos: Option<usize>,
        eos: Vec<usize>,
        chat_template: Option<String>,
    },
}
impl TextTokenizer {
    pub fn encode(&self, text: &str) -> Result<Vec<usize>> {
        match self {
            Self::Legacy(t) => Ok(t.encode(text)),
            Self::HuggingFace { tokenizer, bos, .. } => {
                let ids = tokenizer
                    .encode(text, true)
                    .map_err(|e| anyhow::anyhow!("encode: {e}"))?;
                let mut ids: Vec<_> = ids.get_ids().iter().map(|&i| i as usize).collect();
                if ids.is_empty() {
                    ids.push(bos.context("empty prompt without BOS token")?);
                }
                Ok(ids)
            }
        }
    }
    pub fn is_eos(&self, id: usize) -> bool {
        match self {
            Self::Legacy(_) => id == 1 || id == 2,
            Self::HuggingFace { eos, .. } => eos.contains(&id),
        }
    }
    pub fn chat(&self, messages: &[Value]) -> Result<Vec<usize>> {
        ensure!(!messages.is_empty(), "messages must not be empty");
        for m in messages {
            ensure!(
                matches!(m["role"].as_str(), Some("system" | "user" | "assistant")),
                "only system/user/assistant messages are supported"
            );
            ensure!(m["content"].is_string(), "message content must be a string");
        }
        let Self::HuggingFace {
            tokenizer,
            bos,
            eos,
            chat_template: Some(template),
        } = self
        else {
            anyhow::bail!("this model has no chat template; use /v1/completions or an instruct model with tokenizer_config.json");
        };
        let mut env = minijinja::Environment::new();
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_function(
            "raise_exception",
            |message: String| -> Result<String, minijinja::Error> {
                Err(minijinja::Error::new(
                    minijinja::ErrorKind::InvalidOperation,
                    message,
                ))
            },
        );
        env.add_template("chat", template)?;
        let bos = bos
            .and_then(|id| tokenizer.id_to_token(id as u32))
            .unwrap_or_default();
        let eos = eos
            .first()
            .and_then(|&id| tokenizer.id_to_token(id as u32))
            .unwrap_or_default();
        let text = env.get_template("chat")?.render(minijinja::context!(messages => messages, add_generation_prompt => true, bos_token => bos, eos_token => eos, enable_thinking => false))?;
        // Chat templates already contain their special tokens.
        let encoded = tokenizer
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("encode chat: {e}"))?;
        let ids: Vec<_> = encoded.get_ids().iter().map(|&id| id as usize).collect();
        ensure!(!ids.is_empty(), "chat template produced an empty prompt");
        Ok(ids)
    }
    pub fn decoder(&self, prompt: &[usize]) -> Result<Decoder<'_>> {
        Ok(match self {
            Self::Legacy(t) => Decoder {
                step: Box::new(move |prev, id| t.decode(prev, id)),
            },
            Self::HuggingFace { tokenizer, .. } => {
                let mut stream = tokenizer.decode_stream(true);
                // Preserve tokenizer context, including SentencePiece's leading-space
                // rules and byte fallback across the prompt/generation boundary.
                for &id in prompt {
                    stream
                        .step(u32::try_from(id)?)
                        .map_err(|e| anyhow::anyhow!("decode prompt: {e}"))?;
                }
                Decoder {
                    step: Box::new(move |_, id| {
                        let chunk = stream
                            .step(u32::try_from(id)?)
                            .map_err(|e| anyhow::anyhow!("decode: {e}"))?;
                        Ok(chunk.unwrap_or_default().into_bytes())
                    }),
                }
            }
        })
    }
}
type DecodeFn<'a> = dyn FnMut(usize, usize) -> Result<Vec<u8>> + 'a;
pub struct Decoder<'a> {
    step: Box<DecodeFn<'a>>,
}
impl Decoder<'_> {
    pub fn decode(&mut self, previous: usize, token: usize) -> Result<Vec<u8>> {
        (self.step)(previous, token)
    }
}
