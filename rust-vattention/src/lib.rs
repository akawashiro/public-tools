//! CPU/CUDA Llama and Qwen3 inference with Hugging Face safetensors or llama2.c checkpoints.
pub mod cuda;
pub mod hub;
pub mod model;
pub mod sampler;
pub mod tokenizer;

use anyhow::{ensure, Result};
use std::io::Read;

pub(crate) fn word(reader: &mut impl Read) -> Result<[u8; 4]> {
    let mut bytes = [0; 4];
    reader.read_exact(&mut bytes)?;
    Ok(bytes)
}

pub(crate) fn softmax(values: &mut [f32]) {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.;
    for value in values.iter_mut() {
        *value = (*value - max).exp();
        sum += *value;
    }
    for value in values {
        *value /= sum;
    }
}

pub(crate) fn product(values: &[usize]) -> Result<usize> {
    let mut result = 1usize;
    for value in values {
        result = result
            .checked_mul(*value)
            .ok_or_else(|| anyhow::anyhow!("model dimensions overflow"))?;
    }
    ensure!(
        result <= isize::MAX as usize / 4,
        "model allocation is too large"
    );
    Ok(result)
}

pub mod server;
pub mod text;
