//! Per-model constants shared by all eager, prefill, and captured CUDA paths.
use crate::model::Config;

pub(super) fn prelude(c: &Config) -> String {
    let rope = if let Some(freq) = &c.rope_inv_freq {
        let bits = freq
            .iter()
            .map(|f| format!("{}u", f.to_bits()))
            .collect::<Vec<_>>()
            .join(",");
        format!("__device__ __constant__ unsigned rope_frequency_bits[] = {{{bits}}};\n\
          __device__ __forceinline__ float rope_angle(unsigned long long i, unsigned long long head, unsigned long long pos, float theta) {{\n\
          return float(pos)*__uint_as_float(rope_frequency_bits[(i%head)/2]); }}\n")
    } else {
        "__device__ __forceinline__ float rope_angle(unsigned long long i, unsigned long long head, unsigned long long pos, float theta) { return float(pos)/powf(theta,float(i%head)/float(head)); }\n".into()
    };
    let window = c.sliding_window.map_or("return 0ull;".into(), |w| {
        format!("return length>{w}ull ? length-{w}ull : 0ull;")
    });
    format!("#define WINDOW_ATTENTION {}\n{rope}\n__device__ __forceinline__ unsigned long long attention_first(unsigned long long length) {{ {window} }}\n", usize::from(c.sliding_window.is_some()))
}
