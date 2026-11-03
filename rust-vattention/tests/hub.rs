use rust_vattention::{cuda::CudaSession, hub, text::TextTokenizer};
use serde::Deserialize;
use std::path::{Path, PathBuf};

const FIXTURES: &[&str] = &[
    "llama-f32",
    "qwen3-f16",
    "qwen3-bf16-sharded",
    "qwen2-f16",
    "qwen2-wide-f16",
    "llama3-f32",
    "llama-linear-f16",
    "mistral-window-f16",
    "mistral-wide-f16",
    "mistral-full-f32",
];

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}
#[derive(Deserialize)]
struct Expected {
    tokens: Vec<usize>,
    logits: Vec<Vec<f32>>,
}

#[test]
fn hf_logits_match_transformers() {
    for &name in FIXTURES {
        let path = fixture(name);
        let loaded = hub::load(path.to_str().unwrap(), "main", Path::new("unused")).unwrap();
        let reference: Expected =
            serde_json::from_slice(&std::fs::read(path.join("expected.json")).unwrap()).unwrap();
        let tokenizer = TextTokenizer::HuggingFace {
            tokenizer: Box::new(loaded.tokenizer),
            bos: loaded.bos,
            eos: loaded.eos,
            chat_template: loaded.chat_template,
        };
        assert_eq!(tokenizer.encode("t9 t11").unwrap(), vec![5, 9, 11]);
        assert!(tokenizer.is_eos(7) && tokenizer.is_eos(8));
        assert!(!tokenizer.is_eos(2));
        assert_eq!(
            tokenizer
                .chat(&[serde_json::json!({"role":"user", "content":"t9 t11"})])
                .unwrap(),
            vec![9, 11]
        );
        let mut session = loaded.model.session().unwrap();
        for (&token, expected) in reference.tokens.iter().zip(&reference.logits) {
            let actual = session.forward(token).unwrap();
            let error = actual
                .iter()
                .zip(expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0., f32::max);
            assert!(error < 2e-5, "{name}: max logit error {error}");
        }
    }
}

#[test]
#[ignore = "requires a CUDA GPU, driver and NVRTC"]
fn hf_cuda_logits_match_transformers() {
    for &name in FIXTURES {
        let path = fixture(name);
        let loaded = hub::load(path.to_str().unwrap(), "main", Path::new("unused")).unwrap();
        let reference: Expected =
            serde_json::from_slice(&std::fs::read(path.join("expected.json")).unwrap()).unwrap();
        for fp16 in [false, true] {
            let mut gpu = CudaSession::new_with_fp16(&loaded.model, 0, 65536, fp16).unwrap();
            for (&token, expected) in reference.tokens.iter().zip(&reference.logits) {
                let actual = gpu.forward(token).unwrap();
                let error = actual
                    .iter()
                    .zip(expected)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0., f32::max);
                assert!(
                    error < if fp16 { 5e-4 } else { 2e-5 },
                    "{name}, fp16={fp16}: max logit error {error}"
                );
            }
        }
    }
}

#[test]
fn rejects_corrupt_hf_checkpoints() {
    struct Scratch(PathBuf);
    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    let scratch = Scratch(
        std::env::temp_dir().join(format!("rust-vattention-hf-errors-{}", std::process::id())),
    );
    std::fs::create_dir_all(&scratch.0).unwrap();
    let load_error = || {
        let error = hub::load(scratch.0.to_str().unwrap(), "main", Path::new("unused"))
            .err()
            .expect("corrupt checkpoint accepted");
        format!("{error:#}")
    };
    for entry in std::fs::read_dir(fixture("qwen3-f16")).unwrap() {
        let entry = entry.unwrap();
        std::fs::copy(entry.path(), scratch.0.join(entry.file_name())).unwrap();
    }
    let path = scratch.0.join("model.safetensors");
    let original = std::fs::read(&path).unwrap();
    let mut integer_weights = original.clone();
    let pos = integer_weights
        .windows(3)
        .position(|v| v == b"F16")
        .unwrap();
    integer_weights[pos] = b'I';
    std::fs::write(&path, integer_weights).unwrap();
    assert!(load_error().contains("unsupported weight dtype"));
    let mut nonfinite = original.clone();
    let header = u64::from_le_bytes(nonfinite[..8].try_into().unwrap()) as usize;
    nonfinite[8 + header..10 + header].copy_from_slice(&0x7e00u16.to_le_bytes());
    std::fs::write(&path, nonfinite).unwrap();
    assert!(load_error().contains("NaN or infinity"));
    std::fs::write(&path, original).unwrap();
    let index = scratch.0.join("model.safetensors.index.json");
    std::fs::write(&index, b"{").unwrap();
    assert!(load_error().contains("EOF"));
    std::fs::write(&index, b"{\"weight_map\":{}}").unwrap();
    assert!(load_error().contains("empty safetensors index"));
}

#[test]
#[ignore = "requires CUDA and cuBLAS"]
fn hf_batched_prefill_matches_transformers() {
    for &name in FIXTURES {
        let path = fixture(name);
        let loaded = hub::load(path.to_str().unwrap(), "main", Path::new("unused")).unwrap();
        let reference: Expected =
            serde_json::from_slice(&std::fs::read(path.join("expected.json")).unwrap()).unwrap();
        for fp16 in [false, true] {
            let mut gpu = CudaSession::new_with_fp16(&loaded.model, 0, 65536, fp16).unwrap();
            for split in [1, 2, 4, 5, 16, 256, 257]
                .into_iter()
                .filter(|&n| n <= reference.tokens.len())
            {
                gpu.reset_and_release().unwrap();
                let actual = gpu.prefill(&reference.tokens[..split]).unwrap();
                let err = actual
                    .iter()
                    .zip(&reference.logits[split - 1])
                    .map(|(a, b)| (a - b).abs())
                    .fold(0., f32::max);
                assert!(
                    err < if fp16 { 5e-4 } else { 2e-5 },
                    "{name}, split={split}, fp16={fp16}: {err}"
                );
                assert_eq!(gpu.position(), split);
                for i in split..reference.tokens.len() {
                    let actual = gpu.forward(reference.tokens[i]).unwrap();
                    let err = actual
                        .iter()
                        .zip(&reference.logits[i])
                        .map(|(a, b)| (a - b).abs())
                        .fold(0., f32::max);
                    assert!(
                        err < if fp16 { 5e-4 } else { 2e-5 },
                        "decode after prefill: {name}, token={i}, fp16={fp16}: {err}"
                    );
                }
            }
            gpu.reset();
            gpu.prefill(&reference.tokens[..2]).unwrap();
            let actual = gpu.prefill(&reference.tokens[2..]).unwrap();
            let err = actual
                .iter()
                .zip(reference.logits.last().unwrap())
                .map(|(a, b)| (a - b).abs())
                .fold(0., f32::max);
            assert!(
                err < if fp16 { 5e-4 } else { 2e-5 },
                "appended prefill: {name}, fp16={fp16}: {err}"
            );
            let best = |logits: &[f32]| {
                logits
                    .iter()
                    .enumerate()
                    .max_by(|(i, a), (j, b)| a.total_cmp(b).then_with(|| j.cmp(i)))
                    .unwrap()
                    .0
            };
            gpu.reset_and_release().unwrap();
            assert_eq!(
                gpu.prefill_greedy(&reference.tokens[..3]).unwrap(),
                best(&reference.logits[2])
            );
            for i in 3..reference.tokens.len() {
                assert_eq!(
                    gpu.forward_greedy(reference.tokens[i]).unwrap(),
                    best(&reference.logits[i])
                );
            }
            let position = gpu.position();
            assert!(gpu.prefill(&[]).is_err());
            assert!(gpu.prefill(&[loaded.model.config().vocab_size]).is_err());
            assert_eq!(gpu.position(), position);
        }
    }
}
