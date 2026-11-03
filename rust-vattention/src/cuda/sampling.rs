//! Full-vocabulary GPU top-p sampling. No top-k approximation or CPU sort.
use super::{decode::DecodeGraph, Buffer, CudaSession};
use crate::sampler::Sampler;
use anyhow::{Context, Result};
use cudarc::driver::sys;

pub(super) struct Sampling {
    blocks: usize,
    keys: [Buffer; 2],
    ids: [Buffer; 2],
    weights: Buffer,
    partial: Buffer,
    total: Buffer,
    hist: Buffer,
    offsets: Buffer,
    masses: Buffer,
    prefix: Buffer,
    graph: Option<DecodeGraph>,
}
impl CudaSession {
    pub fn prefill_sample(&mut self, tokens: &[usize], sampler: &mut Sampler) -> Result<usize> {
        self.prefill_compute(tokens)?;
        self.sample_logits(sampler)
    }
    pub fn forward_sample(&mut self, token: usize, sampler: &mut Sampler) -> Result<usize> {
        self.decode(token)?;
        self.sample_logits(sampler)
    }
    pub(super) fn sample_logits(&mut self, sampler: &mut Sampler) -> Result<usize> {
        if sampler.is_greedy() {
            return self.greedy_from_logits();
        }
        if self.sampling.is_none() {
            let n = self.config.vocab_size;
            let blocks = n.div_ceil(1024);
            let array = |len| Buffer::new(&self.ctx, len);
            self.sampling = Some(Sampling {
                blocks,
                keys: [array(n)?, array(n)?],
                ids: [array(n)?, array(n)?],
                weights: array(n)?,
                partial: array(blocks)?,
                total: array(1)?,
                hist: array(blocks * 16)?,
                offsets: array(blocks * 16)?,
                masses: array(blocks)?,
                prefix: array(blocks)?,
                graph: None,
            });
        }
        // Commit RNG state only on successful sampling, like the CPU path's
        // finite-logit validation before drawing a random value.
        let mut staged = sampler.clone();
        let (temperature, top_p, uniform) =
            staged.gpu_draw().context("expected stochastic sampler")?;
        self.launch(
            "set_sampling",
            &[
                temperature.to_bits() as u64,
                top_p.to_bits() as u64,
                uniform.to_bits() as u64,
            ],
            1,
        )?;
        if self.use_graphs {
            if self.sampling.as_ref().unwrap().graph.is_none() {
                self.stream
                    .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
                let result = self.sampling_body();
                let graph = self.stream.end_capture(sys::CUgraphInstantiate_flags(0));
                result?;
                self.sampling.as_mut().unwrap().graph =
                    Some(DecodeGraph::new(graph?.context("empty sampling graph")?));
            }
            self.sampling
                .as_ref()
                .unwrap()
                .graph
                .as_ref()
                .unwrap()
                .launch()?;
        } else {
            self.sampling_body()?;
        }
        let token = self.finish_sample()?;
        *sampler = staged;
        Ok(token)
    }
    fn sampling_body(&self) -> Result<()> {
        let s = self.sampling.as_ref().unwrap();
        let n = self.config.vocab_size;
        let blocks = s.blocks;
        self.argmax_device()?;
        self.launch(
            "sampling_init",
            &[
                self.logits.ptr,
                self.residual.ptr,
                s.keys[0].ptr,
                s.ids[0].ptr,
                s.weights.ptr,
                s.partial.ptr,
                n as u64,
            ],
            blocks,
        )?;
        self.launch(
            "sampling_sum",
            &[s.partial.ptr, s.total.ptr, blocks as u64],
            1,
        )?;
        self.launch(
            "sampling_keys",
            &[
                s.weights.ptr,
                s.keys[0].ptr,
                s.ids[0].ptr,
                s.total.ptr,
                n as u64,
            ],
            n.div_ceil(256),
        )?;
        for pass in 0..8 {
            let input = pass % 2;
            let output = 1 - input;
            let shift = (pass * 4) as u64;
            self.launch(
                "radix_hist",
                &[s.keys[input].ptr, s.hist.ptr, n as u64, shift],
                blocks,
            )?;
            self.launch(
                "radix_offsets",
                &[s.hist.ptr, s.offsets.ptr, blocks as u64],
                1,
            )?;
            self.launch(
                "radix_scatter",
                &[
                    s.keys[input].ptr,
                    s.ids[input].ptr,
                    s.keys[output].ptr,
                    s.ids[output].ptr,
                    s.offsets.ptr,
                    n as u64,
                    blocks as u64,
                    shift,
                ],
                blocks,
            )?;
        }
        self.launch(
            "ordered_mass",
            &[
                s.weights.ptr,
                s.ids[0].ptr,
                s.total.ptr,
                s.masses.ptr,
                n as u64,
            ],
            blocks,
        )?;
        self.launch(
            "mass_prefix",
            &[s.masses.ptr, s.prefix.ptr, blocks as u64],
            1,
        )?;
        self.launch(
            "sample_final",
            &[
                s.weights.ptr,
                s.ids[0].ptr,
                s.total.ptr,
                s.prefix.ptr,
                self.next_token.ptr,
                n as u64,
                blocks as u64,
            ],
            1,
        )?;
        self.copy_token_async()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Model;
    #[test]
    #[ignore = "requires CUDA and RUST_VATTENTION_TEST_MODEL"]
    fn gpu_nucleus_matches_peaky_cpu_and_stable_radix_order() -> Result<()> {
        let model = Model::load(std::env::var("RUST_VATTENTION_TEST_MODEL")?)?;
        let mut gpu = CudaSession::new(&model, 0, 65536)?;
        let n = model.config().vocab_size;
        let mut values = vec![-80.; n];
        for (index, value) in [(1, 1.1), (1023, 2.0), (1024, 1.5), (n - 1, 0.3), (2, 2.0)] {
            values[index] = value;
        }
        for temp in [0.5, 1.0, 2.0, 1e-8] {
            for top_p in [0.1, 0.5, 0.9, 1.0] {
                gpu.logits.upload(&values)?;
                let mut cpu = Sampler::new(temp, top_p, 42)?;
                let mut sampler = cpu.clone();
                for _ in 0..20 {
                    let expected = cpu.sample(&values)?;
                    assert_eq!(
                        gpu.sample_logits(&mut sampler)?,
                        expected,
                        "temp={temp}, top_p={top_p}"
                    );
                }
            }
        }
        // A synthetic vocabulary with many ties and signs crosses every radix
        // digit and multiple thread blocks; check every sorted ID, not just a sample.
        for (i, v) in values.iter_mut().enumerate() {
            *v = ((i * 17 % 97) as f32 - 48.) / 8.;
        }
        values[0] = -0.;
        values[1] = 0.;
        gpu.logits.upload(&values)?;
        gpu.sample_logits(&mut Sampler::new(1., 0.9, 7)?)?;
        let state = gpu.sampling.as_ref().unwrap();
        let mut indices = vec![0u32; n];
        let mut probs = vec![0f32; n];
        // SAFETY: sample_logits synchronized; arrays match the owned allocations.
        unsafe {
            super::super::check(
                sys::cuMemcpyDtoH_v2(indices.as_mut_ptr().cast(), state.ids[0].ptr, n * 4),
                "read sorted IDs",
            )?;
            super::super::check(
                sys::cuMemcpyDtoH_v2(probs.as_mut_ptr().cast(), state.weights.ptr, n * 4),
                "read probabilities",
            )?;
        }
        let mut expected: Vec<_> = (0..n as u32).collect();
        expected.sort_by(|&a, &b| {
            probs[b as usize]
                .total_cmp(&probs[a as usize])
                .then(a.cmp(&b))
        });
        assert_eq!(indices, expected);
        // Uniform weights require the stable lowest-ID ordering even at high temperature.
        values.fill(0.);
        gpu.logits.upload(&values)?;
        let mut sampler = Sampler::new(1e20, 0.5, 9)?;
        for _ in 0..20 {
            assert!(gpu.sample_logits(&mut sampler)? <= n / 2);
        }
        // An exactly representable inverse-CDF boundary belongs to the next
        // interval, matching CPU target < cumulative (including zero masses).
        for support in [2, 2048.min(n)] {
            values.fill(f32::MIN);
            values[..support].fill(0.);
            gpu.logits.upload(&values)?;
            gpu.launch(
                "set_sampling",
                &[
                    1f32.to_bits() as u64,
                    1f32.to_bits() as u64,
                    0.5f32.to_bits() as u64,
                ],
                1,
            )?;
            gpu.sampling_body()?;
            assert_eq!(gpu.finish_sample()?, support / 2);
        }
        // Invalid logits must not consume RNG state.
        let mut sampler = Sampler::new(0.8, 0.9, 101)?;
        values[2] = f32::NAN;
        gpu.logits.upload(&values)?;
        assert!(gpu.sample_logits(&mut sampler).is_err());
        values.fill(-80.);
        values[1] = 1.;
        values[2] = 2.;
        gpu.logits.upload(&values)?;
        let mut expected = Sampler::new(0.8, 0.9, 101)?;
        assert_eq!(gpu.sample_logits(&mut sampler)?, expected.sample(&values)?);
        Ok(())
    }
    #[test]
    #[ignore = "requires CUDA and RUST_VATTENTION_TEST_MODEL"]
    fn radix_sort_handles_partial_blocks_and_large_vocabularies() -> Result<()> {
        let model = Model::load(std::env::var("RUST_VATTENTION_TEST_MODEL")?)?;
        let mut gpu = CudaSession::new(&model, 0, 65536)?;
        for n in [259usize, 1025, 32000, 151936, 262401] {
            // Isolate the vocabulary sampler: resize only its tensors and never
            // invoke model forward with the synthetic vocabulary dimensions.
            gpu.sampling = None;
            gpu.greedy_graph = None;
            gpu.config.vocab_size = n;
            gpu.logits = Buffer::new(&gpu.ctx, n)?;
            gpu.argmax_values = Buffer::new(&gpu.ctx, n.div_ceil(1024))?;
            gpu.argmax_ids = Buffer::new_bytes(&gpu.ctx, n.div_ceil(1024) * 4)?;
            let values: Vec<_> = (0..n)
                .map(|i| ((i * 7919 % 2039) as f32 - 1019.) / 32.)
                .collect();
            gpu.logits.upload(&values)?;
            gpu.sample_logits(&mut Sampler::new(1., 0.9, 42)?)?;
            let state = gpu.sampling.as_ref().unwrap();
            let mut actual = vec![0u32; n];
            let mut probs = vec![0f32; n];
            // SAFETY: the sampler completed and both host buffers match their tensors.
            unsafe {
                super::super::check(
                    sys::cuMemcpyDtoH_v2(actual.as_mut_ptr().cast(), state.ids[0].ptr, n * 4),
                    "read radix output",
                )?;
                super::super::check(
                    sys::cuMemcpyDtoH_v2(probs.as_mut_ptr().cast(), state.weights.ptr, n * 4),
                    "read probabilities",
                )?;
            }
            let mut expected: Vec<_> = (0..n as u32).collect();
            expected.sort_by(|&a, &b| {
                probs[b as usize]
                    .total_cmp(&probs[a as usize])
                    .then(a.cmp(&b))
            });
            assert_eq!(actual, expected, "vocabulary={n}");
        }
        Ok(())
    }
}
