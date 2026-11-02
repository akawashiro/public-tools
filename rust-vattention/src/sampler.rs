use crate::softmax;
use anyhow::{ensure, Result};
use rand::{rngs::StdRng, Rng, SeedableRng};

pub struct Sampler {
    temperature: f32,
    top_p: f32,
    rng: StdRng,
}

impl Sampler {
    pub fn new(temperature: f32, top_p: f32, seed: u64) -> Result<Self> {
        ensure!(
            temperature.is_finite() && temperature >= 0.,
            "temperature must be finite and >= 0"
        );
        ensure!(
            top_p.is_finite() && top_p > 0. && top_p <= 1.,
            "top-p must be in (0, 1]"
        );
        Ok(Self {
            temperature,
            top_p,
            rng: StdRng::seed_from_u64(seed),
        })
    }

    pub fn sample(&mut self, logits: &[f32]) -> Result<usize> {
        ensure!(
            !logits.is_empty() && logits.iter().all(|v| v.is_finite()),
            "logits must be nonempty and finite"
        );
        if self.temperature == 0. {
            let mut best = 0;
            for i in 1..logits.len() {
                if logits[i] > logits[best] {
                    best = i;
                }
            }
            return Ok(best);
        }
        // Subtract in f64 before scaling to avoid overflow at tiny temperatures.
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let mut probs: Vec<f32> = logits
            .iter()
            .map(|v| (((*v as f64) - max) / self.temperature as f64) as f32)
            .collect();
        softmax(&mut probs);
        let mut ids: Vec<usize> = (0..probs.len()).collect();
        ids.sort_by(|a, b| probs[*b].total_cmp(&probs[*a]));
        let mut mass = 0.;
        let mut count = 0;
        for &id in &ids {
            mass += probs[id];
            count += 1;
            if mass >= self.top_p {
                break;
            }
        }
        let target = self.rng.gen::<f32>() * mass;
        let mut cumulative = 0.;
        for &id in &ids[..count] {
            cumulative += probs[id];
            if target < cumulative {
                return Ok(id);
            }
        }
        Ok(ids[count - 1])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn greedy_nucleus_and_seed() {
        assert_eq!(
            Sampler::new(0., 1., 0)
                .unwrap()
                .sample(&[2., 4., 4.])
                .unwrap(),
            1
        );
        let mut s = Sampler::new(1., 0.1, 0).unwrap();
        for _ in 0..100 {
            assert_eq!(s.sample(&[0., 8., 1.]).unwrap(), 1);
        }
        let mut a = Sampler::new(0.8, 0.9, 42).unwrap();
        let mut b = Sampler::new(0.8, 0.9, 42).unwrap();
        for _ in 0..100 {
            assert_eq!(
                a.sample(&[1., 2., 3.]).unwrap(),
                b.sample(&[1., 2., 3.]).unwrap()
            );
        }
        assert!(a.sample(&[f32::NAN]).is_err());
        assert!(Sampler::new(-1., 1., 0).is_err());
    }
}
