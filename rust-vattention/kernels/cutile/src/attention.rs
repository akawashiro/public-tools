// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Online attention structure follows NVIDIA's cutile-rs causal-attention example.
// FP32 Q and VMM K/V are read directly using token-major strides. Only registers
// are converted to the selected MMA type; the cache keeps its FP32 representation and addresses.
#[cutile::module]
pub mod vattention_tiles {
    use cutile::core::*;

    #[cutile::entry(optimization_hints=(sm_120=(num_cta_in_cga=1,),))]
    fn vmm_attention<E: ElementType, const BM: i32, const BN: i32, const D: i32>(
        out: &mut Tensor<f16, { [1, BM, D] }>,
        q: &Tensor<f32, { [-1, -1, D] }>,
        k: &Tensor<f32, { [-1, -1, D] }>,
        v: &Tensor<f32, { [-1, -1, D] }>,
        scale: f32,
        position: i32,
        window: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let h: i32 = pid.0;
        let qb: i32 = pid.1;
        let kh: i32 = h / (q.shape()[0] / k.shape()[0]);
        let qp = q.partition(shape![1, BM, D]);
        let kp = k.partition(shape![1, BN, D]);
        let vp = v.partition(shape![1, BN, D]);
        let qf: Tile<f32, { [BM, D] }> = qp.load([h, qb, 0i32]).reshape(shape![BM, D]);
        let qt: Tile<E, { [BM, D] }> = convert_tile(qf);
        let mi: Tile<i32, { [BM] }> = iota(shape![BM]);
        let ni: Tile<i32, { [BN] }> = iota(shape![BN]);
        let rows: Tile<i32, { [BM, BN] }> = (mi + (position + qb * BM).broadcast(shape![BM]))
            .reshape(shape![BM, 1])
            .broadcast(shape![BM, BN]);
        let columns: Tile<i32, { [BM, BN] }> = ni.reshape(shape![1, BN]).broadcast(shape![BM, BN]);
        let neginf: Tile<f32, { [BM, 1] }> = constant(f32::NEG_INFINITY, shape![BM, 1]);
        let zeros: Tile<f32, { [BM, 1] }> = constant(0.0f32, shape![BM, 1]);
        let zero_scores: Tile<f32, { [BM, BN] }> = constant(0.0f32, shape![BM, BN]);
        let mask_false: Tile<f32, { [BM, BN] }> = constant(f32::NEG_INFINITY, shape![BM, BN]);
        let mut maximum: Tile<f32, { [BM, 1] }> = neginf;
        let mut denominator: Tile<f32, { [BM, 1] }> = zeros;
        let mut acc: Tile<f32, { [BM, D] }> = constant(0.0f32, shape![BM, D]);
        let length: i32 = k.shape()[1];
        let end: i32 = ceil_div(min(position + (qb + 1i32) * BM, length), BN);
        let mut first: i32 = 0i32;
        if window > 0i32 {
            first = max(0i32, position + qb * BM + 1i32 - window) / BN;
        }
        for j in first..end {
            let kf: Tile<f32, { [BN, D] }> = kp.load([kh, j, 0i32]).reshape(shape![BN, D]);
            let kt: Tile<E, { [D, BN] }> = convert_tile(kf.transpose());
            let dot: Tile<f32, { [BM, BN] }> = mma(qt, kt, zero_scores);
            let scores: Tile<f32, { [BM, BN] }> =
                dot * (scale * 1.4426950408889634f32).broadcast(shape![BM, BN]);
            let col: Tile<i32, { [BM, BN] }> = columns + (j * BN).broadcast(shape![BM, BN]);
            let mut valid: Tile<bool, { [BM, BN] }> =
                le_tile(col, rows) & lt_tile(col, length.broadcast(shape![BM, BN]));
            if window > 0i32 {
                valid = valid & gt_tile(col, rows - window.broadcast(shape![BM, BN]));
            }
            let scores: Tile<f32, { [BM, BN] }> = select(valid, scores, mask_false);
            let tile_max: Tile<f32, { [BM] }> = reduce_max(scores, 1i32);
            let tile_max: Tile<f32, { [BM, 1] }> = tile_max.reshape(shape![BM, 1]);
            let next: Tile<f32, { [BM, 1] }> = max_tile(maximum, tile_max);
            // Some query rows can have an entirely masked first window tile.
            let safe_next: Tile<f32, { [BM, 1] }> = select(eq_tile(next, neginf), zeros, next);
            let p: Tile<f32, { [BM, BN] }> =
                exp2(scores - safe_next.broadcast(shape![BM, BN]), ftz::Disabled);
            let alpha: Tile<f32, { [BM, 1] }> = exp2(maximum - safe_next, ftz::Disabled);
            let mass: Tile<f32, { [BM] }> = reduce_sum(p, 1i32);
            let mass: Tile<f32, { [BM, 1] }> = mass.reshape(shape![BM, 1]);
            denominator = denominator * alpha + mass;
            acc = acc * alpha.broadcast(shape![BM, D]);
            let pt: Tile<E, { [BM, BN] }> = convert_tile(p);
            let vf: Tile<f32, { [BN, D] }> = vp.load([kh, j, 0i32]).reshape(shape![BN, D]);
            let vt: Tile<E, { [BN, D] }> = convert_tile(vf);
            acc = mma(pt, vt, acc);
            maximum = next;
        }
        let result: Tile<f32, { [BM, D] }> = true_div(acc, denominator.broadcast(shape![BM, D]));
        let result: Tile<f16, { [1, BM, D] }> = convert_tile(result.reshape(shape![1, BM, D]));
        out.store(result);
    }
}
