//! Lambdaworks BLS12-381 pairing benchmark for hyperfine
//!
//! Run with: hyperfine './target/release/bench_lw_bls12381_pairing'

use lambdaworks_math::cyclic_group::IsGroup;
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::curve::BLS12381Curve;
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::pairing::BLS12381AtePairing;
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::twist::BLS12381TwistCurve;
use lambdaworks_math::elliptic_curve::traits::{IsEllipticCurve, IsPairing};
use std::hint::black_box;

const ITERATIONS: u32 = 1000;

fn main() {
    // Generate points
    let g1 = BLS12381Curve::generator();
    let g2 = BLS12381TwistCurve::generator();

    // Use a scalar to create non-trivial points
    let p = g1.operate_with_self(12345u64);
    let q = g2.operate_with_self(67890u64);

    // Run pairing iterations
    for _ in 0..ITERATIONS {
        let _ = black_box(BLS12381AtePairing::compute_batch(&[(&p, &q)]));
    }
}
