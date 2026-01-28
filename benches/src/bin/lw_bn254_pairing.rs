//! Lambdaworks BN254 pairing benchmark for hyperfine
//!
//! Run with: hyperfine './target/release/bench_lw_bn254_pairing'

use lambdaworks_math::cyclic_group::IsGroup;
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bn_254::curve::BN254Curve;
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bn_254::pairing::BN254AtePairing;
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bn_254::twist::BN254TwistCurve;
use lambdaworks_math::elliptic_curve::traits::{IsEllipticCurve, IsPairing};
use std::hint::black_box;

const ITERATIONS: u32 = 1000;

fn main() {
    // Generate points
    let g1 = BN254Curve::generator();
    let g2 = BN254TwistCurve::generator();

    // Use a scalar to create non-trivial points
    let p = g1.operate_with_self(12345u64);
    let q = g2.operate_with_self(67890u64);

    // Run pairing iterations
    for _ in 0..ITERATIONS {
        let _ = black_box(BN254AtePairing::compute_batch(&[(&p, &q)]));
    }
}
