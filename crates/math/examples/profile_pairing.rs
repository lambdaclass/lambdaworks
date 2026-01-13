//! Pairing benchmark for lambdaworks BLS12-381
use lambdaworks_math::{
    cyclic_group::IsGroup,
    elliptic_curve::{
        short_weierstrass::curves::bls12_381::{
            curve::BLS12381Curve, pairing::BLS12381AtePairing, twist::BLS12381TwistCurve,
        },
        traits::{IsEllipticCurve, IsPairing},
    },
};
use std::hint::black_box;
use std::time::Instant;

fn main() {
    let g1 = BLS12381Curve::generator();
    let g2 = BLS12381TwistCurve::generator();
    let p = g1.operate_with_self(12345u64);
    let q = g2.operate_with_self(67890u64);

    // Warmup
    for _ in 0..10 {
        black_box(BLS12381AtePairing::compute(black_box(&p), black_box(&q)));
    }

    // Benchmark full pairing
    let iterations = 100;
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = black_box(BLS12381AtePairing::compute(black_box(&p), black_box(&q)));
    }
    let elapsed = start.elapsed();
    println!("Full Pairing: {:?} per op", elapsed / iterations as u32);
}
