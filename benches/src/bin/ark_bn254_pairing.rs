//! Arkworks BN254 pairing benchmark for hyperfine
//!
//! Run with: hyperfine './target/release/bench_ark_bn254_pairing'

use ark_bn254::{Bn254, Fr, G1Projective, G2Projective};
use ark_ec::pairing::Pairing;
use ark_ec::PrimeGroup;
use std::hint::black_box;
use std::ops::Mul;

const ITERATIONS: u32 = 1000;

fn main() {
    // Generate points
    let g1 = G1Projective::generator();
    let g2 = G2Projective::generator();

    // Use a scalar to create non-trivial points
    let p = g1.mul(Fr::from(12345u64));
    let q = g2.mul(Fr::from(67890u64));

    // Run pairing iterations
    for _ in 0..ITERATIONS {
        let _ = black_box(Bn254::pairing(p, q));
    }
}
