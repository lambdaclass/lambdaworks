//! Arkworks BLS12-381 G1 scalar multiplication benchmark for hyperfine
//!
//! Run with: hyperfine './target/release/bench_ark_bls12381_g1_scalar_mul'

use ark_bls12_381::{Fr, G1Projective};
use ark_ec::PrimeGroup;
use ark_ff::MontFp;
use std::hint::black_box;
use std::ops::Mul;

const ITERATIONS: u32 = 10000;

fn main() {
    let g1 = G1Projective::generator();

    // Use a 256-bit scalar (same value as lambdaworks benchmark)
    let scalar: Fr =
        MontFp!("52435875175126190479447740508185965837690552500527637822603658699938581184513");

    for _ in 0..ITERATIONS {
        let _ = black_box(g1.mul(scalar));
    }
}
