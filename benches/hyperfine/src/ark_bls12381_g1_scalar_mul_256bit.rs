//! Arkworks BLS12-381 G1 scalar multiplication (256-bit scalar) benchmark
use ark_bls12_381::{Fr, G1Projective};
use ark_ec::Group;
use ark_ff::BigInt;

const ITERATIONS: usize = 1000;

fn main() {
    let g = G1Projective::generator();

    // 256-bit scalar (same value as lambdaworks benchmark)
    let base_scalar = Fr::from(BigInt::new([
        0xffffffff00000001,
        0xfffe5bfeffffffff,
        0x53bda402,
        0x73eda753299d7d48,
    ]));

    for i in 0..ITERATIONS {
        let scalar = base_scalar + Fr::from(i as u64);
        let result = g * scalar;
        std::hint::black_box(result);
    }
}
