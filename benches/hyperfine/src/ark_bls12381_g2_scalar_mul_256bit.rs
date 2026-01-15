//! Arkworks BLS12-381 G2 scalar multiplication (256-bit scalar) benchmark
use ark_bls12_381::{Fr, G2Projective};
use ark_ec::Group;
use ark_ff::BigInt;

const ITERATIONS: usize = 500;

fn main() {
    let g = G2Projective::generator();

    // 256-bit scalar
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
