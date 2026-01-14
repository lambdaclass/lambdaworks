//! Arkworks BN254 G1 scalar multiplication (256-bit scalar) benchmark
use ark_bn254::{Fr, G1Projective};
use ark_ec::Group;

const ITERATIONS: usize = 1000;

fn main() {
    let g = G1Projective::generator();

    // Use a random-looking but deterministic 256-bit scalar
    let base_scalar = Fr::from(0x123456789ABCDEFu64);

    for i in 0..ITERATIONS {
        // Create a varied scalar by squaring and adding
        let scalar = base_scalar + Fr::from(i as u64);
        let result = g * scalar;
        std::hint::black_box(result);
    }
}
