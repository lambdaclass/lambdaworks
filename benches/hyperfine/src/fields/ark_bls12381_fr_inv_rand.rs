//! Arkworks BLS12-381 scalar field (Fr) inversion benchmark with random sampling
use ark_bls12_381::Fr;
use ark_ff::{BigInt, Field, PrimeField};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

const ITERATIONS: usize = 10_000;
const SEED: u64 = 42;

fn main() {
    let mut rng = StdRng::seed_from_u64(SEED);

    // Pre-generate random elements by sampling 4 u64 limbs each
    // Ensure non-zero by OR-ing with 1
    let elements: Vec<Fr> = (0..ITERATIONS)
        .map(|_| {
            let limbs: [u64; 4] = [
                rng.gen::<u64>() | 1, // ensure non-zero
                rng.gen(),
                rng.gen(),
                rng.gen(),
            ];
            Fr::from_bigint(BigInt::new(limbs)).unwrap_or(Fr::from(1u64))
        })
        .collect();

    for elem in &elements {
        let result = elem.inverse().unwrap();
        std::hint::black_box(result);
    }
}
