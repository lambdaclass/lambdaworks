//! Arkworks BN254 scalar field (Fr) multiplication benchmark with random sampling
use ark_bn254::Fr;
use ark_ff::{BigInt, PrimeField};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

const ITERATIONS: usize = 100_000;
const SEED: u64 = 42;

fn main() {
    let mut rng = StdRng::seed_from_u64(SEED);

    // Pre-generate random elements by sampling 4 u64 limbs each
    let elements: Vec<Fr> = (0..ITERATIONS)
        .map(|_| {
            let limbs: [u64; 4] = [
                rng.gen(),
                rng.gen(),
                rng.gen(),
                rng.gen(),
            ];
            Fr::from_bigint(BigInt::new(limbs)).unwrap_or(Fr::from(1u64))
        })
        .collect();

    let mut result = elements[0];
    for i in 1..ITERATIONS {
        result = result * elements[i];
    }
    std::hint::black_box(result);
}
