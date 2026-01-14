//! Arkworks BN254 scalar field (Fr) inversion benchmark with random sampling
use ark_bn254::Fr;
use ark_ff::{BigInt, Field, PrimeField};

const ITERATIONS: usize = 10_000;
const SEED: u64 = 42;

fn lcg_next(state: &mut u64) -> u64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    *state
}

fn main() {
    let mut state = SEED;
    let elements: Vec<Fr> = (0..ITERATIONS)
        .map(|_| {
            let limbs: [u64; 4] = [lcg_next(&mut state) | 1, lcg_next(&mut state), lcg_next(&mut state), lcg_next(&mut state)];
            Fr::from_bigint(BigInt::new(limbs)).unwrap_or(Fr::from(1u64))
        })
        .collect();

    for elem in &elements {
        let result = elem.inverse().unwrap();
        std::hint::black_box(result);
    }
}
