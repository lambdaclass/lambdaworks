//! Arkworks BLS12-381 base field (Fq) multiplication benchmark with random sampling
use ark_bls12_381::Fq;
use ark_ff::{BigInt, PrimeField};

const ITERATIONS: usize = 100_000;
const SEED: u64 = 42;

fn lcg_next(state: &mut u64) -> u64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    *state
}

fn main() {
    let mut state = SEED;

    // Pre-generate random elements by sampling 6 u64 limbs each (384-bit field)
    let elements: Vec<Fq> = (0..ITERATIONS)
        .map(|_| {
            let limbs: [u64; 6] = [
                lcg_next(&mut state),
                lcg_next(&mut state),
                lcg_next(&mut state),
                lcg_next(&mut state),
                lcg_next(&mut state),
                lcg_next(&mut state),
            ];
            Fq::from_bigint(BigInt::new(limbs)).unwrap_or(Fq::from(1u64))
        })
        .collect();

    let mut result = elements[0];
    for i in 1..ITERATIONS {
        result = result * elements[i];
    }
    std::hint::black_box(result);
}
