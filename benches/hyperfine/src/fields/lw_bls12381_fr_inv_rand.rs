//! Lambdaworks BLS12-381 scalar field (Fr) inversion benchmark with random sampling
use lambdaworks_math::{
    elliptic_curve::short_weierstrass::curves::bls12_381::default_types::FrField,
    field::element::FieldElement,
    unsigned_integer::element::UnsignedInteger,
};

const ITERATIONS: usize = 10_000;
const SEED: u64 = 42;

fn lcg_next(state: &mut u64) -> u64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    *state
}

fn main() {
    let mut state = SEED;

    // Pre-generate random non-zero elements
    let elements: Vec<FieldElement<FrField>> = (0..ITERATIONS)
        .map(|_| {
            let limbs: [u64; 4] = [
                lcg_next(&mut state) | 1, // ensure non-zero
                lcg_next(&mut state),
                lcg_next(&mut state),
                lcg_next(&mut state),
            ];
            FieldElement::new(UnsignedInteger { limbs })
        })
        .collect();

    for elem in &elements {
        let result = elem.inv().unwrap();
        std::hint::black_box(result);
    }
}
