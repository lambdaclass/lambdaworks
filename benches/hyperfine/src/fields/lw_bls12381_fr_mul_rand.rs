//! Lambdaworks BLS12-381 scalar field (Fr) multiplication benchmark with random sampling
use lambdaworks_math::{
    elliptic_curve::short_weierstrass::curves::bls12_381::default_types::FrField,
    field::element::FieldElement,
    unsigned_integer::element::UnsignedInteger,
};

const ITERATIONS: usize = 100_000;
const SEED: u64 = 42;

// Simple LCG for reproducible random numbers (same as arkworks benchmark)
fn lcg_next(state: &mut u64) -> u64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    *state
}

fn main() {
    let mut state = SEED;

    // Pre-generate random elements by sampling 4 u64 limbs each
    // Use modular reduction to ensure valid field elements
    let elements: Vec<FieldElement<FrField>> = (0..ITERATIONS)
        .map(|_| {
            let limbs: [u64; 4] = [
                lcg_next(&mut state),
                lcg_next(&mut state),
                lcg_next(&mut state),
                lcg_next(&mut state),
            ];
            // Create element and reduce mod p (new() handles this)
            FieldElement::new(UnsignedInteger { limbs })
        })
        .collect();

    // Multiply all elements together
    let mut result = elements[0].clone();
    for i in 1..ITERATIONS {
        result = &result * &elements[i];
    }
    std::hint::black_box(result);
}
