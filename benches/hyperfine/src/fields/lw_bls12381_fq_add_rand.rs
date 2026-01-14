//! Lambdaworks BLS12-381 base field (Fq) addition benchmark with random sampling
use lambdaworks_math::{
    elliptic_curve::short_weierstrass::curves::bls12_381::field_extension::BLS12381PrimeField,
    field::element::FieldElement,
    unsigned_integer::element::UnsignedInteger,
};

const ITERATIONS: usize = 1_000_000;
const SEED: u64 = 42;

fn lcg_next(state: &mut u64) -> u64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    *state
}

fn main() {
    let mut state = SEED;
    let elements: Vec<FieldElement<BLS12381PrimeField>> = (0..ITERATIONS)
        .map(|_| {
            let limbs: [u64; 6] = [lcg_next(&mut state), lcg_next(&mut state), lcg_next(&mut state),
                                   lcg_next(&mut state), lcg_next(&mut state), lcg_next(&mut state)];
            FieldElement::new(UnsignedInteger { limbs })
        })
        .collect();

    let mut result = elements[0].clone();
    for i in 1..ITERATIONS {
        result = &result + &elements[i];
    }
    std::hint::black_box(result);
}
