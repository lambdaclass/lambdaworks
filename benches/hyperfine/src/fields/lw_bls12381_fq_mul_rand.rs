//! Lambdaworks BLS12-381 base field (Fq) multiplication benchmark with random sampling
use lambdaworks_math::{
    elliptic_curve::short_weierstrass::curves::bls12_381::field_extension::BLS12381PrimeField,
    field::element::FieldElement,
    unsigned_integer::element::UnsignedInteger,
};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

const ITERATIONS: usize = 100_000;
const SEED: u64 = 42;

fn main() {
    let mut rng = StdRng::seed_from_u64(SEED);

    // Pre-generate random elements by sampling 6 u64 limbs each (384-bit field)
    let elements: Vec<FieldElement<BLS12381PrimeField>> = (0..ITERATIONS)
        .map(|_| {
            let limbs: [u64; 6] = [
                rng.gen(),
                rng.gen(),
                rng.gen(),
                rng.gen(),
                rng.gen(),
                rng.gen(),
            ];
            FieldElement::new(UnsignedInteger { limbs })
        })
        .collect();

    let mut result = elements[0].clone();
    for i in 1..ITERATIONS {
        result = &result * &elements[i];
    }
    std::hint::black_box(result);
}
