//! Lambdaworks BLS12-381 base field (Fq) multiplication benchmark
use lambdaworks_math::{
    elliptic_curve::short_weierstrass::curves::bls12_381::field_extension::BLS12381PrimeField,
    field::element::FieldElement,
};

const ITERATIONS: usize = 100_000;

fn main() {
    let a = FieldElement::<BLS12381PrimeField>::from(123456789u64);
    let b = FieldElement::<BLS12381PrimeField>::from(987654321u64);

    let mut result = a.clone();
    for _ in 0..ITERATIONS {
        result = &result * &b;
    }
    std::hint::black_box(result);
}
