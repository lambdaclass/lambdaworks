//! Lambdaworks BN254 base field (Fq) addition benchmark
use lambdaworks_math::{
    elliptic_curve::short_weierstrass::curves::bn_254::field_extension::BN254PrimeField,
    field::element::FieldElement,
};

const ITERATIONS: usize = 1_000_000;

fn main() {
    let a = FieldElement::<BN254PrimeField>::from(123456789u64);
    let b = FieldElement::<BN254PrimeField>::from(987654321u64);

    let mut result = a.clone();
    for _ in 0..ITERATIONS {
        result = &result + &b;
    }
    std::hint::black_box(result);
}
