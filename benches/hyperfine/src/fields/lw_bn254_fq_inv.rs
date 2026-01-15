//! Lambdaworks BN254 base field (Fq) inversion benchmark
use lambdaworks_math::{
    elliptic_curve::short_weierstrass::curves::bn_254::field_extension::BN254PrimeField,
    field::element::FieldElement,
};

const ITERATIONS: usize = 10_000;

fn main() {
    let base = FieldElement::<BN254PrimeField>::from(123456789u64);

    for i in 0..ITERATIONS {
        let a = &base + FieldElement::<BN254PrimeField>::from(i as u64);
        let result = a.inv().unwrap();
        std::hint::black_box(result);
    }
}
