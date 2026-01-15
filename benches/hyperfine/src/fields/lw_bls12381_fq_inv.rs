//! Lambdaworks BLS12-381 base field (Fq) inversion benchmark
use lambdaworks_math::{
    elliptic_curve::short_weierstrass::curves::bls12_381::field_extension::BLS12381PrimeField,
    field::element::FieldElement,
};

const ITERATIONS: usize = 10_000;

fn main() {
    let base = FieldElement::<BLS12381PrimeField>::from(123456789u64);

    for i in 0..ITERATIONS {
        let a = &base + FieldElement::<BLS12381PrimeField>::from(i as u64);
        let result = a.inv().unwrap();
        std::hint::black_box(result);
    }
}
