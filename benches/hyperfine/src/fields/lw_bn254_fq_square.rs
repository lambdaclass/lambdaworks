//! Lambdaworks BN254 base field (Fq) squaring benchmark
use lambdaworks_math::{
    elliptic_curve::short_weierstrass::curves::bn_254::field_extension::BN254PrimeField,
    field::element::FieldElement,
};

const ITERATIONS: usize = 100_000;

fn main() {
    let mut a = FieldElement::<BN254PrimeField>::from(123456789u64);

    for _ in 0..ITERATIONS {
        a = a.square();
    }
    std::hint::black_box(a);
}
