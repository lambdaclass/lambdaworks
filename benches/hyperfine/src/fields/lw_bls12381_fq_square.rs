//! Lambdaworks BLS12-381 base field (Fq) squaring benchmark
use lambdaworks_math::{
    elliptic_curve::short_weierstrass::curves::bls12_381::field_extension::BLS12381PrimeField,
    field::element::FieldElement,
};

const ITERATIONS: usize = 100_000;

fn main() {
    let mut a = FieldElement::<BLS12381PrimeField>::from(123456789u64);

    for _ in 0..ITERATIONS {
        a = a.square();
    }
    std::hint::black_box(a);
}
