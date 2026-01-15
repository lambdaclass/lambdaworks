//! Lambdaworks BN254 scalar field (Fr) multiplication benchmark
use lambdaworks_math::{
    elliptic_curve::short_weierstrass::curves::bn_254::default_types::FrField,
    field::element::FieldElement,
};

const ITERATIONS: usize = 100_000;

fn main() {
    let a = FieldElement::<FrField>::from(123456789u64);
    let b = FieldElement::<FrField>::from(987654321u64);

    let mut result = a.clone();
    for _ in 0..ITERATIONS {
        result = &result * &b;
    }
    std::hint::black_box(result);
}
