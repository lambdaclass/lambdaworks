//! Lambdaworks BLS12-381 scalar field (Fr) addition benchmark
use lambdaworks_math::{
    elliptic_curve::short_weierstrass::curves::bls12_381::default_types::FrElement,
};

const ITERATIONS: usize = 1_000_000;

fn main() {
    let a = FrElement::from(123456789u64);
    let b = FrElement::from(987654321u64);

    let mut result = a.clone();
    for _ in 0..ITERATIONS {
        result = &result + &b;
    }
    std::hint::black_box(result);
}
