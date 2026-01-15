//! Lambdaworks BN254 scalar field (Fr) subtraction benchmark
use lambdaworks_math::{
    elliptic_curve::short_weierstrass::curves::bn_254::default_types::FrElement,
};

const ITERATIONS: usize = 1_000_000;

fn main() {
    let a = FrElement::from(123456789u64);
    let b = FrElement::from(987654321u64);

    let mut result = a.clone();
    for _ in 0..ITERATIONS {
        result = &result - &b;
    }
    std::hint::black_box(result);
}
