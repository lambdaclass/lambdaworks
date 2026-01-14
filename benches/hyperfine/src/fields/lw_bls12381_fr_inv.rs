//! Lambdaworks BLS12-381 scalar field (Fr) inversion benchmark
use lambdaworks_math::{
    elliptic_curve::short_weierstrass::curves::bls12_381::default_types::FrElement,
};

const ITERATIONS: usize = 10_000;

fn main() {
    let base = FrElement::from(123456789u64);

    for i in 0..ITERATIONS {
        let a = &base + FrElement::from(i as u64);
        let result = a.inv().unwrap();
        std::hint::black_box(result);
    }
}
