//! Lambdaworks BN254 scalar field (Fr) squaring benchmark
use lambdaworks_math::{
    elliptic_curve::short_weierstrass::curves::bn_254::default_types::FrElement,
};

const ITERATIONS: usize = 100_000;

fn main() {
    let mut a = FrElement::from(123456789u64);

    for _ in 0..ITERATIONS {
        a = a.square();
    }
    std::hint::black_box(a);
}
