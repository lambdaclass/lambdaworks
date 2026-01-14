//! Lambdaworks BabyBear field multiplication benchmark
use lambdaworks_math::{
    field::element::FieldElement,
    field::fields::fft_friendly::babybear::Babybear31PrimeField,
};

const ITERATIONS: usize = 100_000;

fn main() {
    let a = FieldElement::<Babybear31PrimeField>::from(123456789u64);
    let b = FieldElement::<Babybear31PrimeField>::from(987654321u64);

    let mut result = a.clone();
    for _ in 0..ITERATIONS {
        result = &result * &b;
    }
    std::hint::black_box(result);
}
