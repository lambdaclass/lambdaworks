//! Lambdaworks Stark252 field multiplication benchmark
use lambdaworks_math::{
    field::element::FieldElement,
    field::fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
};

const ITERATIONS: usize = 100_000;

fn main() {
    let a = FieldElement::<Stark252PrimeField>::from(123456789u64);
    let b = FieldElement::<Stark252PrimeField>::from(987654321u64);

    let mut result = a.clone();
    for _ in 0..ITERATIONS {
        result = &result * &b;
    }
    std::hint::black_box(result);
}
