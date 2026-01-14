//! Lambdaworks Stark252 field inversion benchmark
use lambdaworks_math::{
    field::element::FieldElement,
    field::fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
};

const ITERATIONS: usize = 10_000;

fn main() {
    let base = FieldElement::<Stark252PrimeField>::from(123456789u64);

    for i in 0..ITERATIONS {
        let a = &base + FieldElement::<Stark252PrimeField>::from(i as u64);
        let result = a.inv().unwrap();
        std::hint::black_box(result);
    }
}
