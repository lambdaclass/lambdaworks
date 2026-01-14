//! Lambdaworks Stark252 field squaring benchmark
use lambdaworks_math::{
    field::element::FieldElement,
    field::fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
};

const ITERATIONS: usize = 100_000;

fn main() {
    let mut a = FieldElement::<Stark252PrimeField>::from(123456789u64);

    for _ in 0..ITERATIONS {
        a = a.square();
    }
    std::hint::black_box(a);
}
