//! Lambdaworks Goldilocks field squaring benchmark
use lambdaworks_math::{
    field::element::FieldElement,
    field::fields::u64_goldilocks_field::Goldilocks64Field,
};

const ITERATIONS: usize = 100_000;

fn main() {
    let mut a = FieldElement::<Goldilocks64Field>::from(123456789u64);

    for _ in 0..ITERATIONS {
        a = a.square();
    }
    std::hint::black_box(a);
}
