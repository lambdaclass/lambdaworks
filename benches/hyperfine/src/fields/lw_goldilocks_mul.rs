//! Lambdaworks Goldilocks (u64) field multiplication benchmark
use lambdaworks_math::{
    field::element::FieldElement,
    field::fields::u64_goldilocks_field::Goldilocks64Field,
};

const ITERATIONS: usize = 100_000;

fn main() {
    let a = FieldElement::<Goldilocks64Field>::from(123456789u64);
    let b = FieldElement::<Goldilocks64Field>::from(987654321u64);

    let mut result = a.clone();
    for _ in 0..ITERATIONS {
        result = &result * &b;
    }
    std::hint::black_box(result);
}
