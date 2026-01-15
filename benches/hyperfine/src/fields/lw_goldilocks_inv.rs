//! Lambdaworks Goldilocks field inversion benchmark
use lambdaworks_math::{
    field::element::FieldElement,
    field::fields::u64_goldilocks_field::Goldilocks64Field,
};

const ITERATIONS: usize = 10_000;

fn main() {
    let base = FieldElement::<Goldilocks64Field>::from(123456789u64);

    for i in 0..ITERATIONS {
        let a = &base + FieldElement::<Goldilocks64Field>::from(i as u64);
        let result = a.inv().unwrap();
        std::hint::black_box(result);
    }
}
