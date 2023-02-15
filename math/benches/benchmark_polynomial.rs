use criterion::{black_box, criterion_group, criterion_main, Criterion};
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::u64_prime_field::U64PrimeField;
use lambdaworks_math::polynomial::Polynomial;

const ORDER: u64 = 293;
pub type F = U64PrimeField<ORDER>;
pub type FE = FieldElement<F>;

pub fn criterion_benchmark(c: &mut Criterion) {
	let p = Polynomial::new(&[FE::new(10), FE::new(5), -FE::new(22), FE::new(13),FE::new(0), FE::new(5),FE::new(2), -FE::new(34),]);
    c.bench_function("evaluate poly", |b| b.iter(|| p.evaluate(&FE::new(black_box(20)))));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);