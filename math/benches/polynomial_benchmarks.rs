use criterion::{criterion_group, criterion_main, Criterion};

mod benchmarks;

fn run_polynomial_benchmarks(c: &mut Criterion) {
    benchmarks::polynomial::polynomial_benchmark(c);
}

criterion_group!(benches, run_polynomial_benchmarks);
criterion_main!(benches);
