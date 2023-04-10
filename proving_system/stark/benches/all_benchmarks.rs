use criterion::{criterion_group, criterion_main, Criterion};

pub mod benchmarks;

fn run_all_benchmarks(c: &mut Criterion) {
    benchmarks::stark::proof_benchmark(c);
}

criterion_group!(benches, run_all_benchmarks);
criterion_main!(benches);
