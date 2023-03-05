use criterion::{criterion_group, criterion_main, Criterion};

mod benchmarks;

fn run_field_benchmarks(c: &mut Criterion) {
    benchmarks::field::u64_benchmark(c);
}

criterion_group!(benches, run_field_benchmarks);
criterion_main!(benches);
