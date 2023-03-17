use criterion::{criterion_group, criterion_main, Criterion};

mod benchmarks;

fn run_all_benchmarks(c: &mut Criterion) {
    benchmarks::metal::metal_fft_benchmarks(c);
}

criterion_group!(benches, run_all_benchmarks);
criterion_main!(benches);
