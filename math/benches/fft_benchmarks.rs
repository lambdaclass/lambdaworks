use criterion::{criterion_group, criterion_main, Criterion};

mod benchmarks;

fn run_fft_benchmarks(c: &mut Criterion) {
    benchmarks::fft::fft_benchmarks(c);
}

criterion_group!(benches, run_fft_benchmarks);
criterion_main!(benches);
