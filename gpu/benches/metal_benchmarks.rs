use criterion::{criterion_group, criterion_main, Criterion};

mod benchmarks;

fn run_metal_benchmarks(c: &mut Criterion) {
    benchmarks::metal::metal_fft_twiddles_benchmarks(c);
    benchmarks::metal::metal_fft_benchmarks(c);
}

criterion_group!(benches, run_metal_benchmarks);
criterion_main!(benches);
