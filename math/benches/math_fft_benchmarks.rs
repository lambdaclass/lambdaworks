use criterion::{criterion_group, criterion_main, Criterion};

mod benchmarks;

fn run_math_fft_benchmarks(c: &mut Criterion) {
    benchmarks::fft::bitrev_permutation_benchmarks(c);
    benchmarks::fft::twiddles_benchmarks(c);
    benchmarks::fft::fft_benchmarks(c);
    benchmarks::fft::poly_interpolate_benchmarks(c);
    benchmarks::fft::poly_interpolate_fft_benchmarks(c);
}

criterion_group!(benches, run_math_fft_benchmarks);
criterion_main!(benches);
