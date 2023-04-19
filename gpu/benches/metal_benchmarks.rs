use criterion::{criterion_group, criterion_main, Criterion};

mod benchmarks;

fn run_metal_benchmarks(c: &mut Criterion) {
    benchmarks::metal::metal_bitrev_permutation_benchmarks(c);
    benchmarks::metal::metal_twiddles_gen_benchmarks(c);
    benchmarks::metal::metal_fft_benchmarks(c);
    benchmarks::metal::metal_poly_interpolate_fft_benchmarks(c);
    benchmarks::metal::metal_poly_evaluate_fft_benchmarks(c);
}

criterion_group!(benches, run_metal_benchmarks);
criterion_main!(benches);
