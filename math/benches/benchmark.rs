use criterion::{criterion_group, criterion_main, Criterion};

mod benchmark_msm;

pub fn run_msm_benchmarks(c: &mut Criterion) {
    benchmark_msm::msm_with_size(c, 1);
    benchmark_msm::msm_with_size(c, 10);
    benchmark_msm::msm_with_size(c, 100);
    benchmark_msm::msm_with_size(c, 1000);
}

criterion_group!(benches, run_msm_benchmarks);

criterion_main!(benches);
