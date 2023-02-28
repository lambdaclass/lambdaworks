use criterion::{criterion_group, criterion_main, Criterion};
#[cfg(feature = "benchmark_flamegraph")]
use pprof::criterion::PProfProfiler;

mod benchmarks;

fn run_field_benchmarks(c: &mut Criterion) {
    benchmarks::field::u64_benchmark(c);
}

#[cfg(feature = "benchmark_flamegraph")]
criterion_group! {
    name = benches;
    config = Criterion::default().with_profiler(PProfProfiler::new(100, pprof::criterion::Output::Flamegraph(None)));
    targets = run_field_benchmarks
}
#[cfg(all(not(feature = "benchmark_flamegraph")))]
criterion_group!(benches, run_field_benchmarks);
criterion_main!(benches);
