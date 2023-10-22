use criterion::{criterion_group, criterion_main, Criterion};
use pprof::criterion::{Output, PProfProfiler};

mod fields;
use fields::{
    stark252::starkfield_ops_benchmarks, u64_goldilocks::u64_goldilocks_ops_benchmarks,
    u64_goldilocks_montgomery::u64_goldilocks_montgomery_ops_benchmarks,
};
use fields::mersenne31::mersenne31_ops_benchmarks;

criterion_group!(
    name = field_benches;
    config = Criterion::default().with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = starkfield_ops_benchmarks, mersenne31_ops_benchmarks, u64_goldilocks_ops_benchmarks, u64_goldilocks_montgomery_ops_benchmarks
);
criterion_main!(field_benches);
