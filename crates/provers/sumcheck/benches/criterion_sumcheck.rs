mod sumcheck;

use criterion::{criterion_group, criterion_main, Criterion};
use pprof::criterion::{Output, PProfProfiler};
use sumcheck::{
    multilinear_benchmarks, optimized_prover_benchmarks, prover_benchmarks,
    quadratic_prover_benchmarks, verifier_benchmarks,
};

criterion_group!(
    name = sumcheck_benches;
    config = Criterion::default().with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = prover_benchmarks, optimized_prover_benchmarks, quadratic_prover_benchmarks, verifier_benchmarks, multilinear_benchmarks
);
criterion_main!(sumcheck_benches);
