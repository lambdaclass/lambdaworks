mod polynomials;
use criterion::{criterion_group, criterion_main, Criterion};
use polynomials::{
    dense_multilinear_poly::dense_multilinear_polynomial_benchmarks,
    polynomial::polynomial_benchmarks,
    sparse_multilinear_poly::sparse_multilinear_polynomial_benchmarks,
};
use pprof::criterion::{Output, PProfProfiler};

criterion_group!(
    name = polynomial;
    config = Criterion::default().with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = polynomial_benchmarks, dense_multilinear_polynomial_benchmarks, sparse_multilinear_polynomial_benchmarks);
criterion_main!(polynomial);
