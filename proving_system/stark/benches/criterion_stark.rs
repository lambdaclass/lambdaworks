use criterion::{criterion_group, criterion_main, Criterion};

mod functions;
mod util;

pub fn proof_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("STARK");
    group.sample_size(10);

    group.bench_function("Simple Fibonacci", |bench| {
        bench.iter(functions::stark::prove_fib);
    });

    group.bench_function("2 column Fibonacci", |bench| {
        bench.iter(functions::stark::prove_fib_2_cols);
    });

    group.bench_function("Fibonacci F17", |bench| {
        bench.iter(functions::stark::prove_fib17);
    });

    group.bench_function("Quadratic AIR", |bench| {
        bench.iter(functions::stark::prove_quadratic);
    });
}

criterion_group!(benches, proof_benchmarks);
criterion_main!(benches);
