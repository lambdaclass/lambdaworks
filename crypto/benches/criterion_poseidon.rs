use criterion::{criterion_group, criterion_main, Criterion};
use lambdaworks_crypto::hash::poseidon::shared_tests::run_tests;

fn poseidon_benchmarks(c: &mut Criterion) {
    c.bench_function("poseidon_tests", |b| b.iter(|| run_tests()));
}

criterion_group!(poseidon, poseidon_benchmarks);
criterion_main!(poseidon);
