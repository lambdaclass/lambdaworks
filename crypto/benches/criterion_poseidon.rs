use criterion::{criterion_group, criterion_main, Criterion};
use lambdaworks_crypto::hash::poseidon::starknet::PoseidonCairoStark252;
use lambdaworks_crypto::hash::poseidon::Poseidon;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::fft_friendly::stark_252_prime_field::Stark252PrimeField;
use std::time::Duration;

type F = Stark252PrimeField;
type FE = FieldElement<F>;
fn poseidon_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("Poseidon batch");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));
    let points_x: Vec<_> = core::iter::successors(Some(FE::zero()), |s| Some(s + FE::one()))
        // `(1 << 20) + 1` exploits worst cases in terms of rounding up to powers of 2.
        .take((1 << 20) + 1)
        .collect();
    group.bench_with_input("build", points_x.as_slice(), |bench, points_x| {
        bench.iter_with_large_drop(|| PoseidonCairoStark252::hash_many(points_x));
    });
}
criterion_group!(poseidon, poseidon_benchmarks);
criterion_main!(poseidon);
