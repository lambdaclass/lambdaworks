use criterion::{black_box, criterion_group, criterion_main, Criterion};
use lambdaworks_crypto::hash::pedersen::Pedersen;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::fft_friendly::stark_252_prime_field::Stark252PrimeField;

use starknet_types_core::hash::Pedersen;

fn pedersen_benchmarks(c: &mut Criterion) {
    let x = FieldElement::<Stark252PrimeField>::from_hex("0x123456").unwrap();
    let y = FieldElement::<Stark252PrimeField>::from_hex("0x789101").unwrap();
    let mut group = c.benchmark_group("Pedersen Benchmark");

    // Benchmark with black_box is 0.41% faster
    group.bench_function("Hashing with black_box", |bench| {
        let pedersen = black_box(Pedersen::default());
        bench.iter(|| black_box(pedersen.hash(&x, &y)))
    });
}
criterion_group!(pedersen, pedersen_benchmarks);
criterion_main!(pedersen);
