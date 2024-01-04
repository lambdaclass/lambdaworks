use criterion::{black_box, criterion_group, criterion_main, Criterion};
use lambdaworks_crypto::hash::poseidon::starknet::PoseidonCairoStark252;
use lambdaworks_crypto::hash::poseidon::Poseidon;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::fft_friendly::stark_252_prime_field::Stark252PrimeField;

fn poseidon_benchmarks(c: &mut Criterion) {
    let x = FieldElement::<Stark252PrimeField>::from_hex("0x123456").unwrap();
    let y = FieldElement::<Stark252PrimeField>::from_hex("0x789101").unwrap();
    let mut group = c.benchmark_group("Poseidon Benchmark");

    // Benchmark with black_box is 0.41% faster 
    group.bench_function("Hashing with black_box", |bench| {
        bench.iter(|| black_box(PoseidonCairoStark252::hash(&x, &y)))
    });

}
criterion_group!(poseidon, poseidon_benchmarks);
criterion_main!(poseidon);
