use criterion::{black_box, criterion_group, criterion_main, Criterion};
use lambdaworks_crypto::hash::poseidon::starknet::PoseidonCairoStark252;
use lambdaworks_crypto::hash::poseidon::Poseidon;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::fft_friendly::stark_252_prime_field::Stark252PrimeField;
use lambdaworks_math::traits::ByteConversion;
use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;

fn poseidon_benchmarks(c: &mut Criterion) {
    let mut rng = ChaCha8Rng::seed_from_u64(2);
    let mut felt1: [u8; 32] = Default::default();
    rng.fill_bytes(&mut felt1);
    let mut felt2: [u8; 32] = Default::default();
    rng.fill_bytes(&mut felt2);

    let x = FieldElement::<Stark252PrimeField>::from_bytes_be(&felt1).unwrap();
    let y = FieldElement::<Stark252PrimeField>::from_bytes_be(&felt2).unwrap();
    let mut group = c.benchmark_group("Poseidon Benchmark");

    group.bench_function("Hashing with black_box", |bench| {
        bench.iter(|| black_box(PoseidonCairoStark252::hash(&x, &y)))
    });
}
criterion_group!(poseidon, poseidon_benchmarks);
criterion_main!(poseidon);
