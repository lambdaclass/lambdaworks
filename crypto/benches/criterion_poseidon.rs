use criterion::{black_box, criterion_group, criterion_main, Criterion};
use lambdaworks_benches::utils::{generate_random_elements, to_lambdaworks_vec};
use lambdaworks_crypto::hash::poseidon::starknet::PoseidonCairoStark252;
use lambdaworks_crypto::hash::poseidon::Poseidon;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::fft_friendly::stark_252_prime_field::MontgomeryConfigStark252PrimeField;
use lambdaworks_math::field::fields::montgomery_backed_prime_fields::MontgomeryBackendPrimeField;

type F = MontgomeryBackendPrimeField<MontgomeryConfigStark252PrimeField, 4>;
fn generate_points() -> Vec<FieldElement<F>> {
    let random_points = generate_random_elements(2);
    to_lambdaworks_vec(&random_points)
}
fn poseidon_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("Poseidon Benchmark".to_string());
    let points = generate_points();
    group.bench_function("Merkle hashing ", |bench| {
        bench.iter(|| black_box(PoseidonCairoStark252::hash(&points[0], &points[1])))
    });
}
criterion_group!(poseidon, poseidon_benchmarks);
criterion_main!(poseidon);
