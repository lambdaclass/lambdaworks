use core::time::Duration;
use criterion::{criterion_group, criterion_main, Criterion};
use lambdaworks_crypto::hash::poseidon::Poseidon;
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::field_extension::BLS12381PrimeField;
use lambdaworks_math::field::element::FieldElement;

fn poseidon_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("Poseidon hash");
    type FE = FieldElement<BLS12381PrimeField>;
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));

    let unhashed_fields: Vec<_> = core::iter::successors(Some(FE::zero()), |s| Some(s + FE::one()))
        // `(1 << 20) + 1` exploits worst cases in terms of rounding up to powers of 2.
        .take((1 << 20) + 1)
        .collect();
    let poseidon: Poseidon<BLS12381PrimeField> = Poseidon::new();
    group.bench_with_input(
        "build",
        unhashed_fields.as_slice(),
        |bench, unhashed_fields| {
            bench.iter_with_large_drop(|| Poseidon::hash(&poseidon, unhashed_fields));
        },
    );
}
criterion_group!(poseidon, poseidon_benchmarks);
criterion_main!(poseidon);
