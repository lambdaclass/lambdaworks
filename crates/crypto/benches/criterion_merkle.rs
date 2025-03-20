use core::time::Duration;
use criterion::{criterion_group, criterion_main, Criterion};
use lambdaworks_crypto::merkle_tree::{
    backends::field_element::FieldElementBackend, merkle::MerkleTree,
};
use lambdaworks_math::{
    field::element::FieldElement,
    field::fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
};
use sha3::Keccak256;

type F = Stark252PrimeField;
type FE = FieldElement<F>;
type TreeBackend = FieldElementBackend<F, Keccak256, 32>;

fn merkle_tree_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("Merkle Tree");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));

    // NOTE: the values to hash don't really matter, so let's go with the easy ones.

    let unhashed_leaves: Vec<_> = core::iter::successors(Some(FE::zero()), |s| Some(s + FE::one()))
        .take((1 << 20) + 1)
        .collect();
    // `(1 << 20) + 1` exploits worst cases in terms of rounding up to powers of 2.

    group.bench_with_input(
        "build",
        unhashed_leaves.as_slice(),
        |bench, unhashed_leaves| {
            bench.iter_with_large_drop(|| MerkleTree::<TreeBackend>::build(unhashed_leaves));
        },
    );

    group.finish();
}

criterion_group!(merkle_tree, merkle_tree_benchmarks);
criterion_main!(merkle_tree);
