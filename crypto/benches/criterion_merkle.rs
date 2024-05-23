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

    // Large number of leaves benchmark
    let unhashed_leaves: Vec<_> = core::iter::successors(Some(FE::zero()), |s| Some(s + FE::one()))
        .take((1 << 20) + 1)
        .collect();

    group.bench_with_input(
        "build_large",
        unhashed_leaves.as_slice(),
        |bench, unhashed_leaves| {
            bench.iter_with_large_drop(|| MerkleTree::<TreeBackend>::build(unhashed_leaves));
        },
    );

    // Single element benchmark
    // This is a special case that should be optimized
    let single_leaf: Vec<FE> = vec![FE::one()];

    group.bench_with_input(
        "build_single",
        single_leaf.as_slice(),
        |bench, single_leaf| {
            bench.iter_with_large_drop(|| MerkleTree::<TreeBackend>::build(single_leaf));
        },
    );
    group.finish();
}

criterion_group!(merkle_tree, merkle_tree_benchmarks);
criterion_main!(merkle_tree);
