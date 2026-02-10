use core::time::Duration;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use lambdaworks_crypto::{
    hash::poseidon::starknet::PoseidonCairoStark252,
    merkle_tree::{backends::field_element::TreePoseidon, merkle::MerkleTree},
};
use lambdaworks_math::field::{
    element::FieldElement, fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
};

#[cfg(feature = "cuda")]
use lambdaworks_crypto::merkle_tree::cuda::CudaPoseidonBackend;

type F = Stark252PrimeField;
type FE = FieldElement<F>;
type CpuBackend = TreePoseidon<PoseidonCairoStark252>;

fn generate_leaves(n: usize) -> Vec<FE> {
    core::iter::successors(Some(FE::zero()), |s| Some(s + FE::one()))
        .take(n)
        .collect()
}

fn merkle_cpu_vs_cuda(c: &mut Criterion) {
    let mut group = c.benchmark_group("Merkle Tree Poseidon");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(10));

    for exp in [10, 14, 18, 20] {
        let n = 1usize << exp;
        let leaves = generate_leaves(n);

        group.bench_with_input(
            BenchmarkId::new("CPU", format!("2^{exp}")),
            &leaves,
            |b, l| {
                b.iter_with_large_drop(|| MerkleTree::<CpuBackend>::build(l));
            },
        );

        #[cfg(feature = "cuda")]
        group.bench_with_input(
            BenchmarkId::new("CUDA", format!("2^{exp}")),
            &leaves,
            |b, l| {
                b.iter_with_large_drop(|| MerkleTree::<CudaPoseidonBackend>::build(l));
            },
        );

        #[cfg(feature = "cuda")]
        if CudaPoseidonBackend::is_cuda_available() {
            group.bench_with_input(
                BenchmarkId::new("CUDA-full-tree", format!("2^{exp}")),
                &leaves,
                |b, l| {
                    b.iter_with_large_drop(|| CudaPoseidonBackend::build_tree_cuda(l));
                },
            );
        }
    }

    group.finish();
}

criterion_group!(merkle_cuda, merkle_cpu_vs_cuda);
criterion_main!(merkle_cuda);
