//! Differential fuzzer for Metal GPU Poseidon2 Merkle tree vs CPU.
//!
//! Tests three operations with each fuzzed input:
//! 1. `hash_leaves_gpu` vs CPU `hash_single` (leaf hashing)
//! 2. `hash_level_gpu` vs CPU `compress` (pair compression)
//! 3. Full `MerkleTree::build` root comparison (only if GPU proved reliable)
//!
//! GPU resource exhaustion is handled gracefully: each call creates a new
//! DynamicMetalState (device, command queue, pipeline). Under rapid fuzzing
//! this can exhaust GPU resources, causing silent failures (zeros in output
//! buffer) or explicit errors. We only assert on correctness when the GPU
//! call succeeds and produces non-trivial results.

#![no_main]

use libfuzzer_sys::fuzz_target;

use lambdaworks_crypto::{
    hash::poseidon2::Poseidon2,
    merkle_tree::{
        backends::{metal::MetalPoseidon2Backend, poseidon2::Poseidon2Backend},
        merkle::MerkleTree,
    },
};
use lambdaworks_math::field::{
    element::FieldElement, fields::u64_goldilocks_field::Goldilocks64Field,
};

type Fp = FieldElement<Goldilocks64Field>;
type Digest = [Fp; 2];

/// Check if GPU results look like a silent failure (all zeros).
/// hash_single and compress never return [0, 0] for any valid input
/// because the permutation with non-zero domain tags always produces non-zero output.
fn is_silent_gpu_failure(digests: &[Digest]) -> bool {
    let zero = [Fp::zero(), Fp::zero()];
    digests.iter().all(|d| *d == zero)
}

fuzz_target!(|data: Vec<u64>| {
    if data.is_empty() {
        return;
    }

    let leaves: Vec<Fp> = data.iter().map(|&v| Fp::from(v)).collect();

    // --- 1. hash_leaves_gpu vs CPU hash_single ---
    // Pad to >= 64 elements to exercise the GPU path
    let mut padded = leaves.clone();
    while padded.len() < 64 {
        padded.push(Fp::from(0u64));
    }

    let cpu_leaves: Vec<Digest> = padded.iter().map(Poseidon2::hash_single).collect();

    // Track whether the GPU is working correctly this iteration
    let mut gpu_reliable = false;

    if let Ok(gpu_leaves) = MetalPoseidon2Backend::hash_leaves_gpu(&padded) {
        if !is_silent_gpu_failure(&gpu_leaves) {
            for (i, (gpu, cpu)) in gpu_leaves.iter().zip(cpu_leaves.iter()).enumerate() {
                assert_eq!(
                    gpu, cpu,
                    "hash_single mismatch at index {}: GPU={:?} CPU={:?}",
                    i, gpu, cpu
                );
            }
            gpu_reliable = true;
        }
    }

    // --- 2. hash_level_gpu vs CPU compress ---
    let mut nodes = cpu_leaves;
    if nodes.len() % 2 != 0 {
        nodes.push([Fp::from(0u64), Fp::from(0u64)]);
    }
    while nodes.len() < 64 {
        nodes.push([Fp::from(0u64), Fp::from(0u64)]);
        nodes.push([Fp::from(0u64), Fp::from(0u64)]);
    }

    let pair_count = nodes.len() / 2;
    let cpu_level: Vec<Digest> = (0..pair_count)
        .map(|i| Poseidon2::compress(&nodes[i * 2], &nodes[i * 2 + 1]))
        .collect();

    if let Ok(gpu_level) = MetalPoseidon2Backend::hash_level_gpu(&nodes) {
        if !is_silent_gpu_failure(&gpu_level) {
            for (i, (gpu, cpu)) in gpu_level.iter().zip(cpu_level.iter()).enumerate() {
                assert_eq!(
                    gpu, cpu,
                    "compress mismatch at pair {}: GPU={:?} CPU={:?}",
                    i, gpu, cpu
                );
            }
        }
    }

    // --- 3. Full MerkleTree build: compare roots ---
    // Only compare when GPU proved reliable in step 1, since MerkleTree::build
    // calls hash_leaves_gpu internally and won't fallback on silent GPU failures.
    if gpu_reliable {
        let metal_tree = MerkleTree::<MetalPoseidon2Backend>::build(&leaves);
        let cpu_tree = MerkleTree::<Poseidon2Backend>::build(&leaves);

        match (metal_tree, cpu_tree) {
            (Some(metal), Some(cpu)) => {
                assert_eq!(
                    metal.root, cpu.root,
                    "Merkle root mismatch for {} leaves",
                    leaves.len()
                );
            }
            (None, None) => {}
            (metal, cpu) => {
                panic!(
                    "Build disagreement for {} leaves: Metal={} CPU={}",
                    leaves.len(),
                    metal.is_some(),
                    cpu.is_some()
                );
            }
        }
    }
});
