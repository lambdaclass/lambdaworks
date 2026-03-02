//! Metal GPU-accelerated Merkle tree backend using Poseidon2 hash for Goldilocks field.

use crate::hash::poseidon2::{Fp, Poseidon2};
use crate::merkle_tree::traits::IsMerkleTreeBackend;
use lambdaworks_gpu::metal::abstractions::{errors::MetalError, state::DynamicMetalState};

/// Metal shader source code for Poseidon2 hash
const POSEIDON2_SHADER: &str = include_str!("shaders/poseidon2.metal");

/// 128-bit digest: two Goldilocks field elements
type Digest = [Fp; 2];

/// Metal GPU-accelerated Merkle tree backend using Poseidon2 hash for Goldilocks field.
///
/// Uses Apple's Metal framework to accelerate Merkle tree construction.
/// Both leaf hashing ([`hash_leaves`](IsMerkleTreeBackend::hash_leaves)) and internal level
/// compression ([`hash_level`](IsMerkleTreeBackend::hash_level)) are GPU-accelerated.
/// Falls back to CPU for small inputs or when GPU is unavailable.
#[derive(Clone, Default)]
pub struct MetalPoseidon2Backend;

impl MetalPoseidon2Backend {
    /// Hash leaves using GPU acceleration.
    ///
    /// Each leaf is a single Goldilocks field element. The GPU computes
    /// `hash_single(leaf)` for each leaf, producing a 2-element digest.
    ///
    /// Falls back to CPU for fewer than 64 leaves.
    pub fn hash_leaves_gpu(leaves: &[Fp]) -> Result<Vec<Digest>, MetalError> {
        if leaves.is_empty() {
            return Ok(Vec::new());
        }

        // CPU fallback for small inputs
        if leaves.len() < 64 {
            return Ok(leaves.iter().map(Poseidon2::hash_single).collect());
        }

        // Convert leaves to raw u64 values (no Montgomery form needed for Goldilocks)
        let gpu_input: Vec<u64> = leaves.iter().map(|fe| *fe.value()).collect();

        // Creates a new Metal state per call, recompiling the shader from source each time.
        // Same pattern as the Stark252 backend. Caching the compiled pipeline would require
        // architectural changes (thread-local or Arc<Mutex<...>>) — a future optimization.
        let mut state = DynamicMetalState::new()?;
        state.load_library(POSEIDON2_SHADER)?;

        let input_buffer = state.alloc_buffer_with_data(&gpu_input)?;
        // Output: 2 u64 values per leaf
        let output_buffer = state.alloc_buffer(leaves.len() * 2 * std::mem::size_of::<u64>())?;
        let count = [leaves.len() as u32];
        let count_buffer = state.alloc_buffer_with_data(&count)?;

        let max_threads = state.prepare_pipeline("hash_single_kernel")?;
        state.execute_compute(
            "hash_single_kernel",
            &[&input_buffer, &output_buffer, &count_buffer],
            leaves.len() as u64,
            max_threads,
        )?;

        let results: Vec<u64> = unsafe { state.read_buffer(&output_buffer, leaves.len() * 2) };

        Ok(results
            .chunks_exact(2)
            .map(|chunk| [Fp::from(chunk[0]), Fp::from(chunk[1])])
            .collect())
    }

    /// Compress pairs of digests using GPU acceleration.
    ///
    /// Takes `N` digests and produces `N/2` parent digests via `compress(left, right)`.
    ///
    /// Falls back to CPU for fewer than 32 pairs.
    ///
    /// Used by the [`IsMerkleTreeBackend::hash_level`] override to GPU-accelerate
    /// internal tree level construction. Can also be called directly.
    pub fn hash_level_gpu(nodes: &[Digest]) -> Result<Vec<Digest>, MetalError> {
        if nodes.is_empty() {
            return Ok(Vec::new());
        }

        if nodes.len() % 2 != 0 {
            return Err(MetalError::InvalidInputSize {
                expected: nodes.len() + 1,
                actual: nodes.len(),
            });
        }

        let pair_count = nodes.len() / 2;

        // CPU fallback for small inputs
        if pair_count < 32 {
            let mut results = Vec::with_capacity(pair_count);
            for i in 0..pair_count {
                results.push(Poseidon2::compress(&nodes[i * 2], &nodes[i * 2 + 1]));
            }
            return Ok(results);
        }

        // Flatten digests to raw u64: each pair = [left0, left1, right0, right1]
        let gpu_input: Vec<u64> = nodes
            .iter()
            .flat_map(|d| [*d[0].value(), *d[1].value()])
            .collect();

        // Per-call Metal state creation (see hash_leaves_gpu for rationale).
        let mut state = DynamicMetalState::new()?;
        state.load_library(POSEIDON2_SHADER)?;

        let input_buffer = state.alloc_buffer_with_data(&gpu_input)?;
        // Output: 2 u64 values per pair
        let output_buffer = state.alloc_buffer(pair_count * 2 * std::mem::size_of::<u64>())?;
        let count = [pair_count as u32];
        let count_buffer = state.alloc_buffer_with_data(&count)?;

        let max_threads = state.prepare_pipeline("compress_kernel")?;
        state.execute_compute(
            "compress_kernel",
            &[&input_buffer, &output_buffer, &count_buffer],
            pair_count as u64,
            max_threads,
        )?;

        let results: Vec<u64> = unsafe { state.read_buffer(&output_buffer, pair_count * 2) };

        Ok(results
            .chunks_exact(2)
            .map(|chunk| [Fp::from(chunk[0]), Fp::from(chunk[1])])
            .collect())
    }
}

impl IsMerkleTreeBackend for MetalPoseidon2Backend {
    type Node = Digest;
    type Data = Fp;

    fn hash_data(leaf: &Self::Data) -> Self::Node {
        Poseidon2::hash_single(leaf)
    }

    fn hash_leaves(unhashed_leaves: &[Self::Data]) -> Vec<Self::Node> {
        match Self::hash_leaves_gpu(unhashed_leaves) {
            Ok(hashed) => hashed,
            Err(_) => unhashed_leaves.iter().map(Poseidon2::hash_single).collect(),
        }
    }

    fn hash_new_parent(left: &Self::Node, right: &Self::Node) -> Self::Node {
        Poseidon2::compress(left, right)
    }

    fn hash_level(children: &[Self::Node]) -> Vec<Self::Node> {
        match Self::hash_level_gpu(children) {
            Ok(parents) => parents,
            Err(_) => children
                .chunks_exact(2)
                .map(|pair| Poseidon2::compress(&pair[0], &pair[1]))
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::merkle_tree::backends::poseidon2::Poseidon2Backend;
    use crate::merkle_tree::merkle::MerkleTree;

    #[test]
    fn test_hash_leaves_gpu_matches_cpu() {
        // 128 elements: exceeds the 64-element threshold for GPU path
        let leaves: Vec<Fp> = (1..129u64).map(Fp::from).collect();

        let gpu_result = MetalPoseidon2Backend::hash_leaves_gpu(&leaves);
        assert!(
            gpu_result.is_ok(),
            "GPU hashing failed: {:?}",
            gpu_result.err()
        );

        let gpu_hashes = gpu_result.unwrap();
        let cpu_hashes: Vec<Digest> = leaves.iter().map(Poseidon2::hash_single).collect();

        assert_eq!(gpu_hashes.len(), cpu_hashes.len());
        for (i, (gpu, cpu)) in gpu_hashes.iter().zip(cpu_hashes.iter()).enumerate() {
            assert_eq!(gpu, cpu, "Hash mismatch at leaf index {i}");
        }
    }

    #[test]
    fn test_hash_level_gpu_matches_cpu() {
        // 128 digests -> 64 pairs, exceeds the 32-pair threshold for GPU path
        let leaves: Vec<Fp> = (1..129u64).map(Fp::from).collect();
        let nodes: Vec<Digest> = leaves.iter().map(Poseidon2::hash_single).collect();

        let gpu_result = MetalPoseidon2Backend::hash_level_gpu(&nodes);
        assert!(
            gpu_result.is_ok(),
            "GPU level hashing failed: {:?}",
            gpu_result.err()
        );

        let gpu_parents = gpu_result.unwrap();
        let cpu_parents: Vec<Digest> = (0..64)
            .map(|i| Poseidon2::compress(&nodes[i * 2], &nodes[i * 2 + 1]))
            .collect();

        assert_eq!(gpu_parents.len(), cpu_parents.len());
        for (i, (gpu, cpu)) in gpu_parents.iter().zip(cpu_parents.iter()).enumerate() {
            assert_eq!(gpu, cpu, "Level hash mismatch at pair index {i}");
        }
    }

    #[test]
    fn test_metal_poseidon2_merkle_tree_vs_cpu() {
        // 128 leaves: GPU path for leaf hashing
        let values: Vec<Fp> = (1..129u64).map(Fp::from).collect();

        let cpu_tree = MerkleTree::<Poseidon2Backend>::build(&values);
        let metal_tree = MerkleTree::<MetalPoseidon2Backend>::build(&values);

        if let (Some(cpu), Some(metal)) = (cpu_tree, metal_tree) {
            assert_eq!(
                cpu.root, metal.root,
                "GPU and CPU Merkle roots differ for 128 leaves"
            );
        }
    }

    #[test]
    fn test_metal_poseidon2_small_cpu_fallback() {
        // Small input (< 64 leaves): exercises CPU fallback path
        let values: Vec<Fp> = (1..9u64).map(Fp::from).collect();

        let cpu_tree = MerkleTree::<Poseidon2Backend>::build(&values);
        let metal_tree = MerkleTree::<MetalPoseidon2Backend>::build(&values);

        if let (Some(cpu), Some(metal)) = (cpu_tree, metal_tree) {
            assert_eq!(cpu.root, metal.root);
        }
    }

    #[test]
    fn test_hash_leaves_empty() {
        let result = MetalPoseidon2Backend::hash_leaves_gpu(&[]);
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn test_hash_level_empty() {
        let result = MetalPoseidon2Backend::hash_level_gpu(&[]);
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn test_hash_level_odd_count() {
        let nodes = vec![[Fp::from(1u64), Fp::from(2u64)]];
        let result = MetalPoseidon2Backend::hash_level_gpu(&nodes);
        assert!(result.is_err());
    }

    #[test]
    fn test_hash_leaves_all_zeros() {
        // Explicitly test with all-zero inputs (64 elements to hit GPU path)
        let leaves: Vec<Fp> = vec![Fp::from(0u64); 64];

        let gpu_result = MetalPoseidon2Backend::hash_leaves_gpu(&leaves);
        assert!(
            gpu_result.is_ok(),
            "GPU hashing failed: {:?}",
            gpu_result.err()
        );

        let gpu_hashes = gpu_result.unwrap();
        let cpu_hashes: Vec<Digest> = leaves.iter().map(Poseidon2::hash_single).collect();

        for (i, (gpu, cpu)) in gpu_hashes.iter().zip(cpu_hashes.iter()).enumerate() {
            assert_eq!(gpu, cpu, "Hash mismatch at zero-leaf index {i}");
        }
    }

    #[test]
    fn test_hash_leaves_gpu_repeated_calls() {
        // Test rapid sequential GPU calls (fuzzer-like pattern)
        let leaves: Vec<Fp> = (0..64u64).map(Fp::from).collect();
        for iteration in 0..10 {
            let gpu_result = MetalPoseidon2Backend::hash_leaves_gpu(&leaves);
            assert!(gpu_result.is_ok(), "GPU failed on iteration {iteration}");

            let gpu_hashes = gpu_result.unwrap();
            let cpu_hashes: Vec<Digest> = leaves.iter().map(Poseidon2::hash_single).collect();

            for (i, (gpu, cpu)) in gpu_hashes.iter().zip(cpu_hashes.iter()).enumerate() {
                assert_eq!(gpu, cpu, "Mismatch at index {i}, iteration {iteration}");
            }
        }
    }

    use proptest::prelude::*;

    fn arb_fp() -> impl Strategy<Value = Fp> {
        any::<u64>().prop_map(Fp::from)
    }

    fn arb_fp_vec(len: usize) -> impl Strategy<Value = Vec<Fp>> {
        prop::collection::vec(arb_fp(), len)
    }

    proptest! {
        #[test]
        fn proptest_hash_leaves_gpu_vs_cpu(leaves in arb_fp_vec(128)) {
            let gpu_hashes = MetalPoseidon2Backend::hash_leaves_gpu(&leaves)
                .expect("GPU hashing failed");
            let cpu_hashes: Vec<Digest> = leaves
                .iter()
                .map(Poseidon2::hash_single)
                .collect();

            prop_assert_eq!(gpu_hashes.len(), cpu_hashes.len());
            for (i, (gpu, cpu)) in gpu_hashes.iter().zip(cpu_hashes.iter()).enumerate() {
                prop_assert_eq!(gpu, cpu, "Hash mismatch at leaf {}", i);
            }
        }

        #[test]
        fn proptest_hash_level_gpu_vs_cpu(leaves in arb_fp_vec(128)) {
            let nodes: Vec<Digest> = leaves.iter().map(Poseidon2::hash_single).collect();
            let gpu_parents = MetalPoseidon2Backend::hash_level_gpu(&nodes)
                .expect("GPU level hashing failed");
            let cpu_parents: Vec<Digest> = (0..64)
                .map(|i| Poseidon2::compress(&nodes[i * 2], &nodes[i * 2 + 1]))
                .collect();

            prop_assert_eq!(gpu_parents.len(), cpu_parents.len());
            for (i, (gpu, cpu)) in gpu_parents.iter().zip(cpu_parents.iter()).enumerate() {
                prop_assert_eq!(gpu, cpu, "Level hash mismatch at pair {}", i);
            }
        }

        #[test]
        fn proptest_merkle_tree_gpu_vs_cpu(leaves in arb_fp_vec(128)) {
            let cpu_tree = MerkleTree::<Poseidon2Backend>::build(&leaves).unwrap();
            let metal_tree = MerkleTree::<MetalPoseidon2Backend>::build(&leaves).unwrap();

            prop_assert_eq!(cpu_tree.root, metal_tree.root);
        }
    }
}
