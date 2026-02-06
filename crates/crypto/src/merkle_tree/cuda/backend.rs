// CUDA-accelerated Merkle tree backend
//
// This backend uses GPU for batch leaf hashing while falling back to CPU
// for single-element operations where GPU overhead would be counterproductive.

use crate::hash::poseidon::starknet::PoseidonCairoStark252;
use crate::hash::poseidon::Poseidon;
use crate::merkle_tree::traits::IsMerkleTreeBackend;
use alloc::vec::Vec;
use lambdaworks_math::field::{
    element::FieldElement, fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
};
use std::sync::OnceLock;

use super::state::CudaMerkleState;

type Fp = FieldElement<Stark252PrimeField>;

/// Global CUDA state, lazily initialized on first use.
/// This avoids the overhead of re-initializing CUDA for each tree operation.
static CUDA_STATE: OnceLock<Result<CudaMerkleState, String>> = OnceLock::new();

fn get_cuda_state() -> Option<&'static CudaMerkleState> {
    CUDA_STATE
        .get_or_init(|| CudaMerkleState::new().map_err(|e| e.to_string()))
        .as_ref()
        .ok()
}

/// CUDA-accelerated Poseidon Merkle tree backend for Stark252 field.
///
/// This backend uses GPU acceleration for batch operations while maintaining
/// compatibility with the standard `IsMerkleTreeBackend` trait. Single-element
/// operations fall back to CPU since GPU kernel launch overhead would negate
/// any performance benefit.
///
/// # Example
///
/// ```ignore
/// use lambdaworks_crypto::merkle_tree::merkle::MerkleTree;
/// use lambdaworks_crypto::merkle_tree::cuda::CudaPoseidonBackend;
///
/// let leaves: Vec<Fp> = (0..1024).map(Fp::from).collect();
/// let tree = MerkleTree::<CudaPoseidonBackend>::build(&leaves).unwrap();
/// ```
#[derive(Clone, Default)]
pub struct CudaPoseidonBackend;

impl IsMerkleTreeBackend for CudaPoseidonBackend {
    type Node = Fp;
    type Data = Fp;

    /// Hash a single data element to a node.
    /// Falls back to CPU since GPU overhead exceeds benefit for single operations.
    fn hash_data(input: &Fp) -> Fp {
        PoseidonCairoStark252::hash_single(input)
    }

    /// Hash all leaves in parallel using GPU when available.
    /// Falls back to CPU if CUDA initialization failed.
    fn hash_leaves(unhashed_leaves: &[Fp]) -> Vec<Fp> {
        // For small inputs, CPU may be faster due to GPU transfer overhead
        const GPU_THRESHOLD: usize = 64;

        if unhashed_leaves.len() < GPU_THRESHOLD {
            return unhashed_leaves
                .iter()
                .map(PoseidonCairoStark252::hash_single)
                .collect();
        }

        // Try GPU path
        if let Some(state) = get_cuda_state() {
            match state.hash_leaves(unhashed_leaves) {
                Ok(result) => return result,
                Err(e) => {
                    // Log error and fall back to CPU
                    eprintln!("CUDA hash_leaves failed, falling back to CPU: {}", e);
                }
            }
        }

        // CPU fallback
        unhashed_leaves
            .iter()
            .map(PoseidonCairoStark252::hash_single)
            .collect()
    }

    /// Hash two child nodes to produce a parent node.
    /// Falls back to CPU since GPU overhead exceeds benefit for single operations.
    fn hash_new_parent(left: &Fp, right: &Fp) -> Fp {
        PoseidonCairoStark252::hash(left, right)
    }
}

/// CUDA-accelerated full tree builder.
///
/// This provides an alternative tree-building method that keeps data on the GPU
/// throughout construction, avoiding repeated CPU-GPU transfers.
impl CudaPoseidonBackend {
    /// Build a complete Merkle tree using GPU acceleration.
    ///
    /// This method keeps all intermediate data on the GPU, providing better
    /// performance than the trait-based approach for large trees.
    ///
    /// Returns `(root, nodes)` where nodes is the full tree array in the format:
    /// `[internal nodes (root first, then level by level)][leaf hashes]`
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA is not available or if GPU operations fail.
    pub fn build_tree_cuda(leaves: &[Fp]) -> Result<(Fp, Vec<Fp>), String> {
        let state = get_cuda_state().ok_or("CUDA not available")?;
        state
            .build_tree(leaves)
            .map_err(|e| format!("CUDA build_tree failed: {}", e))
    }

    /// Check if CUDA acceleration is available.
    pub fn is_cuda_available() -> bool {
        get_cuda_state().is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::merkle_tree::backends::field_element::TreePoseidon;
    use crate::merkle_tree::merkle::MerkleTree;

    type CpuBackend = TreePoseidon<PoseidonCairoStark252>;

    #[test]
    fn test_cuda_backend_single_hash() {
        let input = Fp::from(42u64);
        let cuda_result = CudaPoseidonBackend::hash_data(&input);
        let cpu_result = CpuBackend::hash_data(&input);
        assert_eq!(cuda_result, cpu_result);
    }

    #[test]
    fn test_cuda_backend_parent_hash() {
        let left = Fp::from(1u64);
        let right = Fp::from(2u64);
        let cuda_result = CudaPoseidonBackend::hash_new_parent(&left, &right);
        let cpu_result = CpuBackend::hash_new_parent(&left, &right);
        assert_eq!(cuda_result, cpu_result);
    }

    #[test]
    fn test_cuda_backend_hash_leaves_small() {
        let leaves: Vec<Fp> = (0..8).map(Fp::from).collect();
        let cuda_result = CudaPoseidonBackend::hash_leaves(&leaves);
        let cpu_result = CpuBackend::hash_leaves(&leaves);
        assert_eq!(cuda_result, cpu_result);
    }

    #[test]
    fn test_cuda_backend_merkle_tree_matches_cpu() {
        let leaves: Vec<Fp> = (0..16).map(Fp::from).collect();

        let cuda_tree = MerkleTree::<CudaPoseidonBackend>::build(&leaves).unwrap();
        let cpu_tree = MerkleTree::<CpuBackend>::build(&leaves).unwrap();

        assert_eq!(cuda_tree.root, cpu_tree.root);
    }

    #[test]
    #[ignore = "requires CUDA GPU"]
    fn test_cuda_backend_large_tree() {
        if !CudaPoseidonBackend::is_cuda_available() {
            return;
        }

        let leaves: Vec<Fp> = (0..1024).map(Fp::from).collect();
        let cuda_result = CudaPoseidonBackend::hash_leaves(&leaves);
        let cpu_result: Vec<Fp> = leaves
            .iter()
            .map(PoseidonCairoStark252::hash_single)
            .collect();

        assert_eq!(cuda_result, cpu_result);
    }

    #[test]
    #[ignore = "requires CUDA GPU"]
    fn test_cuda_build_tree_full() {
        if !CudaPoseidonBackend::is_cuda_available() {
            return;
        }

        let leaves: Vec<Fp> = (0..256).map(Fp::from).collect();
        let (cuda_root, cuda_nodes) = CudaPoseidonBackend::build_tree_cuda(&leaves).unwrap();

        let cpu_tree = MerkleTree::<CpuBackend>::build(&leaves).unwrap();

        // Verify root matches
        assert_eq!(cuda_root, cpu_tree.root);

        // Verify nodes vector layout:
        // For n leaves (padded to power of 2), should have 2n-1 total nodes
        let n = leaves.len().next_power_of_two();
        let expected_node_count = 2 * n - 1;
        assert_eq!(
            cuda_nodes.len(),
            expected_node_count,
            "Expected {} nodes for {} leaves, got {}",
            expected_node_count,
            n,
            cuda_nodes.len()
        );

        // Verify root is first element in nodes array
        assert_eq!(cuda_nodes[0], cuda_root, "First node should be the root");
    }
}
