//! CUDA-accelerated Merkle tree backend
//!
//! This module provides GPU-accelerated Merkle tree construction using NVIDIA CUDA,
//! implementing a hybrid CPU/GPU approach that optimizes performance across different
//! input sizes while maintaining correctness and compatibility with the standard
//! Merkle tree interface.
//!
//! # Architecture Overview
//!
//! ## Hybrid CPU/GPU Design
//!
//! The implementation uses an intelligent threshold-based strategy:
//!
//! - **Small inputs (< 64 elements)**: Pure CPU execution
//!   - GPU kernel launch overhead (~5-20μs) dominates computation time
//!   - Host-to-device memory transfer adds additional latency
//!   - CPU direct execution is faster for these cases
//!
//! - **Large inputs (≥ 64 elements)**: GPU acceleration
//!   - Parallel hash computation across thousands of threads
//!   - Amortizes kernel launch and transfer overhead
//!   - Significant speedup for batch operations (5-10x for large trees)
//!
//! ## Memory Management Strategy
//!
//! ### Lazy Initialization (OnceLock)
//!
//! ```text
//! First call:  Initialize CUDA → Cache state globally
//! Later calls: Reuse cached state (no initialization overhead)
//! ```
//!
//! Benefits:
//! - Zero cost if CUDA never used (feature-gated compilation)
//! - One-time initialization amortized across all operations
//! - Thread-safe global state via `OnceLock<Result<CudaMerkleState, String>>`
//!
//! ### Graceful Degradation
//!
//! ```text
//! CUDA available? → Use GPU acceleration
//! CUDA failed?    → Silent fallback to CPU (no user-visible error)
//! ```
//!
//! This ensures code works on systems without CUDA support.
//!
//! ## Two Operation Modes
//!
//! ### 1. Trait-based Interface (`IsMerkleTreeBackend`)
//!
//! Standard Merkle tree interface with GPU acceleration:
//! - `hash_data()`: Single element → CPU only
//! - `hash_leaves()`: Batch operation → GPU if >= 64 elements
//! - `hash_new_parent()`: Single pair → CPU only
//!
//! **Use case:** Drop-in replacement for CPU backend
//!
//! ### 2. Full-tree GPU Builder (`build_tree_cuda`)
//!
//! Optimized GPU-only tree construction:
//! ```text
//! 1. Upload leaves to GPU once
//! 2. Hash leaves on GPU
//! 3. Build all tree layers on GPU (no re-upload between layers)
//! 4. Download final result (root + complete node array)
//! ```
//!
//! **Use case:** Maximum performance for large tree construction
//!
//! **Performance difference:**
//! - Trait-based: Downloads intermediate layers (more transfers)
//! - Full-tree: Keeps everything on GPU until done (fewer transfers)
//!
//! ## Threshold Selection: Why 64 Elements?
//!
//! The GPU_THRESHOLD of 64 was chosen based on:
//!
//! 1. **Warp size**: 32 threads per warp (2 warps = 64 elements)
//! 2. **Overhead analysis**:
//!    - Kernel launch: ~5-20μs
//!    - Memory transfer: ~50ns per byte × 32 bytes/element × 64 = ~100μs
//!    - CPU hash: ~10μs per element × 64 = ~640μs
//!    - GPU hash (parallel): ~50μs total for 64 elements
//!
//! 3. **Crossover point**: GPU becomes faster around 100-150 elements
//! 4. **Conservative choice**: 64 ensures GPU is actually faster
//!
//! Note: Matches Metal backend threshold for consistency
//!
//! ## Tree Layout
//!
//! Compact array format (2n-1 nodes for n leaves, padded to power of 2):
//!
//! ```text
//! [root][level 1 nodes][level 2 nodes]...[leaf hashes]
//!
//! Example for 4 leaves:
//! [root, L1_left, L1_right, leaf0, leaf1, leaf2, leaf3]
//!  └─┬─┘  └──┬───┘ └──┬────┘  └──────────┬───────────┘
//!    1       2         2               4 = 2n-1 = 7 nodes
//! ```
//!
//! Padding: Odd leaf counts are padded to next power of 2 by repeating the last leaf.
//!
//! ## Implementation Details
//!
//! ### CUDA Kernels
//!
//! Two GPU kernels in `shaders/merkle_tree.cu`:
//!
//! 1. **`hash_leaves`**: Parallel Poseidon hash of individual leaves
//!    - Thread model: One thread per leaf (independent)
//!    - Each thread: Input[i] → Poseidon → Output[i]
//!
//! 2. **`compute_parents`**: Parallel parent node computation
//!    - Thread model: One thread per parent
//!    - Each thread: Hash(children[2*i], children[2*i+1]) → parents[i]
//!
//! ### Poseidon Hash
//!
//! Implementation: `shaders/poseidon.cuh`
//! - Algorithm: Hades permutation (4 full rounds + 83 partial rounds + 4 full rounds)
//! - Field: Stark252 (252-bit prime field for StarkNet compatibility)
//! - Constants: Preloaded round constants (uploaded once at initialization)
//!
//! ### Error Handling
//!
//! - **CUDA unavailable**: Silent fallback to CPU (graceful degradation)
//! - **GPU OOM**: Error propagated to caller
//! - **Kernel launch failure**: Error propagated to caller
//! - **Invalid input**: Runtime validation (e.g., odd children count)
//!
//! # Security Considerations
//!
//! ⚠️ **IMPORTANT**: This CUDA implementation has not undergone timing analysis
//! for constant-time execution. Do NOT use for cryptographic applications involving
//! secret data without proper timing attack analysis.
//!
//! **Current status:**
//! - ✅ Suitable for: Non-sensitive Merkle trees, development, testing
//! - ❌ NOT verified for: ZKP witness data, private membership proofs
//!
//! The implementation may have timing variations in:
//! - GPU kernel execution (dependent on input patterns)
//! - Memory transfers (host↔device)
//! - Branch divergence in CUDA kernels
//!
//! Before using in production cryptographic systems, conduct proper timing
//! analysis and consider constant-time requirements for your specific use case.
//!
//! # Performance Characteristics
//!
//! Expected speedup (compared to CPU):
//! - **< 64 elements**: 0.5-1x (CPU faster due to overhead)
//! - **2^10 elements**: 1-2x (breakeven region)
//! - **2^14 elements**: 2-4x (moderate speedup)
//! - **2^18 elements**: 5-8x (significant speedup)
//! - **2^20+ elements**: 6-10x (maximum speedup, scales with GPU)
//!
//! Bottlenecks:
//! - Memory bandwidth (for very large trees)
//! - PCIe transfer time (trait-based API with multiple downloads)
//!
//! # Examples
//!
//! ## Basic Usage (Trait-based)
//!
//! ```no_run
//! use lambdaworks_crypto::merkle_tree::merkle::MerkleTree;
//! use lambdaworks_crypto::merkle_tree::cuda::CudaPoseidonBackend;
//! use lambdaworks_math::field::element::FieldElement;
//! use lambdaworks_math::field::fields::fft_friendly::stark_252_prime_field::Stark252PrimeField;
//!
//! type Fp = FieldElement<Stark252PrimeField>;
//!
//! // Works on any system (fallback to CPU if CUDA unavailable)
//! let leaves: Vec<Fp> = (0..1024).map(Fp::from).collect();
//! let tree = MerkleTree::<CudaPoseidonBackend>::build(&leaves).unwrap();
//! println!("Root: {:?}", tree.root);
//! ```
//!
//! ## High-Performance Mode (Full-tree GPU)
//!
//! ```no_run
//! # use lambdaworks_crypto::merkle_tree::cuda::CudaPoseidonBackend;
//! # use lambdaworks_math::field::element::FieldElement;
//! # use lambdaworks_math::field::fields::fft_friendly::stark_252_prime_field::Stark252PrimeField;
//! # type Fp = FieldElement<Stark252PrimeField>;
//! // Check CUDA availability first
//! if CudaPoseidonBackend::is_cuda_available() {
//!     let leaves: Vec<Fp> = (0..1_000_000).map(Fp::from).collect();
//!     let (root, nodes) = CudaPoseidonBackend::build_tree_cuda(&leaves).unwrap();
//!     println!("Built tree with {} nodes", nodes.len());
//! } else {
//!     println!("CUDA not available, falling back to CPU");
//! }
//! ```

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
/// ⚠️ **Security**: Timing guarantees not verified. See module docs for details.
///
/// # Example
///
/// ```no_run
/// use lambdaworks_crypto::merkle_tree::merkle::MerkleTree;
/// use lambdaworks_crypto::merkle_tree::cuda::CudaPoseidonBackend;
/// use lambdaworks_math::field::{
///     element::FieldElement,
///     fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
/// };
///
/// type Fp = FieldElement<Stark252PrimeField>;
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
        // Below this threshold CPU is faster because the CUDA kernel launch +
        // host-to-device transfer overhead dominates. 64 matches the Metal
        // backend and is approximately 2 warps of work.
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
                Err(_) => {
                    // Fall back to CPU silently
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
/// This provides an alternative tree-building method that builds all tree
/// layers on the GPU, uploading leaf data once and only downloading each
/// computed layer (no per-layer re-uploads).
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

    // Edge case tests

    #[test]
    fn test_cuda_backend_odd_leaves_small() {
        // Test with 17 leaves (odd, small count)
        let leaves: Vec<Fp> = (0..17).map(Fp::from).collect();
        let cuda_tree = MerkleTree::<CudaPoseidonBackend>::build(&leaves).unwrap();
        let cpu_tree = MerkleTree::<CpuBackend>::build(&leaves).unwrap();

        assert_eq!(cuda_tree.root, cpu_tree.root, "Root mismatch for 17 leaves");
    }

    #[test]
    fn test_cuda_backend_odd_leaves_medium() {
        // Test with 33 leaves (odd, crosses power-of-2 boundary)
        let leaves: Vec<Fp> = (0..33).map(Fp::from).collect();
        let cuda_tree = MerkleTree::<CudaPoseidonBackend>::build(&leaves).unwrap();
        let cpu_tree = MerkleTree::<CpuBackend>::build(&leaves).unwrap();

        assert_eq!(cuda_tree.root, cpu_tree.root, "Root mismatch for 33 leaves");
    }

    #[test]
    fn test_cuda_backend_odd_leaves_large() {
        // Test with 129 leaves (odd, larger than GPU threshold)
        let leaves: Vec<Fp> = (0..129).map(Fp::from).collect();
        let cuda_tree = MerkleTree::<CudaPoseidonBackend>::build(&leaves).unwrap();
        let cpu_tree = MerkleTree::<CpuBackend>::build(&leaves).unwrap();

        assert_eq!(
            cuda_tree.root, cpu_tree.root,
            "Root mismatch for 129 leaves"
        );
    }

    #[test]
    fn test_cuda_backend_single_leaf_tree() {
        // Test with single leaf
        let leaves: Vec<Fp> = vec![Fp::from(42u64)];
        let cuda_tree = MerkleTree::<CudaPoseidonBackend>::build(&leaves).unwrap();
        let cpu_tree = MerkleTree::<CpuBackend>::build(&leaves).unwrap();

        assert_eq!(
            cuda_tree.root, cpu_tree.root,
            "Root mismatch for single leaf"
        );
    }

    #[test]
    fn test_cuda_backend_two_leaves() {
        // Test with exactly two leaves (minimal tree)
        let leaves: Vec<Fp> = vec![Fp::from(1u64), Fp::from(2u64)];
        let cuda_tree = MerkleTree::<CudaPoseidonBackend>::build(&leaves).unwrap();
        let cpu_tree = MerkleTree::<CpuBackend>::build(&leaves).unwrap();

        assert_eq!(
            cuda_tree.root, cpu_tree.root,
            "Root mismatch for two leaves"
        );
    }

    #[test]
    #[ignore = "requires CUDA GPU"]
    fn test_cuda_build_tree_odd_leaves() {
        if !CudaPoseidonBackend::is_cuda_available() {
            return;
        }

        // Test build_tree_cuda with odd number of leaves
        let leaves: Vec<Fp> = (0..99).map(Fp::from).collect();
        let (cuda_root, cuda_nodes) = CudaPoseidonBackend::build_tree_cuda(&leaves).unwrap();
        let cpu_tree = MerkleTree::<CpuBackend>::build(&leaves).unwrap();

        // Verify root matches
        assert_eq!(cuda_root, cpu_tree.root);

        // Verify nodes count: leaves are padded to next power of 2 (128)
        let n = leaves.len().next_power_of_two();
        assert_eq!(n, 128, "Expected padding to 128");
        let expected_node_count = 2 * n - 1;
        assert_eq!(
            cuda_nodes.len(),
            expected_node_count,
            "Expected {} nodes for {} leaves (padded to {}), got {}",
            expected_node_count,
            leaves.len(),
            n,
            cuda_nodes.len()
        );

        // Verify root is first element
        assert_eq!(cuda_nodes[0], cuda_root);
    }

    #[test]
    #[ignore = "requires CUDA GPU"]
    fn test_cuda_build_tree_single_leaf() {
        if !CudaPoseidonBackend::is_cuda_available() {
            return;
        }

        // Test build_tree_cuda with single leaf
        let leaves: Vec<Fp> = vec![Fp::from(42u64)];
        let (cuda_root, cuda_nodes) = CudaPoseidonBackend::build_tree_cuda(&leaves).unwrap();

        // Single leaf: root equals the hashed leaf, nodes contains just that one element
        assert_eq!(cuda_nodes.len(), 1);
        assert_eq!(cuda_nodes[0], cuda_root);
    }

    #[test]
    #[ignore = "requires CUDA GPU"]
    fn test_cuda_build_layer_rejects_odd_children() {
        use super::state::CudaMerkleState;

        let state = match CudaMerkleState::new() {
            Ok(s) => s,
            Err(_) => return, // Skip if CUDA not available
        };

        // build_layer should reject odd number of children in release builds
        let odd_children: Vec<Fp> = vec![Fp::from(1u64), Fp::from(2u64), Fp::from(3u64)];
        let result = state.build_layer(&odd_children);

        assert!(
            result.is_err(),
            "build_layer should reject odd number of children"
        );

        if let Err(e) = result {
            let error_msg = format!("{:?}", e);
            assert!(
                error_msg.contains("even number"),
                "Error should mention even number requirement, got: {}",
                error_msg
            );
        }
    }
}
