//! Metal GPU-accelerated Merkle tree backend using Poseidon hash.

use crate::hash::poseidon::parameters::PermutationParameters;
use crate::hash::poseidon::starknet::PoseidonCairoStark252;
use crate::hash::poseidon::Poseidon;
use crate::merkle_tree::traits::IsMerkleTreeBackend;
use lambdaworks_gpu::metal::abstractions::{errors::MetalError, state::DynamicMetalState};
use lambdaworks_math::field::{
    element::FieldElement, fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
};
use lambdaworks_math::unsigned_integer::element::U256;
use std::marker::PhantomData;

type FE = FieldElement<Stark252PrimeField>;

/// Metal shader source code for Poseidon hash
const POSEIDON_SHADER: &str = include_str!("shaders/poseidon.metal");

/// GPU representation of a 256-bit field element (4 x 64-bit limbs)
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct GpuFieldElement {
    pub limbs: [u64; 4],
}

impl From<&FE> for GpuFieldElement {
    fn from(fe: &FE) -> Self {
        // Send values in Montgomery form (aR mod p) to the GPU, which operates
        // entirely in Montgomery form for fast CIOS multiplication.
        // Reverse limbs: Rust U256 is MSB-first (limbs[0] = most significant),
        // Metal shader is LSB-first (limbs[0] = least significant).
        let mont = fe.value();
        Self {
            limbs: [mont.limbs[3], mont.limbs[2], mont.limbs[1], mont.limbs[0]],
        }
    }
}

impl From<GpuFieldElement> for FE {
    fn from(gpu: GpuFieldElement) -> Self {
        // Reverse limbs: Metal LSB-first -> Rust MSB-first
        FE::from(&U256 {
            limbs: [gpu.limbs[3], gpu.limbs[2], gpu.limbs[1], gpu.limbs[0]],
        })
    }
}

/// Convert Poseidon round constants to GPU format
fn get_round_constants_gpu() -> Vec<GpuFieldElement> {
    <PoseidonCairoStark252 as PermutationParameters>::ROUND_CONSTANTS
        .iter()
        .map(GpuFieldElement::from)
        .collect()
}

/// Metal GPU-accelerated Merkle tree backend using Poseidon hash.
///
/// This backend uses Apple's Metal framework to accelerate Merkle tree
/// construction on Apple Silicon devices.
#[derive(Clone, Default)]
pub struct MetalPoseidonBackend {
    _phantom: PhantomData<()>,
}

impl MetalPoseidonBackend {
    /// Create a new Metal Poseidon backend.
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }

    /// Hash leaves using GPU acceleration.
    ///
    /// Falls back to CPU implementation if GPU is unavailable.
    pub fn hash_leaves_gpu(leaves: &[FE]) -> Result<Vec<FE>, MetalError> {
        if leaves.is_empty() {
            return Ok(Vec::new());
        }

        // For small inputs, CPU is likely faster due to GPU overhead
        if leaves.len() < 64 {
            return Ok(leaves
                .iter()
                .map(PoseidonCairoStark252::hash_single)
                .collect());
        }

        // Convert leaves to GPU format
        let gpu_leaves: Vec<GpuFieldElement> = leaves.iter().map(GpuFieldElement::from).collect();
        let round_constants = get_round_constants_gpu();

        // Create Metal state
        let mut state = DynamicMetalState::new()?;
        state.load_library(POSEIDON_SHADER)?;

        // Allocate Metal buffers
        let input_buffer = state.alloc_buffer_with_data(&gpu_leaves)?;
        let output_buffer =
            state.alloc_buffer(leaves.len() * std::mem::size_of::<GpuFieldElement>())?;
        let constants_buffer = state.alloc_buffer_with_data(&round_constants)?;
        let count = [leaves.len() as u32];
        let count_buffer = state.alloc_buffer_with_data(&count)?;

        // Prepare pipeline and execute
        let max_threads = state.prepare_pipeline("hash_single")?;
        state.execute_compute(
            "hash_single",
            &[
                &input_buffer,
                &output_buffer,
                &constants_buffer,
                &count_buffer,
            ],
            leaves.len() as u64,
            max_threads,
        )?;

        // Read results
        let results: Vec<GpuFieldElement> =
            unsafe { state.read_buffer(&output_buffer, leaves.len()) };

        Ok(results.into_iter().map(FE::from).collect())
    }

    /// Hash pairs of nodes to build a tree level using GPU.
    ///
    /// Takes `2N` nodes and produces `N` parent nodes.
    pub fn hash_level_gpu(nodes: &[FE]) -> Result<Vec<FE>, MetalError> {
        if nodes.is_empty() {
            return Ok(Vec::new());
        }

        if !nodes.len().is_multiple_of(2) {
            return Err(MetalError::InvalidInputSize {
                expected: nodes.len() + 1,
                actual: nodes.len(),
            });
        }

        let output_count = nodes.len() / 2;

        // For small inputs, CPU is likely faster
        if output_count < 32 {
            let mut results = Vec::with_capacity(output_count);
            for i in 0..output_count {
                results.push(PoseidonCairoStark252::hash(
                    &nodes[i * 2],
                    &nodes[i * 2 + 1],
                ));
            }
            return Ok(results);
        }

        // Convert nodes to GPU format
        let gpu_nodes: Vec<GpuFieldElement> = nodes.iter().map(GpuFieldElement::from).collect();
        let round_constants = get_round_constants_gpu();

        // Create Metal state
        let mut state = DynamicMetalState::new()?;
        state.load_library(POSEIDON_SHADER)?;

        let input_buffer = state.alloc_buffer_with_data(&gpu_nodes)?;
        let output_buffer =
            state.alloc_buffer(output_count * std::mem::size_of::<GpuFieldElement>())?;
        let constants_buffer = state.alloc_buffer_with_data(&round_constants)?;
        let count = [output_count as u32];
        let count_buffer = state.alloc_buffer_with_data(&count)?;

        // Prepare pipeline and execute
        let max_threads = state.prepare_pipeline("merkle_hash_level")?;
        state.execute_compute(
            "merkle_hash_level",
            &[
                &input_buffer,
                &output_buffer,
                &constants_buffer,
                &count_buffer,
            ],
            output_count as u64,
            max_threads,
        )?;

        let results: Vec<GpuFieldElement> =
            unsafe { state.read_buffer(&output_buffer, output_count) };

        Ok(results.into_iter().map(FE::from).collect())
    }
}

impl IsMerkleTreeBackend for MetalPoseidonBackend {
    type Node = FE;
    type Data = FE;

    fn hash_data(leaf: &Self::Data) -> Self::Node {
        // Single element - use CPU for efficiency
        PoseidonCairoStark252::hash_single(leaf)
    }

    fn hash_leaves(unhashed_leaves: &[Self::Data]) -> Vec<Self::Node> {
        // Try GPU acceleration, fall back to CPU on error
        match Self::hash_leaves_gpu(unhashed_leaves) {
            Ok(hashed) => hashed,
            Err(_) => {
                // Fall back to CPU implementation
                unhashed_leaves
                    .iter()
                    .map(PoseidonCairoStark252::hash_single)
                    .collect()
            }
        }
    }

    fn hash_new_parent(left: &Self::Node, right: &Self::Node) -> Self::Node {
        // Single pair - use CPU for efficiency
        PoseidonCairoStark252::hash(left, right)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::merkle_tree::backends::field_element::TreePoseidon;
    use crate::merkle_tree::merkle::MerkleTree;

    #[test]
    fn test_gpu_field_element_sends_montgomery_form() {
        // GpuFieldElement now sends Montgomery form to the GPU.
        // Verify that the limbs match fe.value() (reversed for LSB-first).
        let fe = FE::from(12345u64);
        let gpu = GpuFieldElement::from(&fe);
        let mont = fe.value();
        assert_eq!(gpu.limbs[0], mont.limbs[3]); // LSB in Metal = MSB in Rust
        assert_eq!(gpu.limbs[1], mont.limbs[2]);
        assert_eq!(gpu.limbs[2], mont.limbs[1]);
        assert_eq!(gpu.limbs[3], mont.limbs[0]);
    }

    #[test]
    fn test_gpu_field_element_canonical_roundtrip() {
        // The full pipeline: CPU sends Montgomery form to GPU, GPU returns
        // canonical form. Verify using FE::from(GpuFieldElement) which
        // takes a canonical value and converts to Montgomery internally.
        let values = [
            FE::from(0u64),
            FE::from(1u64),
            FE::from(u64::MAX),
            FE::from_hex_unchecked(
                "0x800000000000010FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFE",
            ),
        ];
        for fe in &values {
            // Simulate what the GPU does: receive Montgomery, return canonical
            let canonical = fe.canonical();
            let gpu_output = GpuFieldElement {
                limbs: [
                    canonical.limbs[3],
                    canonical.limbs[2],
                    canonical.limbs[1],
                    canonical.limbs[0],
                ],
            };
            let back = FE::from(gpu_output);
            assert_eq!(*fe, back, "Roundtrip failed for {fe}");
        }
    }

    #[test]
    fn test_metal_backend_matches_cpu_small() {
        // Small input (< 64 leaves): exercises CPU fallback path
        let values: Vec<FE> = (1..9u64).map(FE::from).collect();

        let cpu_tree = MerkleTree::<TreePoseidon<PoseidonCairoStark252>>::build(&values);
        let metal_tree = MerkleTree::<MetalPoseidonBackend>::build(&values);

        if let (Some(cpu), Some(metal)) = (cpu_tree, metal_tree) {
            assert_eq!(cpu.root, metal.root);
        }
    }

    #[test]
    fn test_hash_leaves_gpu_matches_cpu() {
        // 128 elements: exceeds the 64-element threshold to exercise the GPU path
        let leaves: Vec<FE> = (1..129u64).map(FE::from).collect();

        let gpu_result = MetalPoseidonBackend::hash_leaves_gpu(&leaves);
        assert!(
            gpu_result.is_ok(),
            "GPU hashing failed: {:?}",
            gpu_result.err()
        );

        let gpu_hashes = gpu_result.unwrap();
        let cpu_hashes: Vec<FE> = leaves
            .iter()
            .map(|l| PoseidonCairoStark252::hash_single(l))
            .collect();

        assert_eq!(gpu_hashes.len(), cpu_hashes.len());
        for (i, (gpu, cpu)) in gpu_hashes.iter().zip(cpu_hashes.iter()).enumerate() {
            assert_eq!(gpu, cpu, "Hash mismatch at leaf index {i}");
        }
    }

    #[test]
    fn test_metal_backend_merkle_tree_large() {
        // 128 leaves: GPU path for leaf hashing
        let values: Vec<FE> = (1..129u64).map(FE::from).collect();

        let cpu_tree = MerkleTree::<TreePoseidon<PoseidonCairoStark252>>::build(&values);
        let metal_tree = MerkleTree::<MetalPoseidonBackend>::build(&values);

        if let (Some(cpu), Some(metal)) = (cpu_tree, metal_tree) {
            assert_eq!(
                cpu.root, metal.root,
                "GPU and CPU Merkle roots differ for 128 leaves"
            );
        }
    }

    #[test]
    fn test_hash_leaves_small_cpu_fallback() {
        // Small input should use CPU path and produce correct results
        let leaves: Vec<FE> = (1..5u64).map(FE::from).collect();

        let result = MetalPoseidonBackend::hash_leaves_gpu(&leaves);
        assert!(result.is_ok());

        let gpu_result = result.unwrap();
        let cpu_result: Vec<FE> = leaves
            .iter()
            .map(|l| PoseidonCairoStark252::hash_single(l))
            .collect();

        assert_eq!(gpu_result, cpu_result);
    }

    #[test]
    fn test_hash_leaves_empty() {
        let result = MetalPoseidonBackend::hash_leaves_gpu(&[]);
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn test_hash_level_gpu_matches_cpu() {
        // 128 nodes -> 64 pairs, exceeds the 32-element threshold for GPU path
        let nodes: Vec<FE> = (1..129u64).map(FE::from).collect();

        let gpu_result = MetalPoseidonBackend::hash_level_gpu(&nodes);
        assert!(
            gpu_result.is_ok(),
            "GPU level hashing failed: {:?}",
            gpu_result.err()
        );

        let gpu_parents = gpu_result.unwrap();
        let cpu_parents: Vec<FE> = (0..64)
            .map(|i| PoseidonCairoStark252::hash(&nodes[i * 2], &nodes[i * 2 + 1]))
            .collect();

        assert_eq!(gpu_parents.len(), cpu_parents.len());
        for (i, (gpu, cpu)) in gpu_parents.iter().zip(cpu_parents.iter()).enumerate() {
            assert_eq!(gpu, cpu, "Level hash mismatch at pair index {i}");
        }
    }
}
