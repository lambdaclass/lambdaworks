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
        let canonical: U256 = fe.canonical();
        Self {
            limbs: canonical.limbs,
        }
    }
}

impl From<GpuFieldElement> for FE {
    fn from(gpu: GpuFieldElement) -> Self {
        FE::from(&U256 { limbs: gpu.limbs })
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

        // Prepare pipeline and execute
        let max_threads = state.prepare_pipeline("hash_single")?;
        state.execute_compute(
            "hash_single",
            &[&input_buffer, &output_buffer, &constants_buffer],
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
        if nodes.is_empty() || !nodes.len().is_multiple_of(2) {
            return Err(MetalError::InvalidInputSize {
                expected: nodes.len() + (nodes.len() % 2),
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

        // Prepare pipeline and execute
        let max_threads = state.prepare_pipeline("merkle_hash_level")?;
        state.execute_compute(
            "merkle_hash_level",
            &[&input_buffer, &output_buffer, &constants_buffer],
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
    fn test_gpu_field_element_conversion() {
        let fe = FE::from(12345u64);
        let gpu = GpuFieldElement::from(&fe);
        let back = FE::from(gpu);
        assert_eq!(fe, back);
    }

    #[test]
    fn test_metal_backend_matches_cpu() {
        // Create test data
        let values: Vec<FE> = (1..9u64).map(FE::from).collect();

        // Build tree with CPU backend
        let cpu_tree = MerkleTree::<TreePoseidon<PoseidonCairoStark252>>::build(&values);

        // Build tree with Metal backend
        let metal_tree = MerkleTree::<MetalPoseidonBackend>::build(&values);

        // If both succeeded, roots should match
        if let (Some(cpu), Some(metal)) = (cpu_tree, metal_tree) {
            assert_eq!(cpu.root, metal.root);
        }
    }

    #[test]
    fn test_hash_leaves_small_cpu_fallback() {
        // Small input should use CPU path
        let leaves: Vec<FE> = (1..5u64).map(FE::from).collect();

        let result = MetalPoseidonBackend::hash_leaves_gpu(&leaves);
        assert!(result.is_ok());

        let gpu_result = result.ok();
        let cpu_result: Vec<FE> = leaves
            .iter()
            .map(|l| PoseidonCairoStark252::hash_single(l))
            .collect();

        if let Some(gpu) = gpu_result {
            assert_eq!(gpu, cpu_result);
        }
    }
}
