// CUDA state management for Merkle tree operations
//
// Manages GPU device, memory, and kernel execution for Merkle tree building.

use cudarc::{
    driver::{safe::CudaSlice, CudaDevice, LaunchAsync, LaunchConfig},
    nvrtc::safe::Ptx,
};
use lambdaworks_gpu::cuda::abstractions::errors::CudaError;
use lambdaworks_math::field::{
    element::FieldElement,
    fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
};
use std::sync::Arc;

use crate::hash::poseidon::starknet::PoseidonCairoStark252;
use crate::hash::poseidon::PermutationParameters;

// Include the compiled PTX
const MERKLE_TREE_PTX: &str = include_str!("./shaders/merkle_tree.ptx");

const WARP_SIZE: usize = 32;

type Fp = FieldElement<Stark252PrimeField>;

/// CUDA field element representation (matches the u256 layout in CUDA)
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CudaFieldElement {
    limbs: [u64; 4],
}

impl From<&Fp> for CudaFieldElement {
    fn from(fe: &Fp) -> Self {
        let value = fe.value();
        Self {
            limbs: value.limbs,
        }
    }
}

impl From<CudaFieldElement> for Fp {
    fn from(cuda_fe: CudaFieldElement) -> Self {
        use lambdaworks_math::unsigned_integer::element::UnsignedInteger;
        Fp::from_raw(UnsignedInteger::from_limbs(cuda_fe.limbs))
    }
}

unsafe impl cudarc::driver::safe::DeviceRepr for CudaFieldElement {
    fn as_kernel_param(&self) -> *mut core::ffi::c_void {
        self as *const _ as *mut _
    }
}

/// CUDA state for Merkle tree operations
pub struct CudaMerkleState {
    device: Arc<CudaDevice>,
    round_constants: CudaSlice<CudaFieldElement>,
}

impl CudaMerkleState {
    /// Creates a new CUDA state for Merkle tree operations
    pub fn new() -> Result<Self, CudaError> {
        let device = CudaDevice::new(0)
            .map_err(|err| CudaError::DeviceNotFound(err.to_string()))?;

        let functions = ["hash_leaves", "compute_parents"];
        device
            .load_ptx(Ptx::from_src(MERKLE_TREE_PTX), "merkle_tree", &functions)
            .map_err(|err| CudaError::PtxError(err.to_string()))?;

        let round_constants = Self::upload_round_constants(&device)?;

        Ok(Self {
            device: Arc::new(device),
            round_constants,
        })
    }

    fn upload_round_constants(
        device: &CudaDevice,
    ) -> Result<CudaSlice<CudaFieldElement>, CudaError> {
        let constants: &[Fp] = PoseidonCairoStark252::ROUND_CONSTANTS;
        let cuda_constants: Vec<CudaFieldElement> =
            constants.iter().map(CudaFieldElement::from).collect();

        device
            .htod_sync_copy(&cuda_constants)
            .map_err(|err| CudaError::AllocateMemory(err.to_string()))
    }

    /// Hash leaves in parallel on the GPU
    pub fn hash_leaves(&self, leaves: &[Fp]) -> Result<Vec<Fp>, CudaError> {
        let num_leaves = leaves.len();
        if num_leaves == 0 {
            return Ok(Vec::new());
        }

        let cuda_leaves: Vec<CudaFieldElement> =
            leaves.iter().map(CudaFieldElement::from).collect();

        let input_buffer = self
            .device
            .htod_sync_copy(&cuda_leaves)
            .map_err(|err| CudaError::AllocateMemory(err.to_string()))?;

        let mut output_buffer: CudaSlice<CudaFieldElement> = self
            .device
            .alloc_zeros(num_leaves)
            .map_err(|err| CudaError::AllocateMemory(err.to_string()))?;

        let hash_leaves_fn = self
            .device
            .get_func("merkle_tree", "hash_leaves")
            .ok_or(CudaError::FunctionError("hash_leaves not found".into()))?;

        let block_size = WARP_SIZE;
        let grid_size = num_leaves.div_ceil(block_size);

        let config = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            hash_leaves_fn.launch(
                config,
                (
                    &input_buffer,
                    &mut output_buffer,
                    &self.round_constants,
                    num_leaves as u32,
                ),
            )
        }
        .map_err(|err| CudaError::Launch(err.to_string()))?;

        let result = self
            .device
            .sync_reclaim(output_buffer)
            .map_err(|err| CudaError::RetrieveMemory(err.to_string()))?;

        Ok(result.into_iter().map(Fp::from).collect())
    }

    /// Build one layer of parents from children
    pub fn build_layer(&self, children: &[Fp]) -> Result<Vec<Fp>, CudaError> {
        let num_children = children.len();
        if num_children < 2 {
            return Err(CudaError::FunctionError(
                "Need at least 2 children".into(),
            ));
        }

        let num_parents = num_children / 2;

        let cuda_children: Vec<CudaFieldElement> =
            children.iter().map(CudaFieldElement::from).collect();

        let children_buffer = self
            .device
            .htod_sync_copy(&cuda_children)
            .map_err(|err| CudaError::AllocateMemory(err.to_string()))?;

        let mut parents_buffer: CudaSlice<CudaFieldElement> = self
            .device
            .alloc_zeros(num_parents)
            .map_err(|err| CudaError::AllocateMemory(err.to_string()))?;

        let compute_parents_fn = self
            .device
            .get_func("merkle_tree", "compute_parents")
            .ok_or(CudaError::FunctionError("compute_parents not found".into()))?;

        let block_size = WARP_SIZE;
        let grid_size = num_parents.div_ceil(block_size);

        let config = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            compute_parents_fn.launch(
                config,
                (
                    &children_buffer,
                    &mut parents_buffer,
                    &self.round_constants,
                    num_parents as u32,
                ),
            )
        }
        .map_err(|err| CudaError::Launch(err.to_string()))?;

        let result = self
            .device
            .sync_reclaim(parents_buffer)
            .map_err(|err| CudaError::RetrieveMemory(err.to_string()))?;

        Ok(result.into_iter().map(Fp::from).collect())
    }

    /// Build entire Merkle tree from leaves
    /// Returns (root, all_nodes) where all_nodes is in the format:
    /// [internal nodes (root first, then level by level)][leaves]
    pub fn build_tree(&self, leaves: &[Fp]) -> Result<(Fp, Vec<Fp>), CudaError> {
        if leaves.is_empty() {
            return Err(CudaError::FunctionError(
                "Cannot build tree from empty leaves".into(),
            ));
        }

        // Hash leaves first
        let mut current_layer = self.hash_leaves(leaves)?;

        // Pad to power of 2 if necessary
        while !current_layer.len().is_power_of_two() {
            // Safe: we check for empty above and only add elements
            let last = current_layer[current_layer.len() - 1].clone();
            current_layer.push(last);
        }

        // Build tree bottom-up
        let mut all_layers: Vec<Vec<Fp>> = vec![current_layer.clone()];

        while current_layer.len() > 1 {
            let parent_layer = self.build_layer(&current_layer)?;
            all_layers.push(parent_layer.clone());
            current_layer = parent_layer;
        }

        // The root is the single element in the last layer
        // Safe: we always have at least one layer with at least one element
        let root = current_layer[0].clone();

        // Reconstruct nodes array in the expected format
        let mut nodes = Vec::new();

        // Add internal nodes in reverse order (from root down)
        for layer in all_layers.iter().rev().skip(1) {
            nodes.extend(layer.iter().cloned());
        }

        // Add leaf hashes
        nodes.extend(all_layers[0].iter().cloned());

        Ok((root, nodes))
    }
}
