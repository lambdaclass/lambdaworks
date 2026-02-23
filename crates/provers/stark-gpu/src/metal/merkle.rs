use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::IsField;
use lambdaworks_math::traits::AsBytes;
use stark_platinum_prover::config::{BatchedMerkleTree, Commitment};

/// Build a batched Merkle tree from row-major evaluation data (CPU Keccak256).
pub fn cpu_batch_commit<F: IsField>(
    vectors: &[Vec<FieldElement<F>>],
) -> Option<(BatchedMerkleTree<F>, Commitment)>
where
    FieldElement<F>: AsBytes + Sync + Send,
{
    let tree = BatchedMerkleTree::<F>::build(vectors)?;
    let root = tree.root;
    Some((tree, root))
}

// =============================================================================
// GPU Keccak256 Merkle tree construction for Goldilocks field
// =============================================================================

#[cfg(all(target_os = "macos", feature = "metal"))]
use lambdaworks_gpu::metal::abstractions::{
    errors::MetalError,
    state::{void_ptr, DynamicMetalState},
};
#[cfg(all(target_os = "macos", feature = "metal"))]
use lambdaworks_math::field::fields::u64_goldilocks_field::Goldilocks64Field;

#[cfg(all(test, target_os = "macos", feature = "metal"))]
use crate::metal::canonical;
#[cfg(all(target_os = "macos", feature = "metal"))]
use crate::metal::to_raw_u64;

#[cfg(all(target_os = "macos", feature = "metal"))]
const KECCAK256_SHADER: &str = include_str!("shaders/keccak256.metal");

#[cfg(all(target_os = "macos", feature = "metal"))]
const TRANSPOSE_BITREV_SHADER: &str = include_str!("shaders/transpose_bitrev.metal");

/// Pre-compiled Metal state for GPU Keccak256 Merkle tree operations.
///
/// Caches compiled pipelines for leaf hashing, pair hashing, and grinding.
/// Create once and reuse across the entire prove call.
#[cfg(all(target_os = "macos", feature = "metal"))]
#[allow(dead_code)] // Some fields only used in #[cfg(test)] functions
pub struct GpuMerkleState {
    pub(crate) state: DynamicMetalState,
    hash_leaves_max_threads: u64,
    hash_pairs_max_threads: u64,
    transpose_max_threads: u64,
    fused_leaves_max_threads: u64,
    fused_leaves_paired_max_threads: u64,
    grind_max_threads: u64,
    transpose_bitrev_state: Option<TransposeBitrevState>,
    leaf_kernel: &'static str,
    pair_kernel: &'static str,
    fused_leaf_kernel: &'static str,
    fused_leaf_paired_kernel: &'static str,
    grind_kernel: Option<&'static str>,
    cpu_pair_hasher: fn(&[u8; 32], &[u8; 32]) -> [u8; 32],
}

/// Backward-compatible type alias.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub type GpuKeccakMerkleState = GpuMerkleState;

#[cfg(all(target_os = "macos", feature = "metal"))]
fn cpu_keccak_pair(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    use sha3::{Digest, Keccak256};
    let mut hasher = Keccak256::new();
    hasher.update(left);
    hasher.update(right);
    let mut result = [0u8; 32];
    result.copy_from_slice(&hasher.finalize());
    result
}

#[cfg(all(target_os = "macos", feature = "metal"))]
impl GpuMerkleState {
    pub fn new() -> Result<Self, MetalError> {
        Self::new_keccak()
    }

    pub fn new_keccak() -> Result<Self, MetalError> {
        let mut state = DynamicMetalState::new()?;
        state.load_library(KECCAK256_SHADER)?;
        let hash_leaves_max_threads = state.prepare_pipeline("keccak256_hash_leaves")?;
        let hash_pairs_max_threads = state.prepare_pipeline("keccak256_hash_pairs")?;
        let transpose_max_threads = state.prepare_pipeline("transpose_bitrev_goldilocks")?;
        let fused_leaves_max_threads =
            state.prepare_pipeline("keccak256_hash_leaves_from_columns")?;
        let fused_leaves_paired_max_threads =
            state.prepare_pipeline("keccak256_hash_leaves_from_columns_paired")?;
        let grind_max_threads = state.prepare_pipeline("keccak256_grind_nonce")?;

        let transpose_bitrev_state = TransposeBitrevState::new().ok();

        Ok(Self {
            state,
            hash_leaves_max_threads,
            hash_pairs_max_threads,
            transpose_max_threads,
            fused_leaves_max_threads,
            fused_leaves_paired_max_threads,
            grind_max_threads,
            transpose_bitrev_state,
            leaf_kernel: "keccak256_hash_leaves",
            pair_kernel: "keccak256_hash_pairs",
            fused_leaf_kernel: "keccak256_hash_leaves_from_columns",
            fused_leaf_paired_kernel: "keccak256_hash_leaves_from_columns_paired",
            grind_kernel: Some("keccak256_grind_nonce"),
            cpu_pair_hasher: cpu_keccak_pair,
        })
    }

    pub fn transpose_state(&self) -> Option<&TransposeBitrevState> {
        self.transpose_bitrev_state.as_ref()
    }
}

/// Pre-compiled Metal state for the transpose+bit-reverse kernel.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub struct TransposeBitrevState {
    pub(crate) state: DynamicMetalState,
    unpaired_max_threads: u64,
    paired_max_threads: u64,
}

#[cfg(all(target_os = "macos", feature = "metal"))]
impl TransposeBitrevState {
    pub fn new() -> Result<Self, MetalError> {
        let mut state = DynamicMetalState::new()?;
        state.load_library(TRANSPOSE_BITREV_SHADER)?;
        let unpaired_max_threads = state.prepare_pipeline("transpose_bitrev")?;
        let paired_max_threads = state.prepare_pipeline("transpose_bitrev_paired")?;
        Ok(Self {
            state,
            unpaired_max_threads,
            paired_max_threads,
        })
    }
}

// =============================================================================
// Shared helpers
// =============================================================================

/// Convert a flat byte buffer into a Vec of 32-byte hash arrays.
#[cfg(all(test, target_os = "macos", feature = "metal"))]
fn bytes_to_hashes(raw: &[u8]) -> Vec<[u8; 32]> {
    raw.chunks_exact(32)
        .map(|chunk| {
            let mut hash = [0u8; 32];
            hash.copy_from_slice(chunk);
            hash
        })
        .collect()
}

/// Read back the full tree from a GPU buffer, applying leaf padding if needed.
///
/// `leaf_rows` is the actual number of hashed leaves; `leaves_len` is the
/// padded power-of-two count. Padding duplicates the last real leaf hash.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub(crate) fn read_tree_nodes(
    tree_buf: &metal::Buffer,
    total_nodes: usize,
    leaf_rows: usize,
    leaves_len: usize,
) -> Vec<[u8; 32]> {
    let mut nodes = vec![[0u8; 32]; total_nodes];
    unsafe {
        let ptr = tree_buf.contents() as *const u8;
        std::ptr::copy_nonoverlapping(ptr, nodes.as_mut_ptr() as *mut u8, total_nodes * 32);
    }
    if leaf_rows < leaves_len {
        let last_real = leaves_len - 1 + leaf_rows - 1;
        let pad_hash = nodes[last_real];
        for node in nodes
            .iter_mut()
            .take(leaves_len - 1 + leaves_len)
            .skip(last_real + 1)
        {
            *node = pad_hash;
        }
    }
    nodes
}

/// Encode bottom-up pair-hash tree levels into an existing command buffer.
///
/// Dispatches GPU compute encoders for each tree level from `level_begin..=level_end`
/// upward. Does NOT commit — caller manages the command buffer lifecycle.
#[cfg(all(target_os = "macos", feature = "metal"))]
fn encode_tree_levels(
    command_buffer: &metal::CommandBufferRef,
    tree_buf: &metal::Buffer,
    pair_hash_pipeline: &metal::ComputePipelineStateRef,
    threads_per_group: u64,
    mut level_begin: usize,
    mut level_end: usize,
) {
    use metal::MTLSize;
    while level_begin != level_end {
        let new_level_begin = level_begin / 2;
        let num_pairs = (level_end - level_begin).div_ceil(2);
        let num_pairs_u32 = num_pairs as u32;

        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pair_hash_pipeline);
        encoder.set_buffer(0, Some(tree_buf), (level_begin * 32) as u64);
        encoder.set_buffer(1, Some(tree_buf), (new_level_begin * 32) as u64);
        encoder.set_bytes(
            2,
            std::mem::size_of::<u32>() as u64,
            void_ptr(&num_pairs_u32),
        );
        let thread_groups_count = (num_pairs as u64).div_ceil(threads_per_group);
        encoder.dispatch_thread_groups(
            MTLSize::new(thread_groups_count, 1, 1),
            MTLSize::new(threads_per_group, 1, 1),
        );
        encoder.end_encoding();

        level_end = level_begin - 1;
        level_begin = new_level_begin;
    }
}

/// Submit a command buffer, wait for completion, and check for errors.
#[cfg(all(target_os = "macos", feature = "metal"))]
fn commit_and_wait(
    command_buffer: &metal::CommandBufferRef,
    error_msg: &str,
) -> Result<(), MetalError> {
    use metal::MTLCommandBufferStatus;
    command_buffer.commit();
    command_buffer.wait_until_completed();
    if command_buffer.status() == MTLCommandBufferStatus::Error {
        return Err(MetalError::ExecutionError(error_msg.to_string()));
    }
    Ok(())
}

/// Blit-copy column buffers into a single flat column-major GPU buffer.
#[cfg(all(target_os = "macos", feature = "metal"))]
fn blit_columns_to_flat(
    command_buffer: &metal::CommandBufferRef,
    column_buffers: &[&metal::Buffer],
    flat_cols: &metal::Buffer,
    num_rows: usize,
) {
    let blit = command_buffer.new_blit_command_encoder();
    for (i, col_buf) in column_buffers.iter().enumerate() {
        blit.copy_from_buffer(
            col_buf,
            0,
            flat_cols,
            (i * num_rows * std::mem::size_of::<u64>()) as u64,
            col_buf.length(),
        );
    }
    blit.end_encoding();
}

// =============================================================================
// Transpose operations
// =============================================================================

/// Transpose column GPU buffers to row-major buffer with bit-reversed row ordering.
///
/// When `paired` is false: output has `num_rows` rows, each with `num_cols` elements.
/// When `paired` is true: merges consecutive bit-reversed rows into `num_rows/2` rows
/// of `2 * num_cols` elements each.
#[cfg(all(target_os = "macos", feature = "metal"))]
#[allow(dead_code)]
fn gpu_transpose_bitrev_dispatch(
    column_buffers: &[&metal::Buffer],
    num_rows: usize,
    paired: bool,
    transpose_state: &TransposeBitrevState,
) -> Result<metal::Buffer, MetalError> {
    use metal::{MTLResourceOptions, MTLSize};

    let num_cols = column_buffers.len();
    let log_n = num_rows.trailing_zeros();

    let flat_cols_size = (num_rows * num_cols * std::mem::size_of::<u64>()) as u64;
    let flat_cols = transpose_state
        .state
        .device()
        .new_buffer(flat_cols_size, MTLResourceOptions::StorageModeShared);

    let command_buffer = transpose_state.state.command_queue().new_command_buffer();
    blit_columns_to_flat(command_buffer, column_buffers, &flat_cols, num_rows);

    // Compute output dimensions and select kernel
    let (output_rows, output_cols_per_row, kernel_name, max_threads) = if paired {
        let nr = num_rows / 2;
        (
            nr,
            2 * num_cols,
            "transpose_bitrev_paired",
            transpose_state.paired_max_threads,
        )
    } else {
        (
            num_rows,
            num_cols,
            "transpose_bitrev",
            transpose_state.unpaired_max_threads,
        )
    };

    let output_size = (output_rows * output_cols_per_row * std::mem::size_of::<u64>()) as u64;
    let output_buf = transpose_state
        .state
        .device()
        .new_buffer(output_size, MTLResourceOptions::StorageModeShared);

    let num_cols_u32 = num_cols as u32;
    let num_rows_u32 = num_rows as u32;
    let buf_num_cols = transpose_state
        .state
        .alloc_buffer_with_data(std::slice::from_ref(&num_cols_u32))?;
    let buf_num_rows = transpose_state
        .state
        .alloc_buffer_with_data(std::slice::from_ref(&num_rows_u32))?;
    let buf_log_n = transpose_state
        .state
        .alloc_buffer_with_data(std::slice::from_ref(&log_n))?;

    let pipeline = transpose_state
        .state
        .get_pipeline_ref(kernel_name)
        .ok_or_else(|| MetalError::FunctionError(kernel_name.to_string()))?;

    let threads_per_group = max_threads.min(256);
    let thread_groups = (output_rows as u64).div_ceil(threads_per_group);

    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(&flat_cols), 0);
    encoder.set_buffer(1, Some(&output_buf), 0);
    encoder.set_buffer(2, Some(&buf_num_cols), 0);
    encoder.set_buffer(3, Some(&buf_num_rows), 0);
    encoder.set_buffer(4, Some(&buf_log_n), 0);
    encoder.dispatch_thread_groups(
        MTLSize::new(thread_groups, 1, 1),
        MTLSize::new(threads_per_group, 1, 1),
    );
    encoder.end_encoding();

    commit_and_wait(command_buffer, &format!("GPU {kernel_name} command error"))?;

    Ok(output_buf)
}

// =============================================================================
// Test-only: individual leaf hashing and tree building functions
// =============================================================================

/// Hash row-major Goldilocks field element data into 32-byte leaf digests on GPU.
#[cfg(all(test, target_os = "macos", feature = "metal"))]
fn gpu_hash_leaves_goldilocks(
    rows: &[Vec<FieldElement<Goldilocks64Field>>],
    keccak_state: &GpuKeccakMerkleState,
) -> Result<Vec<[u8; 32]>, MetalError> {
    if rows.is_empty() {
        return Ok(Vec::new());
    }
    let num_rows = rows.len();
    let num_cols = rows[0].len();
    let flat_data: Vec<u64> = rows
        .iter()
        .flat_map(|row| row.iter().map(canonical))
        .collect();
    gpu_hash_leaves_flat(&flat_data, num_rows, num_cols, keccak_state)
}

/// Hash pre-flattened row-major u64 data into 32-byte leaf digests on GPU.
#[cfg(all(test, target_os = "macos", feature = "metal"))]
fn gpu_hash_leaves_flat(
    flat_data: &[u64],
    num_rows: usize,
    num_cols: usize,
    keccak_state: &GpuKeccakMerkleState,
) -> Result<Vec<[u8; 32]>, MetalError> {
    if num_rows == 0 {
        return Ok(Vec::new());
    }
    let buf_data = keccak_state.state.alloc_buffer_with_data(flat_data)?;
    let buf_output = keccak_state.state.alloc_buffer(num_rows * 32)?;
    let num_cols_u32 = num_cols as u32;
    let num_rows_u32 = num_rows as u32;
    let buf_num_cols = keccak_state
        .state
        .alloc_buffer_with_data(std::slice::from_ref(&num_cols_u32))?;
    let buf_num_rows = keccak_state
        .state
        .alloc_buffer_with_data(std::slice::from_ref(&num_rows_u32))?;

    keccak_state.state.execute_compute(
        keccak_state.leaf_kernel,
        &[&buf_data, &buf_output, &buf_num_cols, &buf_num_rows],
        num_rows as u64,
        keccak_state.hash_leaves_max_threads,
    )?;

    let raw_output: Vec<u8> = unsafe { keccak_state.state.read_buffer(&buf_output, num_rows * 32) };
    Ok(bytes_to_hashes(&raw_output))
}

/// Hash pairs of 32-byte child nodes into parent nodes.
/// Falls back to CPU for fewer than 64 pairs.
#[cfg(all(test, target_os = "macos", feature = "metal"))]
fn gpu_hash_tree_level(
    children: &[[u8; 32]],
    keccak_state: &GpuKeccakMerkleState,
) -> Result<Vec<[u8; 32]>, MetalError> {
    let num_pairs = children.len() / 2;
    if num_pairs == 0 {
        return Ok(Vec::new());
    }
    if num_pairs < 64 {
        let hash_pair = keccak_state.cpu_pair_hasher;
        let parents: Vec<[u8; 32]> = (0..num_pairs)
            .map(|i| hash_pair(&children[2 * i], &children[2 * i + 1]))
            .collect();
        return Ok(parents);
    }

    let flat_children: Vec<u8> = children.iter().flat_map(|h| h.iter().copied()).collect();
    let buf_children = keccak_state.state.alloc_buffer_with_data(&flat_children)?;
    let buf_parents = keccak_state.state.alloc_buffer(num_pairs * 32)?;
    let num_pairs_u32 = num_pairs as u32;
    let buf_num_pairs = keccak_state
        .state
        .alloc_buffer_with_data(std::slice::from_ref(&num_pairs_u32))?;

    keccak_state.state.execute_compute(
        keccak_state.pair_kernel,
        &[&buf_children, &buf_parents, &buf_num_pairs],
        num_pairs as u64,
        keccak_state.hash_pairs_max_threads,
    )?;

    let raw_output: Vec<u8> =
        unsafe { keccak_state.state.read_buffer(&buf_parents, num_pairs * 32) };
    Ok(bytes_to_hashes(&raw_output))
}

/// Build a complete Merkle tree from leaf hashes, returning the flat node array and root.
///
/// Node layout matches `MerkleTree::build()`: `[inner_nodes | leaves]` where `nodes[0]` = root.
/// Uses a single GPU command buffer with all tree levels as sequential compute dispatches.
#[cfg(all(test, target_os = "macos", feature = "metal"))]
fn gpu_build_merkle_tree(
    leaf_hashes: &[[u8; 32]],
    keccak_state: &GpuKeccakMerkleState,
) -> Result<(Vec<[u8; 32]>, [u8; 32]), MetalError> {
    use metal::{MTLResourceOptions, MTLSize};

    if leaf_hashes.is_empty() {
        return Err(MetalError::ExecutionError("Empty leaf hashes".to_string()));
    }

    let mut leaves: Vec<[u8; 32]> = leaf_hashes.to_vec();
    while !leaves.len().is_power_of_two() {
        leaves.push(*leaves.last().unwrap());
    }

    let leaves_len = leaves.len();
    let total_nodes = 2 * leaves_len - 1;

    let tree_buf = keccak_state.state.device().new_buffer(
        (total_nodes * 32) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Copy leaf hashes into the second half of the GPU buffer
    unsafe {
        let ptr = tree_buf.contents() as *mut u8;
        let leaf_offset = (leaves_len - 1) * 32;
        std::ptr::copy_nonoverlapping(
            leaves.as_ptr() as *const u8,
            ptr.add(leaf_offset),
            leaves_len * 32,
        );
    }

    let pair_hash_pipeline = keccak_state
        .state
        .get_pipeline_ref(keccak_state.pair_kernel)
        .ok_or_else(|| MetalError::FunctionError(keccak_state.pair_kernel.to_string()))?;
    let threads_per_group = keccak_state.hash_pairs_max_threads.min(256);

    let command_buffer = keccak_state.state.command_queue().new_command_buffer();

    // Encode tree levels, breaking out of GPU for very small levels
    let mut level_begin = leaves_len - 1;
    let mut level_end = 2 * level_begin;

    while level_begin != level_end {
        let new_level_begin = level_begin / 2;
        let num_pairs = (level_end - level_begin).div_ceil(2);

        if num_pairs < 64 {
            break;
        }

        let num_pairs_u32 = num_pairs as u32;
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pair_hash_pipeline);
        encoder.set_buffer(0, Some(&tree_buf), (level_begin * 32) as u64);
        encoder.set_buffer(1, Some(&tree_buf), (new_level_begin * 32) as u64);
        encoder.set_bytes(
            2,
            std::mem::size_of::<u32>() as u64,
            void_ptr(&num_pairs_u32),
        );
        let thread_groups_count = (num_pairs as u64).div_ceil(threads_per_group);
        encoder.dispatch_thread_groups(
            MTLSize::new(thread_groups_count, 1, 1),
            MTLSize::new(threads_per_group, 1, 1),
        );
        encoder.end_encoding();

        level_end = level_begin - 1;
        level_begin = new_level_begin;
    }

    commit_and_wait(command_buffer, "GPU tree build command buffer error")?;

    let mut nodes = vec![[0u8; 32]; total_nodes];
    unsafe {
        let ptr = tree_buf.contents() as *const u8;
        std::ptr::copy_nonoverlapping(ptr, nodes.as_mut_ptr() as *mut u8, total_nodes * 32);
    }

    // Finish remaining small levels on CPU
    while level_begin != level_end {
        let new_level_begin = level_begin / 2;
        let children_slice = &nodes[level_begin..=level_end];
        let parents = gpu_hash_tree_level(children_slice, keccak_state)?;
        for (i, parent) in parents.iter().enumerate() {
            nodes[new_level_begin + i] = *parent;
        }
        level_end = level_begin - 1;
        level_begin = new_level_begin;
    }

    let root = nodes[0];
    Ok((nodes, root))
}

/// Encode leaf hashing + tree building into an existing Metal command buffer.
///
/// Encodes compute dispatches for hashing leaves from `buf_data` into `tree_buf`
/// leaf positions, then building all tree levels bottom-up.
///
/// Does NOT commit or wait -- the caller manages the command buffer lifecycle.
/// This enables fusing Merkle encoding with prior compute work (e.g., FRI fold).
#[cfg(all(target_os = "macos", feature = "metal"))]
pub(crate) fn encode_hash_and_build_tree(
    command_buffer: &metal::CommandBufferRef,
    buf_data: &metal::Buffer,
    num_rows: usize,
    num_cols: usize,
    tree_buf: &metal::Buffer,
    leaves_len: usize,
    keccak_state: &GpuMerkleState,
) -> Result<(), MetalError> {
    use metal::MTLSize;

    let leaf_hash_pipeline = keccak_state
        .state
        .get_pipeline_ref(keccak_state.leaf_kernel)
        .ok_or_else(|| MetalError::FunctionError(keccak_state.leaf_kernel.to_string()))?;
    let pair_hash_pipeline = keccak_state
        .state
        .get_pipeline_ref(keccak_state.pair_kernel)
        .ok_or_else(|| MetalError::FunctionError(keccak_state.pair_kernel.to_string()))?;

    let num_cols_u32 = num_cols as u32;
    let num_rows_u32 = num_rows as u32;
    let buf_num_cols = keccak_state
        .state
        .alloc_buffer_with_data(std::slice::from_ref(&num_cols_u32))?;
    let buf_num_rows = keccak_state
        .state
        .alloc_buffer_with_data(std::slice::from_ref(&num_rows_u32))?;

    // Hash leaves into tree buffer at leaf positions
    {
        let leaf_byte_offset = ((leaves_len - 1) * 32) as u64;
        let threads_per_group = keccak_state.hash_leaves_max_threads.min(256);
        let thread_groups = (num_rows as u64).div_ceil(threads_per_group);

        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(leaf_hash_pipeline);
        encoder.set_buffer(0, Some(buf_data), 0);
        encoder.set_buffer(1, Some(tree_buf), leaf_byte_offset);
        encoder.set_buffer(2, Some(&buf_num_cols), 0);
        encoder.set_buffer(3, Some(&buf_num_rows), 0);
        encoder.dispatch_thread_groups(
            MTLSize::new(thread_groups, 1, 1),
            MTLSize::new(threads_per_group, 1, 1),
        );
        encoder.end_encoding();
    }

    // Build tree levels bottom-up
    let threads_per_group = keccak_state.hash_pairs_max_threads.min(256);
    let level_begin = leaves_len - 1;
    let level_end = 2 * level_begin;
    encode_tree_levels(
        command_buffer,
        tree_buf,
        pair_hash_pipeline,
        threads_per_group,
        level_begin,
        level_end,
    );

    Ok(())
}

/// Hash leaves from a GPU buffer and build tree in a single GPU command buffer.
#[cfg(all(target_os = "macos", feature = "metal"))]
fn gpu_hash_and_build_tree_from_buffer(
    buf_data: &metal::Buffer,
    num_rows: usize,
    num_cols: usize,
    keccak_state: &GpuKeccakMerkleState,
) -> Result<(Vec<[u8; 32]>, [u8; 32]), MetalError> {
    use metal::MTLResourceOptions;

    if num_rows == 0 {
        return Err(MetalError::ExecutionError("Empty data".to_string()));
    }

    let leaves_len = num_rows.next_power_of_two();
    let total_nodes = 2 * leaves_len - 1;

    let tree_buf = keccak_state.state.device().new_buffer(
        (total_nodes * 32) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let command_buffer = keccak_state.state.command_queue().new_command_buffer();
    encode_hash_and_build_tree(
        command_buffer,
        buf_data,
        num_rows,
        num_cols,
        &tree_buf,
        leaves_len,
        keccak_state,
    )?;
    commit_and_wait(command_buffer, "GPU hash+tree command buffer error")?;

    let nodes = read_tree_nodes(&tree_buf, total_nodes, num_rows, leaves_len);
    let root = nodes[0];
    Ok((nodes, root))
}

// =============================================================================
// Commit from column buffers (fused transpose+hash pipeline)
// =============================================================================

/// GPU Merkle commit from column GPU buffers (FFT output).
///
/// Takes per-column Metal Buffers and builds a Merkle tree by fusing
/// transpose + bit-reverse + hash into a single GPU command buffer.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn gpu_batch_commit_from_column_buffers(
    column_buffers: &[&metal::Buffer],
    num_rows: usize,
    keccak_state: &GpuKeccakMerkleState,
) -> Option<(BatchedMerkleTree<Goldilocks64Field>, Commitment)> {
    if column_buffers.is_empty() || num_rows == 0 {
        return None;
    }
    let num_cols = column_buffers.len();
    let transpose_state = keccak_state.transpose_state()?;
    let (nodes, root) = gpu_transpose_hash_and_build_tree(
        column_buffers,
        num_rows,
        num_cols,
        false,
        keccak_state,
        transpose_state,
    )
    .ok()?;
    let tree = BatchedMerkleTree::<Goldilocks64Field>::from_nodes(nodes)?;
    Some((tree, root))
}

/// GPU Merkle commit from column GPU buffers with paired-row layout (composition poly).
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn gpu_batch_commit_paired_from_column_buffers(
    column_buffers: &[&metal::Buffer],
    lde_len: usize,
    keccak_state: &GpuKeccakMerkleState,
) -> Option<(BatchedMerkleTree<Goldilocks64Field>, Commitment)> {
    if column_buffers.is_empty() || lde_len == 0 {
        return None;
    }
    let num_cols = column_buffers.len();
    let transpose_state = keccak_state.transpose_state()?;
    let (nodes, root) = gpu_transpose_hash_and_build_tree(
        column_buffers,
        lde_len,
        num_cols,
        true,
        keccak_state,
        transpose_state,
    )
    .ok()?;
    let tree = BatchedMerkleTree::<Goldilocks64Field>::from_nodes(nodes)?;
    Some((tree, root))
}

/// Fused pipeline: blit + fused hash-from-columns + build tree in one command buffer.
///
/// Uses a fused kernel that reads column-major data with bit-reversed indexing
/// and hashes leaves in a single dispatch, eliminating the intermediate transpose
/// buffer entirely. When `paired` is true, merges two consecutive bit-reversed
/// rows into each leaf.
#[cfg(all(target_os = "macos", feature = "metal"))]
fn gpu_transpose_hash_and_build_tree(
    column_buffers: &[&metal::Buffer],
    num_rows: usize,
    num_cols: usize,
    paired: bool,
    keccak_state: &GpuKeccakMerkleState,
    _transpose_state: &TransposeBitrevState,
) -> Result<(Vec<[u8; 32]>, [u8; 32]), MetalError> {
    use metal::{MTLResourceOptions, MTLSize};

    let log_n = num_rows.trailing_zeros();
    let leaf_rows = if paired { num_rows / 2 } else { num_rows };
    let leaves_len = leaf_rows.next_power_of_two();
    let total_nodes = 2 * leaves_len - 1;

    let flat_cols_size = (num_rows * num_cols * std::mem::size_of::<u64>()) as u64;
    let flat_cols = keccak_state
        .state
        .device()
        .new_buffer(flat_cols_size, MTLResourceOptions::StorageModeShared);
    let tree_buf = keccak_state.state.device().new_buffer(
        (total_nodes * 32) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let (fused_kernel, fused_max_threads) = if paired {
        (
            keccak_state.fused_leaf_paired_kernel,
            keccak_state.fused_leaves_paired_max_threads,
        )
    } else {
        (
            keccak_state.fused_leaf_kernel,
            keccak_state.fused_leaves_max_threads,
        )
    };
    let fused_pipeline = keccak_state
        .state
        .get_pipeline_ref(fused_kernel)
        .ok_or_else(|| MetalError::FunctionError(fused_kernel.to_string()))?;
    let pair_hash_pipeline = keccak_state
        .state
        .get_pipeline_ref(keccak_state.pair_kernel)
        .ok_or_else(|| MetalError::FunctionError(keccak_state.pair_kernel.to_string()))?;

    let command_buffer = keccak_state.state.command_queue().new_command_buffer();

    // Phase 1: Blit-copy column buffers into flat column-major buffer
    blit_columns_to_flat(command_buffer, column_buffers, &flat_cols, num_rows);

    // Phase 2: Fused transpose+hash -- reads column-major with bitrev, outputs leaf hashes
    {
        let leaf_byte_offset = ((leaves_len - 1) * 32) as u64;
        let threads_per_group = fused_max_threads.min(256);
        let thread_groups = (leaf_rows as u64).div_ceil(threads_per_group);
        let num_cols_u32 = num_cols as u32;
        let num_rows_u32 = num_rows as u32;

        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(fused_pipeline);
        encoder.set_buffer(0, Some(&flat_cols), 0);
        encoder.set_buffer(1, Some(&tree_buf), leaf_byte_offset);
        encoder.set_bytes(
            2,
            std::mem::size_of::<u32>() as u64,
            void_ptr(&num_cols_u32),
        );
        encoder.set_bytes(
            3,
            std::mem::size_of::<u32>() as u64,
            void_ptr(&num_rows_u32),
        );
        encoder.set_bytes(4, std::mem::size_of::<u32>() as u64, void_ptr(&log_n));
        encoder.dispatch_thread_groups(
            MTLSize::new(thread_groups, 1, 1),
            MTLSize::new(threads_per_group, 1, 1),
        );
        encoder.end_encoding();
    }

    // Phase 3: Build tree levels (all on GPU)
    let threads_per_group = keccak_state.hash_pairs_max_threads.min(256);
    let level_begin = leaves_len - 1;
    let level_end = 2 * level_begin;
    encode_tree_levels(
        command_buffer,
        &tree_buf,
        pair_hash_pipeline,
        threads_per_group,
        level_begin,
        level_end,
    );

    commit_and_wait(command_buffer, "GPU fused-hash+tree command error")?;

    let nodes = read_tree_nodes(&tree_buf, total_nodes, leaf_rows, leaves_len);
    let root = nodes[0];
    Ok((nodes, root))
}

// =============================================================================
// FRI layer commit
// =============================================================================

/// GPU FRI-layer Merkle commit from a GPU buffer of bit-reversed evaluations (u64 layout).
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn gpu_fri_layer_commit_from_buffer(
    eval_buffer: &metal::Buffer,
    eval_len: usize,
    keccak_state: &GpuKeccakMerkleState,
) -> Result<(BatchedMerkleTree<Goldilocks64Field>, Commitment), MetalError> {
    let num_leaves = eval_len / 2;
    let (nodes, root) =
        gpu_hash_and_build_tree_from_buffer(eval_buffer, num_leaves, 2, keccak_state)?;
    let tree = BatchedMerkleTree::<Goldilocks64Field>::from_nodes(nodes)
        .ok_or_else(|| MetalError::ExecutionError("Failed to build FRI Merkle tree".into()))?;
    Ok((tree, root))
}

/// GPU FRI-layer Merkle commit from CPU-side bit-reversed evaluations.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn gpu_fri_layer_commit(
    evaluation: &[FieldElement<Goldilocks64Field>],
    keccak_state: &GpuKeccakMerkleState,
) -> Result<(BatchedMerkleTree<Goldilocks64Field>, Commitment), MetalError> {
    let num_leaves = evaluation.len() / 2;
    let flat_data: Vec<u64> = to_raw_u64(evaluation);
    let buf_data = keccak_state.state.alloc_buffer_with_data(&flat_data)?;
    let (nodes, root) =
        gpu_hash_and_build_tree_from_buffer(&buf_data, num_leaves, 2, keccak_state)?;
    let tree = BatchedMerkleTree::<Goldilocks64Field>::from_nodes(nodes)
        .ok_or_else(|| MetalError::ExecutionError("Failed to build FRI Merkle tree".into()))?;
    Ok((tree, root))
}

// =============================================================================
// Test-only: CPU-data commit paths (column-major LDE input)
// =============================================================================

/// GPU Merkle commit from column-major LDE evaluations.
///
/// Computes bit-reversed row-major flat data on CPU, then hashes + builds
/// tree on GPU in a single command buffer.
#[cfg(all(test, target_os = "macos", feature = "metal"))]
fn gpu_batch_commit_goldilocks(
    lde_columns: &[Vec<FieldElement<Goldilocks64Field>>],
    keccak_state: &GpuKeccakMerkleState,
) -> Option<(BatchedMerkleTree<Goldilocks64Field>, Commitment)> {
    let num_cols = lde_columns.len();
    let num_rows = lde_columns[0].len();
    let log_n = num_rows.trailing_zeros();

    let mut flat_data: Vec<u64> = Vec::with_capacity(num_rows * num_cols);
    for i in 0..num_rows {
        let src_idx = i.reverse_bits() >> (usize::BITS - log_n);
        for col in lde_columns {
            flat_data.push(canonical(&col[src_idx]));
        }
    }

    let buf_data = keccak_state.state.alloc_buffer_with_data(&flat_data).ok()?;
    let (nodes, root) =
        gpu_hash_and_build_tree_from_buffer(&buf_data, num_rows, num_cols, keccak_state).ok()?;
    let tree = BatchedMerkleTree::<Goldilocks64Field>::from_nodes(nodes)?;
    Some((tree, root))
}

/// GPU Merkle commit for composition polynomial (paired rows variant).
///
/// Transposes, bit-reverses, and pairs consecutive rows before hashing.
#[cfg(all(test, target_os = "macos", feature = "metal"))]
fn gpu_batch_commit_paired_goldilocks(
    lde_evaluations: &[Vec<FieldElement<Goldilocks64Field>>],
    keccak_state: &GpuKeccakMerkleState,
) -> Option<(BatchedMerkleTree<Goldilocks64Field>, Commitment)> {
    let num_cols = lde_evaluations.len();
    let lde_len = lde_evaluations[0].len();
    let num_merged_rows = lde_len / 2;
    let cols_per_merged_row = 2 * num_cols;
    let log_n = lde_len.trailing_zeros();

    let mut flat_data: Vec<u64> = Vec::with_capacity(num_merged_rows * cols_per_merged_row);
    for i in 0..num_merged_rows {
        let idx0 = (2 * i).reverse_bits() >> (usize::BITS - log_n);
        let idx1 = (2 * i + 1).reverse_bits() >> (usize::BITS - log_n);
        for col in lde_evaluations {
            flat_data.push(canonical(&col[idx0]));
        }
        for col in lde_evaluations {
            flat_data.push(canonical(&col[idx1]));
        }
    }

    let buf_data = keccak_state.state.alloc_buffer_with_data(&flat_data).ok()?;
    let (nodes, root) = gpu_hash_and_build_tree_from_buffer(
        &buf_data,
        num_merged_rows,
        cols_per_merged_row,
        keccak_state,
    )
    .ok()?;
    let tree = BatchedMerkleTree::<Goldilocks64Field>::from_nodes(nodes)?;
    Some((tree, root))
}

// =============================================================================
// Test-only: GPU transpose (returns Vec<Vec<FieldElement>>)
// =============================================================================

/// Transpose column-major LDE data to row-major bit-reversed layout on GPU.
#[cfg(all(test, target_os = "macos", feature = "metal"))]
fn gpu_transpose_bitrev(
    columns: &[Vec<FieldElement<Goldilocks64Field>>],
    keccak_state: &GpuKeccakMerkleState,
) -> Result<Vec<Vec<FieldElement<Goldilocks64Field>>>, MetalError> {
    if columns.is_empty() {
        return Ok(Vec::new());
    }
    let num_cols = columns.len();
    let num_rows = columns[0].len();

    let flat_cols: Vec<u64> = columns
        .iter()
        .flat_map(|col| col.iter().map(canonical))
        .collect();

    let buf_cols = keccak_state.state.alloc_buffer_with_data(&flat_cols)?;
    let buf_rows = keccak_state
        .state
        .alloc_buffer(num_rows * num_cols * std::mem::size_of::<u64>())?;
    let num_cols_u32 = num_cols as u32;
    let num_rows_u32 = num_rows as u32;
    let buf_num_cols = keccak_state
        .state
        .alloc_buffer_with_data(std::slice::from_ref(&num_cols_u32))?;
    let buf_num_rows = keccak_state
        .state
        .alloc_buffer_with_data(std::slice::from_ref(&num_rows_u32))?;

    keccak_state.state.execute_compute(
        "transpose_bitrev_goldilocks",
        &[&buf_cols, &buf_rows, &buf_num_cols, &buf_num_rows],
        num_rows as u64,
        keccak_state.transpose_max_threads,
    )?;

    let raw_rows: Vec<u64> = unsafe {
        keccak_state
            .state
            .read_buffer(&buf_rows, num_rows * num_cols)
    };

    let rows: Vec<Vec<FieldElement<Goldilocks64Field>>> = raw_rows
        .chunks_exact(num_cols)
        .map(|row| row.iter().map(|&v| FieldElement::from(v)).collect())
        .collect();

    Ok(rows)
}

// =============================================================================
// GPU Grinding
// =============================================================================

/// Search for a valid grinding nonce on the GPU.
///
/// Each GPU thread computes `Keccak256(inner_hash || nonce_be)` and checks
/// if the first 8 bytes (big-endian u64) < `limit`. Returns `None` if the
/// backend has no GPU grind kernel.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn gpu_generate_nonce(
    seed: &[u8; 32],
    grinding_factor: u8,
    keccak_state: &GpuMerkleState,
) -> Option<u64> {
    use lambdaworks_gpu::metal::abstractions::state::MetalState;
    use sha3::{Digest, Keccak256};

    if grinding_factor == 0 {
        return Some(0);
    }

    let grind_kernel = keccak_state.grind_kernel?;

    // Compute inner_hash on CPU (single hash, not worth GPU)
    // Matches the grinding module's GRINDING_PREFIX constant.
    const GRINDING_PREFIX: [u8; 8] = [0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xed];
    let mut inner_data = [0u8; 41];
    inner_data[0..8].copy_from_slice(&GRINDING_PREFIX);
    inner_data[8..40].copy_from_slice(seed);
    inner_data[40] = grinding_factor;
    let digest = Keccak256::digest(inner_data);
    let inner_hash: [u8; 32] = digest[..32].try_into().unwrap();

    let limit: u64 = 1u64 << (64 - grinding_factor);

    let inner_hash_buf = keccak_state
        .state
        .alloc_buffer_with_data(&inner_hash)
        .ok()?;
    let limit_buf = keccak_state.state.alloc_buffer_with_data(&[limit]).ok()?;

    let batch_size: u64 = 1 << 20;
    let mut batch_offset: u64 = 0;

    loop {
        let result_buf = keccak_state
            .state
            .alloc_buffer_with_data(&[u32::MAX])
            .ok()?;
        let offset_buf = keccak_state
            .state
            .alloc_buffer_with_data(&[batch_offset])
            .ok()?;

        keccak_state
            .state
            .execute_compute(
                grind_kernel,
                &[&inner_hash_buf, &result_buf, &limit_buf, &offset_buf],
                batch_size,
                keccak_state.grind_max_threads,
            )
            .ok()?;

        let result: Vec<u32> = MetalState::retrieve_contents(&result_buf);
        if result[0] != u32::MAX {
            return Some(batch_offset + result[0] as u64);
        }

        batch_offset += batch_size;
        if batch_offset >= u64::MAX - batch_size {
            return None;
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use lambdaworks_math::field::element::FieldElement;
    use lambdaworks_math::field::fields::u64_goldilocks_field::Goldilocks64Field;

    type FpE = FieldElement<Goldilocks64Field>;

    #[test]
    fn cpu_batch_commit_produces_valid_tree() {
        let col1: Vec<FpE> = (0..8).map(|i| FpE::from(i as u64)).collect();
        let col2: Vec<FpE> = (0..8).map(|i| FpE::from(i as u64 + 100)).collect();
        let (tree, root) = cpu_batch_commit(&[col1, col2]).unwrap();
        assert_ne!(root, [0u8; 32]);
        assert!(tree.root != [0u8; 32]);
    }
}

#[cfg(all(test, target_os = "macos", feature = "metal"))]
mod gpu_tests {
    use super::*;
    use lambdaworks_math::field::element::FieldElement;
    use lambdaworks_math::field::fields::u64_goldilocks_field::Goldilocks64Field;
    use lambdaworks_math::traits::AsBytes;
    use sha3::{Digest, Keccak256};
    use stark_platinum_prover::trace::columns2rows_bit_reversed;

    type FpE = FieldElement<Goldilocks64Field>;

    fn cpu_hash_row(row: &[FpE]) -> [u8; 32] {
        let mut hasher = Keccak256::new();
        for element in row.iter() {
            hasher.update(element.as_bytes());
        }
        let mut result = [0u8; 32];
        result.copy_from_slice(&hasher.finalize());
        result
    }

    fn cpu_hash_pair(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
        let mut hasher = Keccak256::new();
        hasher.update(left);
        hasher.update(right);
        let mut result = [0u8; 32];
        result.copy_from_slice(&hasher.finalize());
        result
    }

    fn make_columns(num_cols: usize, num_rows: usize) -> Vec<Vec<FpE>> {
        (0..num_cols)
            .map(|c| {
                (0..num_rows)
                    .map(|r| FpE::from((c * num_rows + r + 1) as u64))
                    .collect()
            })
            .collect()
    }

    fn make_rows(num_rows: usize, num_cols: usize) -> Vec<Vec<FpE>> {
        (0..num_rows)
            .map(|r| {
                (0..num_cols)
                    .map(|c| FpE::from((r * num_cols + c + 1) as u64))
                    .collect()
            })
            .collect()
    }

    fn columns_to_metal_buffers(
        columns: &[Vec<FpE>],
        device: &metal::DeviceRef,
    ) -> Vec<metal::Buffer> {
        use metal::MTLResourceOptions;
        columns
            .iter()
            .map(|col| {
                let data: Vec<u64> = to_raw_u64(col);
                device.new_buffer_with_data(
                    data.as_ptr() as *const _,
                    (data.len() * std::mem::size_of::<u64>()) as u64,
                    MTLResourceOptions::StorageModeShared,
                )
            })
            .collect()
    }

    fn assert_proofs_match(
        cpu_tree: &BatchedMerkleTree<Goldilocks64Field>,
        gpu_tree: &BatchedMerkleTree<Goldilocks64Field>,
        positions: &[usize],
    ) {
        for &pos in positions {
            let cpu_proof = cpu_tree.get_proof_by_pos(pos);
            let gpu_proof = gpu_tree.get_proof_by_pos(pos);
            assert_eq!(
                cpu_proof.is_some(),
                gpu_proof.is_some(),
                "Proof availability mismatch at pos {pos}"
            );
            if let (Some(cp), Some(gp)) = (cpu_proof, gpu_proof) {
                assert_eq!(
                    cp.merkle_path, gp.merkle_path,
                    "Proof path mismatch at pos {pos}"
                );
            }
        }
    }

    #[test]
    fn gpu_keccak256_leaf_hashes_match_cpu() {
        let keccak_state = GpuKeccakMerkleState::new().unwrap();
        let rows = make_rows(256, 3);
        let gpu_hashes = gpu_hash_leaves_goldilocks(&rows, &keccak_state).unwrap();
        assert_eq!(gpu_hashes.len(), 256);
        for (i, row) in rows.iter().enumerate() {
            let cpu_hash = cpu_hash_row(row);
            assert_eq!(gpu_hashes[i], cpu_hash, "Leaf hash mismatch at row {i}");
        }
    }

    #[test]
    fn gpu_keccak256_leaf_hashes_single_column() {
        let keccak_state = GpuKeccakMerkleState::new().unwrap();
        let rows: Vec<Vec<FpE>> = (0..128).map(|r| vec![FpE::from(r as u64 + 42)]).collect();
        let gpu_hashes = gpu_hash_leaves_goldilocks(&rows, &keccak_state).unwrap();
        for (i, row) in rows.iter().enumerate() {
            assert_eq!(gpu_hashes[i], cpu_hash_row(row), "Mismatch at row {i}");
        }
    }

    #[test]
    fn gpu_keccak256_leaf_hashes_wide_rows() {
        let keccak_state = GpuKeccakMerkleState::new().unwrap();
        let rows: Vec<Vec<FpE>> = (0..64)
            .map(|r| {
                (0..8)
                    .map(|c| FpE::from((r * 1000 + c * 7 + 13) as u64))
                    .collect()
            })
            .collect();
        let gpu_hashes = gpu_hash_leaves_goldilocks(&rows, &keccak_state).unwrap();
        for (i, row) in rows.iter().enumerate() {
            assert_eq!(gpu_hashes[i], cpu_hash_row(row), "Mismatch at row {i}");
        }
    }

    #[test]
    fn gpu_keccak256_pair_hashes_match_cpu() {
        let keccak_state = GpuKeccakMerkleState::new().unwrap();
        let children: Vec<[u8; 32]> = (0..256)
            .map(|i| {
                let mut hash = [0u8; 32];
                for (j, byte) in hash.iter_mut().enumerate() {
                    *byte = ((i * 31 + j * 7 + 13) % 256) as u8;
                }
                hash
            })
            .collect();
        let gpu_parents = gpu_hash_tree_level(&children, &keccak_state).unwrap();
        assert_eq!(gpu_parents.len(), 128);
        for i in 0..128 {
            let cpu_parent = cpu_hash_pair(&children[2 * i], &children[2 * i + 1]);
            assert_eq!(gpu_parents[i], cpu_parent, "Pair hash mismatch at {i}");
        }
    }

    #[test]
    fn gpu_keccak256_full_tree_matches_cpu_batch_commit() {
        let keccak_state = GpuKeccakMerkleState::new().unwrap();
        let columns = make_columns(3, 256);
        let cpu_rows = columns2rows_bit_reversed(&columns);
        let (cpu_tree, cpu_root) = cpu_batch_commit(&cpu_rows).unwrap();
        let (gpu_tree, gpu_root) = gpu_batch_commit_goldilocks(&columns, &keccak_state).unwrap();
        assert_eq!(gpu_root, cpu_root, "Merkle root mismatch");
        assert_proofs_match(&cpu_tree, &gpu_tree, &[0, 1, 128, 255]);
    }

    #[test]
    fn gpu_keccak256_tree_non_power_of_two_leaves() {
        let keccak_state = GpuKeccakMerkleState::new().unwrap();
        let columns = make_columns(2, 200);
        let cpu_rows = columns2rows_bit_reversed(&columns);
        let (_, cpu_root) = cpu_batch_commit(&cpu_rows).unwrap();
        let leaf_hashes = gpu_hash_leaves_goldilocks(&cpu_rows, &keccak_state).unwrap();
        let (_, gpu_root) = gpu_build_merkle_tree(&leaf_hashes, &keccak_state).unwrap();
        assert_eq!(gpu_root, cpu_root, "Non-power-of-two tree root mismatch");
    }

    #[test]
    fn gpu_transpose_bitrev_matches_cpu() {
        let keccak_state = GpuKeccakMerkleState::new().unwrap();
        let columns = make_columns(3, 256);
        let cpu_rows = columns2rows_bit_reversed(&columns);
        let gpu_rows = gpu_transpose_bitrev(&columns, &keccak_state).unwrap();
        assert_eq!(gpu_rows.len(), cpu_rows.len());
        for (i, (gpu_row, cpu_row)) in gpu_rows.iter().zip(cpu_rows.iter()).enumerate() {
            assert_eq!(
                gpu_row.len(),
                cpu_row.len(),
                "Row {i} column count mismatch"
            );
            for (j, (g, c)) in gpu_row.iter().zip(cpu_row.iter()).enumerate() {
                assert_eq!(g, c, "Mismatch at row {i}, col {j}");
            }
        }
    }

    #[test]
    fn gpu_transpose_bitrev_small() {
        let keccak_state = GpuKeccakMerkleState::new().unwrap();
        let columns: Vec<Vec<FpE>> = vec![
            (0..8).map(|i| FpE::from(i as u64)).collect(),
            (0..8).map(|i| FpE::from(i as u64 + 100)).collect(),
        ];
        let cpu_rows = columns2rows_bit_reversed(&columns);
        let gpu_rows = gpu_transpose_bitrev(&columns, &keccak_state).unwrap();
        assert_eq!(gpu_rows.len(), cpu_rows.len());
        for (i, (gpu_row, cpu_row)) in gpu_rows.iter().zip(cpu_rows.iter()).enumerate() {
            for (j, (g, c)) in gpu_row.iter().zip(cpu_row.iter()).enumerate() {
                assert_eq!(g, c, "Mismatch at row {i}, col {j}");
            }
        }
    }

    #[test]
    fn gpu_transpose_bitrev_buffer_matches_cpu() {
        let keccak_state = GpuKeccakMerkleState::new().unwrap();
        let transpose_state = keccak_state.transpose_state().expect("transpose state");
        let columns = make_columns(3, 256);
        let cpu_rows = columns2rows_bit_reversed(&columns);

        let col_buffers = columns_to_metal_buffers(&columns, transpose_state.state.device());
        let col_buf_refs: Vec<&metal::Buffer> = col_buffers.iter().collect();
        let result_buf =
            gpu_transpose_bitrev_dispatch(&col_buf_refs, 256, false, transpose_state).unwrap();

        let gpu_raw: Vec<u64> = unsafe {
            let ptr = result_buf.contents() as *const u64;
            std::slice::from_raw_parts(ptr, 256 * 3).to_vec()
        };
        for (row_idx, cpu_row) in cpu_rows.iter().enumerate() {
            for (col_idx, cpu_val) in cpu_row.iter().enumerate() {
                let gpu_val = gpu_raw[row_idx * 3 + col_idx];
                let expected = canonical(cpu_val);
                assert_eq!(
                    gpu_val, expected,
                    "Buffer transpose mismatch at row {row_idx}, col {col_idx}"
                );
            }
        }
    }

    #[test]
    fn gpu_transpose_bitrev_paired_buffer_matches_cpu() {
        use lambdaworks_math::fft::cpu::bit_reversing::in_place_bit_reverse_permute;

        let keccak_state = GpuKeccakMerkleState::new().unwrap();
        let transpose_state = keccak_state.transpose_state().expect("transpose state");

        let num_cols = 2;
        let num_rows: usize = 128;
        let columns = make_columns(num_cols, num_rows);

        // CPU reference: transpose + bit-reverse + pair
        let mut cpu_rows: Vec<Vec<FpE>> = (0..num_rows)
            .map(|i| columns.iter().map(|col| col[i]).collect())
            .collect();
        in_place_bit_reverse_permute(&mut cpu_rows);
        let mut merged_rows = Vec::with_capacity(num_rows / 2);
        let mut iter = cpu_rows.into_iter();
        while let (Some(mut chunk0), Some(chunk1)) = (iter.next(), iter.next()) {
            chunk0.extend(chunk1);
            merged_rows.push(chunk0);
        }

        let col_buffers = columns_to_metal_buffers(&columns, transpose_state.state.device());
        let col_buf_refs: Vec<&metal::Buffer> = col_buffers.iter().collect();
        let result_buf =
            gpu_transpose_bitrev_dispatch(&col_buf_refs, num_rows, true, transpose_state).unwrap();

        let num_merged_rows = num_rows / 2;
        let cols_per_merged = 2 * num_cols;
        let gpu_raw: Vec<u64> = unsafe {
            let ptr = result_buf.contents() as *const u64;
            std::slice::from_raw_parts(ptr, num_merged_rows * cols_per_merged).to_vec()
        };
        for (row_idx, cpu_row) in merged_rows.iter().enumerate() {
            for (col_idx, cpu_val) in cpu_row.iter().enumerate() {
                let gpu_val = gpu_raw[row_idx * cols_per_merged + col_idx];
                let expected = canonical(cpu_val);
                assert_eq!(
                    gpu_val, expected,
                    "Paired buffer transpose mismatch at row {row_idx}, col {col_idx}"
                );
            }
        }
    }

    #[test]
    fn gpu_transpose_bitrev_buffer_small() {
        let keccak_state = GpuKeccakMerkleState::new().unwrap();
        let transpose_state = keccak_state.transpose_state().expect("transpose state");

        let columns: Vec<Vec<FpE>> = vec![
            (0..8).map(|i| FpE::from(i as u64)).collect(),
            (0..8).map(|i| FpE::from(i as u64 + 100)).collect(),
        ];
        let cpu_rows = columns2rows_bit_reversed(&columns);

        let col_buffers = columns_to_metal_buffers(&columns, transpose_state.state.device());
        let col_buf_refs: Vec<&metal::Buffer> = col_buffers.iter().collect();
        let result_buf =
            gpu_transpose_bitrev_dispatch(&col_buf_refs, 8, false, transpose_state).unwrap();

        let gpu_raw: Vec<u64> = unsafe {
            let ptr = result_buf.contents() as *const u64;
            std::slice::from_raw_parts(ptr, 8 * 2).to_vec()
        };
        for (row_idx, cpu_row) in cpu_rows.iter().enumerate() {
            for (col_idx, cpu_val) in cpu_row.iter().enumerate() {
                let gpu_val = gpu_raw[row_idx * 2 + col_idx];
                let expected = canonical(cpu_val);
                assert_eq!(
                    gpu_val, expected,
                    "Small buffer transpose mismatch at row {row_idx}, col {col_idx}"
                );
            }
        }
    }

    #[test]
    fn gpu_batch_commit_matches_cpu() {
        let keccak_state = GpuKeccakMerkleState::new().unwrap();
        let columns = make_columns(3, 512);
        let cpu_rows = columns2rows_bit_reversed(&columns);
        let (cpu_tree, cpu_root) = cpu_batch_commit(&cpu_rows).unwrap();
        let (gpu_tree, gpu_root) = gpu_batch_commit_goldilocks(&columns, &keccak_state).unwrap();
        assert_eq!(gpu_root, cpu_root, "Batch commit root mismatch");
        assert_proofs_match(&cpu_tree, &gpu_tree, &[0, 1, 128, 256, 511]);
    }

    #[test]
    fn gpu_batch_commit_paired_matches_cpu() {
        let keccak_state = GpuKeccakMerkleState::new().unwrap();
        use lambdaworks_math::fft::cpu::bit_reversing::in_place_bit_reverse_permute;

        let num_parts = 2;
        let lde_len = 128;
        let lde_evaluations: Vec<Vec<FpE>> = (0..num_parts)
            .map(|p| {
                (0..lde_len)
                    .map(|i| FpE::from((p * lde_len + i + 1) as u64))
                    .collect()
            })
            .collect();

        // CPU path: transpose + bit-reverse + pair + commit
        let mut cpu_rows: Vec<Vec<FpE>> = (0..lde_len)
            .map(|i| lde_evaluations.iter().map(|col| col[i]).collect())
            .collect();
        in_place_bit_reverse_permute(&mut cpu_rows);
        let mut merged_rows = Vec::with_capacity(lde_len / 2);
        let mut iter = cpu_rows.into_iter();
        while let (Some(mut chunk0), Some(chunk1)) = (iter.next(), iter.next()) {
            chunk0.extend(chunk1);
            merged_rows.push(chunk0);
        }
        let (cpu_tree, cpu_root) = cpu_batch_commit(&merged_rows).unwrap();

        let (gpu_tree, gpu_root) =
            gpu_batch_commit_paired_goldilocks(&lde_evaluations, &keccak_state).unwrap();
        assert_eq!(gpu_root, cpu_root, "Paired commit root mismatch");
        assert_proofs_match(&cpu_tree, &gpu_tree, &[0, 1, 32, 63]);
    }

    #[test]
    fn gpu_grinding_matches_cpu() {
        use stark_platinum_prover::grinding;

        let keccak_state = GpuKeccakMerkleState::new().unwrap();

        let seed1 = [
            37, 68, 26, 150, 139, 142, 66, 175, 33, 47, 199, 160, 9, 109, 79, 234, 135, 254, 39,
            11, 225, 219, 206, 108, 224, 165, 25, 72, 189, 96, 218, 95,
        ];

        assert_eq!(gpu_generate_nonce(&seed1, 0, &keccak_state), Some(0));

        let cpu_nonce = grinding::generate_nonce(&seed1, 1).unwrap();
        let gpu_nonce = gpu_generate_nonce(&seed1, 1, &keccak_state).unwrap();
        assert_eq!(gpu_nonce, cpu_nonce, "grinding_factor=1 mismatch");
        assert!(grinding::is_valid_nonce(&seed1, gpu_nonce, 1));

        let cpu_nonce = grinding::generate_nonce(&seed1, 10).unwrap();
        let gpu_nonce = gpu_generate_nonce(&seed1, 10, &keccak_state).unwrap();
        assert_eq!(gpu_nonce, cpu_nonce, "grinding_factor=10 mismatch");
        assert!(grinding::is_valid_nonce(&seed1, gpu_nonce, 10));

        let seed2 = [
            174, 187, 26, 134, 6, 43, 222, 151, 140, 48, 52, 67, 69, 181, 177, 165, 111, 222, 148,
            92, 130, 241, 171, 2, 62, 34, 95, 159, 37, 116, 155, 217,
        ];
        let cpu_nonce = grinding::generate_nonce(&seed2, 1).unwrap();
        let gpu_nonce = gpu_generate_nonce(&seed2, 1, &keccak_state).unwrap();
        assert_eq!(gpu_nonce, cpu_nonce, "seed2 grinding_factor=1 mismatch");
    }
}
