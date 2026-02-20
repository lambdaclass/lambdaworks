use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::IsField;
use lambdaworks_math::traits::AsBytes;
use stark_platinum_prover::config::{BatchedMerkleTree, Commitment};

/// Build a batched Merkle tree from row-major evaluation data.
/// Currently uses CPU (Keccak256). Will be replaced with GPU Poseidon.
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
use lambdaworks_gpu::metal::abstractions::{errors::MetalError, state::DynamicMetalState};
#[cfg(all(target_os = "macos", feature = "metal"))]
use lambdaworks_math::field::fields::u64_goldilocks_field::Goldilocks64Field;
#[cfg(all(target_os = "macos", feature = "metal"))]
use lambdaworks_math::field::traits::IsPrimeField;

#[cfg(all(target_os = "macos", feature = "metal"))]
const KECCAK256_SHADER: &str = include_str!("shaders/keccak256.metal");

#[cfg(all(target_os = "macos", feature = "metal"))]
const TRANSPOSE_BITREV_SHADER: &str = include_str!("shaders/transpose_bitrev.metal");

/// Pre-compiled Metal state for Keccak256 Merkle tree operations.
///
/// Caches compiled pipelines for both leaf hashing and pair hashing kernels.
/// Create once and reuse across the entire prove call.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub struct GpuKeccakMerkleState {
    state: DynamicMetalState,
    hash_leaves_max_threads: u64,
    hash_pairs_max_threads: u64,
    transpose_max_threads: u64,
    grind_max_threads: u64,
    /// Cached transpose+bitrev state for buffer-based operations.
    transpose_bitrev_state: Option<TransposeBitrevState>,
}

#[cfg(all(target_os = "macos", feature = "metal"))]
impl GpuKeccakMerkleState {
    /// Compile the Keccak256 shader and prepare all pipelines.
    pub fn new() -> Result<Self, MetalError> {
        let mut state = DynamicMetalState::new()?;
        state.load_library(KECCAK256_SHADER)?;
        let hash_leaves_max_threads = state.prepare_pipeline("keccak256_hash_leaves")?;
        let hash_pairs_max_threads = state.prepare_pipeline("keccak256_hash_pairs")?;
        let transpose_max_threads = state.prepare_pipeline("transpose_bitrev_goldilocks")?;
        let grind_max_threads = state.prepare_pipeline("keccak256_grind_nonce")?;

        // Also compile the standalone transpose+bitrev shader for buffer-based ops
        let transpose_bitrev_state = TransposeBitrevState::new().ok();

        Ok(Self {
            state,
            hash_leaves_max_threads,
            hash_pairs_max_threads,
            transpose_max_threads,
            grind_max_threads,
            transpose_bitrev_state,
        })
    }

    /// Get a reference to the transpose state (if initialized).
    pub fn transpose_state(&self) -> Option<&TransposeBitrevState> {
        self.transpose_bitrev_state.as_ref()
    }
}

/// Pre-compiled Metal state for the transpose+bit-reverse kernel.
///
/// This shader doesn't need the Goldilocks header — it operates on raw `ulong` values.
/// Contains pipelines for both unpaired and paired transpose variants.
/// Create once and reuse across the entire prove call.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub struct TransposeBitrevState {
    pub(crate) state: DynamicMetalState,
    unpaired_max_threads: u64,
    paired_max_threads: u64,
}

#[cfg(all(target_os = "macos", feature = "metal"))]
impl TransposeBitrevState {
    /// Compile the transpose+bit-reverse shader and prepare pipelines.
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

/// Transpose column GPU buffers to a single row-major buffer with bit-reversed row ordering.
///
/// Takes per-column Metal Buffers (e.g. FFT output) and produces a single flat buffer
/// laid out as `[row_br(0), row_br(1), ...]` where `br(i) = bit_reverse(i, log2(num_rows))`.
///
/// Steps:
/// 1. Concatenate column buffers into a flat column-major buffer using Metal blit copies
/// 2. Dispatch the `transpose_bitrev` compute kernel
/// 3. Return the row-major output buffer
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn gpu_transpose_bitrev_to_buffer(
    column_buffers: &[&metal::Buffer],
    num_rows: usize,
    transpose_state: &TransposeBitrevState,
) -> Result<metal::Buffer, MetalError> {
    use metal::{MTLCommandBufferStatus, MTLResourceOptions, MTLSize};

    let num_cols = column_buffers.len();
    let log_n = num_rows.trailing_zeros();

    // Allocate flat column-major input buffer
    let flat_cols_size = (num_rows * num_cols * std::mem::size_of::<u64>()) as u64;
    let flat_cols = transpose_state
        .state
        .device()
        .new_buffer(flat_cols_size, MTLResourceOptions::StorageModeShared);

    // Concatenate column buffers via blit encoder
    let command_buffer = transpose_state.state.command_queue().new_command_buffer();
    let blit = command_buffer.new_blit_command_encoder();
    for (i, col_buf) in column_buffers.iter().enumerate() {
        blit.copy_from_buffer(
            col_buf,
            0,
            &flat_cols,
            (i * num_rows * std::mem::size_of::<u64>()) as u64,
            col_buf.length(),
        );
    }
    blit.end_encoding();

    // Allocate output buffer (row-major)
    let output_size = flat_cols_size;
    let output_buf = transpose_state
        .state
        .device()
        .new_buffer(output_size, MTLResourceOptions::StorageModeShared);

    // Parameter buffers
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

    // Dispatch transpose_bitrev kernel
    let pipeline = transpose_state
        .state
        .get_pipeline_ref("transpose_bitrev")
        .ok_or_else(|| MetalError::FunctionError("transpose_bitrev".to_string()))?;

    let threads_per_group = transpose_state.unpaired_max_threads.min(256);
    let thread_groups = (num_rows as u64).div_ceil(threads_per_group);

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

    command_buffer.commit();
    command_buffer.wait_until_completed();

    if command_buffer.status() == MTLCommandBufferStatus::Error {
        return Err(MetalError::ExecutionError(
            "GPU transpose_bitrev command error".to_string(),
        ));
    }

    Ok(output_buf)
}

/// Transpose column GPU buffers to a paired-row-major buffer with bit-reversed row ordering.
///
/// Like [`gpu_transpose_bitrev_to_buffer`] but merges consecutive bit-reversed rows:
///   `merged_row[i] = row[br(2*i)] ++ row[br(2*i+1)]`
///
/// Output has `num_rows / 2` merged rows, each with `2 * num_cols` elements.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn gpu_transpose_bitrev_paired_to_buffer(
    column_buffers: &[&metal::Buffer],
    num_rows: usize,
    transpose_state: &TransposeBitrevState,
) -> Result<metal::Buffer, MetalError> {
    use metal::{MTLCommandBufferStatus, MTLResourceOptions, MTLSize};

    let num_cols = column_buffers.len();
    let log_n = num_rows.trailing_zeros();

    // Allocate flat column-major input buffer
    let flat_cols_size = (num_rows * num_cols * std::mem::size_of::<u64>()) as u64;
    let flat_cols = transpose_state
        .state
        .device()
        .new_buffer(flat_cols_size, MTLResourceOptions::StorageModeShared);

    // Concatenate column buffers via blit encoder
    let command_buffer = transpose_state.state.command_queue().new_command_buffer();
    let blit = command_buffer.new_blit_command_encoder();
    for (i, col_buf) in column_buffers.iter().enumerate() {
        blit.copy_from_buffer(
            col_buf,
            0,
            &flat_cols,
            (i * num_rows * std::mem::size_of::<u64>()) as u64,
            col_buf.length(),
        );
    }
    blit.end_encoding();

    // Allocate output buffer: (num_rows / 2) merged rows × (2 * num_cols) elements
    let num_merged_rows = num_rows / 2;
    let cols_per_merged_row = 2 * num_cols;
    let output_size = (num_merged_rows * cols_per_merged_row * std::mem::size_of::<u64>()) as u64;
    let output_buf = transpose_state
        .state
        .device()
        .new_buffer(output_size, MTLResourceOptions::StorageModeShared);

    // Parameter buffers
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

    // Dispatch transpose_bitrev_paired kernel
    let pipeline = transpose_state
        .state
        .get_pipeline_ref("transpose_bitrev_paired")
        .ok_or_else(|| MetalError::FunctionError("transpose_bitrev_paired".to_string()))?;

    let threads_per_group = transpose_state.paired_max_threads.min(256);
    let thread_groups = (num_merged_rows as u64).div_ceil(threads_per_group);

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

    command_buffer.commit();
    command_buffer.wait_until_completed();

    if command_buffer.status() == MTLCommandBufferStatus::Error {
        return Err(MetalError::ExecutionError(
            "GPU transpose_bitrev_paired command error".to_string(),
        ));
    }

    Ok(output_buf)
}

/// Hash row-major Goldilocks field element data into 32-byte leaf digests on GPU.
///
/// `rows` is a slice of rows, where each row is a `Vec<FieldElement<Goldilocks64Field>>`.
/// Returns one 32-byte Keccak256 hash per row.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn gpu_hash_leaves_goldilocks(
    rows: &[Vec<FieldElement<Goldilocks64Field>>],
    keccak_state: &GpuKeccakMerkleState,
) -> Result<Vec<[u8; 32]>, MetalError> {
    if rows.is_empty() {
        return Ok(Vec::new());
    }

    let num_rows = rows.len();
    let num_cols = rows[0].len();

    // Flatten rows into a flat Vec<u64> (canonical form, row-major)
    let flat_data: Vec<u64> = rows
        .iter()
        .flat_map(|row| {
            row.iter()
                .map(|fe| Goldilocks64Field::canonical(fe.value()))
        })
        .collect();

    gpu_hash_leaves_flat(&flat_data, num_rows, num_cols, keccak_state)
}

/// Hash pre-flattened row-major u64 data into 32-byte leaf digests on GPU.
///
/// `flat_data` must be laid out as row-major: `[row0_col0, row0_col1, ..., row1_col0, ...]`
/// with `num_rows * num_cols` elements total, each in canonical Goldilocks form.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn gpu_hash_leaves_flat(
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
        "keccak256_hash_leaves",
        &[&buf_data, &buf_output, &buf_num_cols, &buf_num_rows],
        num_rows as u64,
        keccak_state.hash_leaves_max_threads,
    )?;

    // Read back hashes
    let raw_output: Vec<u8> = unsafe { keccak_state.state.read_buffer(&buf_output, num_rows * 32) };

    let hashes: Vec<[u8; 32]> = raw_output
        .chunks_exact(32)
        .map(|chunk| {
            let mut hash = [0u8; 32];
            hash.copy_from_slice(chunk);
            hash
        })
        .collect();

    Ok(hashes)
}

/// Hash pairs of 32-byte child nodes into parent nodes on GPU.
///
/// Returns `children.len() / 2` parent hashes.
/// Falls back to CPU for fewer than 64 pairs (GPU overhead not worth it).
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn gpu_hash_tree_level(
    children: &[[u8; 32]],
    keccak_state: &GpuKeccakMerkleState,
) -> Result<Vec<[u8; 32]>, MetalError> {
    let num_pairs = children.len() / 2;
    if num_pairs == 0 {
        return Ok(Vec::new());
    }

    // CPU fallback for small levels
    if num_pairs < 64 {
        use sha3::{Digest, Keccak256};
        let parents: Vec<[u8; 32]> = (0..num_pairs)
            .map(|i| {
                let mut hasher = Keccak256::new();
                hasher.update(children[2 * i]);
                hasher.update(children[2 * i + 1]);
                let mut result = [0u8; 32];
                result.copy_from_slice(&hasher.finalize());
                result
            })
            .collect();
        return Ok(parents);
    }

    // Flatten children into a flat byte buffer
    let flat_children: Vec<u8> = children.iter().flat_map(|h| h.iter().copied()).collect();

    let buf_children = keccak_state.state.alloc_buffer_with_data(&flat_children)?;
    let buf_parents = keccak_state.state.alloc_buffer(num_pairs * 32)?;
    let num_pairs_u32 = num_pairs as u32;
    let buf_num_pairs = keccak_state
        .state
        .alloc_buffer_with_data(std::slice::from_ref(&num_pairs_u32))?;

    keccak_state.state.execute_compute(
        "keccak256_hash_pairs",
        &[&buf_children, &buf_parents, &buf_num_pairs],
        num_pairs as u64,
        keccak_state.hash_pairs_max_threads,
    )?;

    let raw_output: Vec<u8> =
        unsafe { keccak_state.state.read_buffer(&buf_parents, num_pairs * 32) };

    let parents: Vec<[u8; 32]> = raw_output
        .chunks_exact(32)
        .map(|chunk| {
            let mut hash = [0u8; 32];
            hash.copy_from_slice(chunk);
            hash
        })
        .collect();

    Ok(parents)
}

/// Build a complete Merkle tree from leaf hashes, returning the flat node array and root.
///
/// The node layout matches `MerkleTree::build()`:
/// `[inner_nodes | leaves]` where `nodes[0]` = root.
///
/// Leaves are padded to the next power of two by repeating the last hash.
///
/// Uses a single GPU command buffer with all tree levels encoded as sequential
/// compute dispatches, avoiding per-level CPU-GPU round-trips.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn gpu_build_merkle_tree(
    leaf_hashes: &[[u8; 32]],
    keccak_state: &GpuKeccakMerkleState,
) -> Result<(Vec<[u8; 32]>, [u8; 32]), MetalError> {
    use metal::{MTLCommandBufferStatus, MTLResourceOptions, MTLSize};

    if leaf_hashes.is_empty() {
        return Err(MetalError::ExecutionError("Empty leaf hashes".to_string()));
    }

    // Pad to power of two
    let mut leaves: Vec<[u8; 32]> = leaf_hashes.to_vec();
    while !leaves.len().is_power_of_two() {
        leaves.push(*leaves.last().unwrap());
    }

    let leaves_len = leaves.len();
    let total_nodes = 2 * leaves_len - 1;

    // Allocate a single GPU buffer for the entire tree (all nodes × 32 bytes)
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

    // Get the pipeline for pair hashing
    let pipeline = keccak_state
        .state
        .get_pipeline_ref("keccak256_hash_pairs")
        .ok_or_else(|| MetalError::FunctionError("keccak256_hash_pairs".to_string()))?;

    let threads_per_group = keccak_state.hash_pairs_max_threads.min(256);

    // Encode all tree levels in a single command buffer
    let command_buffer = keccak_state.state.command_queue().new_command_buffer();

    let mut level_begin = leaves_len - 1;
    let mut level_end = 2 * level_begin; // inclusive

    while level_begin != level_end {
        let new_level_begin = level_begin / 2;
        let num_pairs = (level_end - level_begin).div_ceil(2);

        if num_pairs < 64 {
            // For very small levels, we'll handle on CPU after the GPU work completes.
            // But first, commit what we have so far.
            break;
        }

        // Each encoder reads children from [level_begin..=level_end] and writes
        // parents to [new_level_begin..level_begin-1] within the same tree_buf.
        let children_byte_offset = (level_begin * 32) as u64;
        let parents_byte_offset = (new_level_begin * 32) as u64;

        // Allocate a tiny buffer for num_pairs parameter
        let num_pairs_u32 = num_pairs as u32;
        let param_buf = keccak_state.state.device().new_buffer_with_data(
            &num_pairs_u32 as *const u32 as *const _,
            std::mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(&tree_buf), children_byte_offset);
        encoder.set_buffer(1, Some(&tree_buf), parents_byte_offset);
        encoder.set_buffer(2, Some(&param_buf), 0);

        let thread_groups_count = (num_pairs as u64).div_ceil(threads_per_group);
        encoder.dispatch_thread_groups(
            MTLSize::new(thread_groups_count, 1, 1),
            MTLSize::new(threads_per_group, 1, 1),
        );
        encoder.end_encoding();

        level_end = level_begin - 1;
        level_begin = new_level_begin;
    }

    command_buffer.commit();
    command_buffer.wait_until_completed();

    if command_buffer.status() == MTLCommandBufferStatus::Error {
        return Err(MetalError::ExecutionError(
            "GPU tree build command buffer error".to_string(),
        ));
    }

    // Read back the entire tree
    let mut nodes = vec![[0u8; 32]; total_nodes];
    unsafe {
        let ptr = tree_buf.contents() as *const u8;
        std::ptr::copy_nonoverlapping(ptr, nodes.as_mut_ptr() as *mut u8, total_nodes * 32);
    }

    // Finish any remaining small levels on CPU
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

/// Transpose column-major LDE data to row-major bit-reversed layout on GPU.
///
/// Takes column-major data (as `&[Vec<FieldElement<Goldilocks64Field>>]`) and
/// returns row-major bit-reversed data as `Vec<Vec<FieldElement<Goldilocks64Field>>>`.
///
/// This replaces the CPU `columns2rows_bit_reversed()` function.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn gpu_transpose_bitrev(
    columns: &[Vec<FieldElement<Goldilocks64Field>>],
    keccak_state: &GpuKeccakMerkleState,
) -> Result<Vec<Vec<FieldElement<Goldilocks64Field>>>, MetalError> {
    if columns.is_empty() {
        return Ok(Vec::new());
    }

    let num_cols = columns.len();
    let num_rows = columns[0].len();

    // Flatten column-major: col0[0..M], col1[0..M], ...
    let flat_cols: Vec<u64> = columns
        .iter()
        .flat_map(|col| {
            col.iter()
                .map(|fe| Goldilocks64Field::canonical(fe.value()))
        })
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

    // Read back row-major data
    let raw_rows: Vec<u64> = unsafe {
        keccak_state
            .state
            .read_buffer(&buf_rows, num_rows * num_cols)
    };

    // Convert back to Vec<Vec<FieldElement>>
    let rows: Vec<Vec<FieldElement<Goldilocks64Field>>> = raw_rows
        .chunks_exact(num_cols)
        .map(|row| row.iter().map(|&v| FieldElement::from(v)).collect())
        .collect();

    Ok(rows)
}

/// Hash leaves and build Merkle tree in a single GPU command buffer.
///
/// This eliminates the CPU readback between leaf hashing and tree building:
/// 1. Uploads flat u64 data to GPU
/// 2. Hashes leaves directly into the leaf positions of the tree buffer
/// 3. Builds all tree levels in the same command buffer
/// 4. Reads back the full tree once
///
/// Returns the flat node array `[inner_nodes | leaves]` and root hash.
#[cfg(all(target_os = "macos", feature = "metal"))]
fn gpu_hash_and_build_tree(
    flat_data: &[u64],
    num_rows: usize,
    num_cols: usize,
    keccak_state: &GpuKeccakMerkleState,
) -> Result<(Vec<[u8; 32]>, [u8; 32]), MetalError> {
    use metal::{MTLCommandBufferStatus, MTLResourceOptions, MTLSize};

    if num_rows == 0 {
        return Err(MetalError::ExecutionError("Empty data".to_string()));
    }

    // Pad leaf count to power of two
    let leaves_len = num_rows.next_power_of_two();
    let total_nodes = 2 * leaves_len - 1;

    // Allocate GPU buffers
    let buf_data = keccak_state.state.alloc_buffer_with_data(flat_data)?;
    let tree_buf = keccak_state.state.device().new_buffer(
        (total_nodes * 32) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Parameter buffers
    let num_cols_u32 = num_cols as u32;
    let num_rows_u32 = num_rows as u32;
    let buf_num_cols = keccak_state
        .state
        .alloc_buffer_with_data(std::slice::from_ref(&num_cols_u32))?;
    let buf_num_rows = keccak_state
        .state
        .alloc_buffer_with_data(std::slice::from_ref(&num_rows_u32))?;

    let leaf_hash_pipeline = keccak_state
        .state
        .get_pipeline_ref("keccak256_hash_leaves")
        .ok_or_else(|| MetalError::FunctionError("keccak256_hash_leaves".to_string()))?;
    let pair_hash_pipeline = keccak_state
        .state
        .get_pipeline_ref("keccak256_hash_pairs")
        .ok_or_else(|| MetalError::FunctionError("keccak256_hash_pairs".to_string()))?;

    let command_buffer = keccak_state.state.command_queue().new_command_buffer();

    // Dispatch 1: Hash leaves directly into tree buffer at leaf positions
    {
        let leaf_byte_offset = ((leaves_len - 1) * 32) as u64;
        let threads_per_group = keccak_state.hash_leaves_max_threads.min(256);
        let thread_groups = (num_rows as u64).div_ceil(threads_per_group);

        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(leaf_hash_pipeline);
        encoder.set_buffer(0, Some(&buf_data), 0);
        encoder.set_buffer(1, Some(&tree_buf), leaf_byte_offset);
        encoder.set_buffer(2, Some(&buf_num_cols), 0);
        encoder.set_buffer(3, Some(&buf_num_rows), 0);
        encoder.dispatch_thread_groups(
            MTLSize::new(thread_groups, 1, 1),
            MTLSize::new(threads_per_group, 1, 1),
        );
        encoder.end_encoding();
    }

    // If num_rows < leaves_len, pad by duplicating the last leaf hash.
    // We handle this on CPU after readback (only needed for non-power-of-two).

    // Dispatch tree levels (bottom-up)
    let threads_per_group = keccak_state.hash_pairs_max_threads.min(256);
    let mut level_begin = leaves_len - 1;
    let mut level_end = 2 * level_begin;
    let mut cpu_level_begin = level_begin;
    let mut cpu_level_end = level_end;

    while level_begin != level_end {
        let new_level_begin = level_begin / 2;
        let num_pairs = (level_end - level_begin).div_ceil(2);

        if num_pairs < 64 {
            // Small levels will be handled on CPU
            cpu_level_begin = level_begin;
            cpu_level_end = level_end;
            break;
        }

        let children_byte_offset = (level_begin * 32) as u64;
        let parents_byte_offset = (new_level_begin * 32) as u64;
        let num_pairs_u32 = num_pairs as u32;
        let param_buf = keccak_state.state.device().new_buffer_with_data(
            &num_pairs_u32 as *const u32 as *const _,
            std::mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pair_hash_pipeline);
        encoder.set_buffer(0, Some(&tree_buf), children_byte_offset);
        encoder.set_buffer(1, Some(&tree_buf), parents_byte_offset);
        encoder.set_buffer(2, Some(&param_buf), 0);
        let thread_groups_count = (num_pairs as u64).div_ceil(threads_per_group);
        encoder.dispatch_thread_groups(
            MTLSize::new(thread_groups_count, 1, 1),
            MTLSize::new(threads_per_group, 1, 1),
        );
        encoder.end_encoding();

        cpu_level_begin = new_level_begin;
        cpu_level_end = level_begin - 1;
        level_end = level_begin - 1;
        level_begin = new_level_begin;
    }

    // Submit all GPU work at once
    command_buffer.commit();
    command_buffer.wait_until_completed();

    if command_buffer.status() == MTLCommandBufferStatus::Error {
        return Err(MetalError::ExecutionError(
            "GPU hash+tree command buffer error".to_string(),
        ));
    }

    // Read back entire tree
    let mut nodes = vec![[0u8; 32]; total_nodes];
    unsafe {
        let ptr = tree_buf.contents() as *const u8;
        std::ptr::copy_nonoverlapping(ptr, nodes.as_mut_ptr() as *mut u8, total_nodes * 32);
    }

    // Handle padding: if num_rows < leaves_len, duplicate last real hash
    if num_rows < leaves_len {
        let last_real = leaves_len - 1 + num_rows - 1;
        let pad_hash = nodes[last_real];
        for node in nodes
            .iter_mut()
            .take(leaves_len - 1 + leaves_len)
            .skip(last_real + 1)
        {
            *node = pad_hash;
        }
    }

    // Finish remaining small levels on CPU
    let mut lb = cpu_level_begin;
    let mut le = cpu_level_end;
    while lb != le {
        let new_lb = lb / 2;
        let children_slice = &nodes[lb..=le];
        let parents = gpu_hash_tree_level(children_slice, keccak_state)?;
        for (i, parent) in parents.iter().enumerate() {
            nodes[new_lb + i] = *parent;
        }
        le = lb - 1;
        lb = new_lb;
    }

    let root = nodes[0];
    Ok((nodes, root))
}

/// Hash leaves and build Merkle tree from a GPU buffer (zero CPU readback for data).
///
/// Like [`gpu_hash_and_build_tree`] but takes a Metal Buffer containing pre-computed
/// flat u64 data instead of a CPU slice. This avoids an extra CPU→GPU upload when
/// the data was already produced by a GPU operation (e.g., FFT).
///
/// The buffer must contain `num_rows * num_cols` u64 values in row-major order,
/// each in canonical Goldilocks form.
#[cfg(all(target_os = "macos", feature = "metal"))]
fn gpu_hash_and_build_tree_from_buffer(
    buf_data: &metal::Buffer,
    num_rows: usize,
    num_cols: usize,
    keccak_state: &GpuKeccakMerkleState,
) -> Result<(Vec<[u8; 32]>, [u8; 32]), MetalError> {
    use metal::{MTLCommandBufferStatus, MTLResourceOptions, MTLSize};

    if num_rows == 0 {
        return Err(MetalError::ExecutionError("Empty data".to_string()));
    }

    let leaves_len = num_rows.next_power_of_two();
    let total_nodes = 2 * leaves_len - 1;

    let tree_buf = keccak_state.state.device().new_buffer(
        (total_nodes * 32) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let num_cols_u32 = num_cols as u32;
    let num_rows_u32 = num_rows as u32;
    let buf_num_cols = keccak_state
        .state
        .alloc_buffer_with_data(std::slice::from_ref(&num_cols_u32))?;
    let buf_num_rows = keccak_state
        .state
        .alloc_buffer_with_data(std::slice::from_ref(&num_rows_u32))?;

    let leaf_hash_pipeline = keccak_state
        .state
        .get_pipeline_ref("keccak256_hash_leaves")
        .ok_or_else(|| MetalError::FunctionError("keccak256_hash_leaves".to_string()))?;
    let pair_hash_pipeline = keccak_state
        .state
        .get_pipeline_ref("keccak256_hash_pairs")
        .ok_or_else(|| MetalError::FunctionError("keccak256_hash_pairs".to_string()))?;

    let command_buffer = keccak_state.state.command_queue().new_command_buffer();

    // Dispatch 1: Hash leaves directly into tree buffer at leaf positions
    {
        let leaf_byte_offset = ((leaves_len - 1) * 32) as u64;
        let threads_per_group = keccak_state.hash_leaves_max_threads.min(256);
        let thread_groups = (num_rows as u64).div_ceil(threads_per_group);

        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(leaf_hash_pipeline);
        encoder.set_buffer(0, Some(buf_data), 0);
        encoder.set_buffer(1, Some(&tree_buf), leaf_byte_offset);
        encoder.set_buffer(2, Some(&buf_num_cols), 0);
        encoder.set_buffer(3, Some(&buf_num_rows), 0);
        encoder.dispatch_thread_groups(
            MTLSize::new(thread_groups, 1, 1),
            MTLSize::new(threads_per_group, 1, 1),
        );
        encoder.end_encoding();
    }

    // Dispatch tree levels (bottom-up)
    let threads_per_group = keccak_state.hash_pairs_max_threads.min(256);
    let mut level_begin = leaves_len - 1;
    let mut level_end = 2 * level_begin;

    while level_begin != level_end {
        let new_level_begin = level_begin / 2;
        let num_pairs = (level_end - level_begin).div_ceil(2);

        let children_byte_offset = (level_begin * 32) as u64;
        let parents_byte_offset = (new_level_begin * 32) as u64;
        let num_pairs_u32 = num_pairs as u32;
        let param_buf = keccak_state.state.device().new_buffer_with_data(
            &num_pairs_u32 as *const u32 as *const _,
            std::mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pair_hash_pipeline);
        encoder.set_buffer(0, Some(&tree_buf), children_byte_offset);
        encoder.set_buffer(1, Some(&tree_buf), parents_byte_offset);
        encoder.set_buffer(2, Some(&param_buf), 0);
        let thread_groups_count = (num_pairs as u64).div_ceil(threads_per_group);
        encoder.dispatch_thread_groups(
            MTLSize::new(thread_groups_count, 1, 1),
            MTLSize::new(threads_per_group, 1, 1),
        );
        encoder.end_encoding();

        level_end = level_begin - 1;
        level_begin = new_level_begin;
    }

    command_buffer.commit();
    command_buffer.wait_until_completed();

    if command_buffer.status() == MTLCommandBufferStatus::Error {
        return Err(MetalError::ExecutionError(
            "GPU hash+tree from buffer command error".to_string(),
        ));
    }

    let mut nodes = vec![[0u8; 32]; total_nodes];
    unsafe {
        let ptr = tree_buf.contents() as *const u8;
        std::ptr::copy_nonoverlapping(ptr, nodes.as_mut_ptr() as *mut u8, total_nodes * 32);
    }

    if num_rows < leaves_len {
        let last_real = leaves_len - 1 + num_rows - 1;
        let pad_hash = nodes[last_real];
        for node in nodes
            .iter_mut()
            .take(leaves_len - 1 + leaves_len)
            .skip(last_real + 1)
        {
            *node = pad_hash;
        }
    }

    let root = nodes[0];
    Ok((nodes, root))
}

/// GPU Merkle commit from column GPU buffers (FFT output).
///
/// Takes per-column Metal Buffers from `fft_to_buffer` (natural-order u64 layout)
/// and builds a Merkle tree by transposing + bit-reversing row indices on GPU,
/// then hashing on GPU.
///
/// Uses a single Metal command buffer for the entire pipeline (blit + transpose +
/// hash leaves + hash tree levels) with only ONE `wait_until_completed()` call,
/// eliminating the sync barrier between transpose and hashing.
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
        false, // unpaired
        keccak_state,
        transpose_state,
    )
    .ok()?;

    let tree = BatchedMerkleTree::<Goldilocks64Field>::from_nodes(nodes)?;
    Some((tree, root))
}

/// GPU Merkle commit from column GPU buffers with paired-row layout (composition poly).
///
/// Like [`gpu_batch_commit_from_column_buffers`] but merges consecutive bit-reversed
/// rows into paired leaves, matching the composition polynomial commit layout.
/// Uses a single Metal command buffer for the entire pipeline.
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
        true, // paired
        keccak_state,
        transpose_state,
    )
    .ok()?;

    let tree = BatchedMerkleTree::<Goldilocks64Field>::from_nodes(nodes)?;
    Some((tree, root))
}

/// Merged single-command-buffer pipeline: blit + transpose + hash leaves + build tree.
///
/// Combines the work of `gpu_transpose_bitrev[_paired]_to_buffer` and
/// `gpu_hash_and_build_tree_from_buffer` into ONE Metal command buffer with
/// a single `wait_until_completed()` call. This eliminates the CPU-GPU sync
/// barrier that previously existed between transpose and hashing.
///
/// When `paired` is true, uses the `transpose_bitrev_paired` kernel and adjusts
/// the leaf layout accordingly (num_rows/2 merged rows, 2*num_cols per row).
///
/// All tree levels are built on GPU, including small ones (no CPU fallback).
#[cfg(all(target_os = "macos", feature = "metal"))]
fn gpu_transpose_hash_and_build_tree(
    column_buffers: &[&metal::Buffer],
    num_rows: usize,
    num_cols: usize,
    paired: bool,
    keccak_state: &GpuKeccakMerkleState,
    transpose_state: &TransposeBitrevState,
) -> Result<(Vec<[u8; 32]>, [u8; 32]), MetalError> {
    use metal::{MTLCommandBufferStatus, MTLResourceOptions, MTLSize};

    let log_n = num_rows.trailing_zeros();

    // Compute effective leaf dimensions
    let (leaf_rows, leaf_cols) = if paired {
        (num_rows / 2, 2 * num_cols)
    } else {
        (num_rows, num_cols)
    };
    let leaves_len = leaf_rows.next_power_of_two();
    let total_nodes = 2 * leaves_len - 1;

    // Allocate all buffers upfront
    let flat_cols_size = (num_rows * num_cols * std::mem::size_of::<u64>()) as u64;
    let flat_cols = keccak_state.state.device().new_buffer(
        flat_cols_size,
        MTLResourceOptions::StorageModeShared,
    );
    let output_size = if paired {
        ((num_rows / 2) * 2 * num_cols * std::mem::size_of::<u64>()) as u64
    } else {
        flat_cols_size
    };
    let transpose_out = keccak_state.state.device().new_buffer(
        output_size,
        MTLResourceOptions::StorageModeShared,
    );
    let tree_buf = keccak_state.state.device().new_buffer(
        (total_nodes * 32) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Parameter buffers
    let num_cols_u32 = num_cols as u32;
    let num_rows_u32 = num_rows as u32;
    let leaf_cols_u32 = leaf_cols as u32;
    let leaf_rows_u32 = leaf_rows as u32;
    let buf_num_cols = keccak_state
        .state
        .alloc_buffer_with_data(std::slice::from_ref(&num_cols_u32))?;
    let buf_num_rows = keccak_state
        .state
        .alloc_buffer_with_data(std::slice::from_ref(&num_rows_u32))?;
    let buf_log_n = keccak_state
        .state
        .alloc_buffer_with_data(std::slice::from_ref(&log_n))?;
    let buf_leaf_cols = keccak_state
        .state
        .alloc_buffer_with_data(std::slice::from_ref(&leaf_cols_u32))?;
    let buf_leaf_rows = keccak_state
        .state
        .alloc_buffer_with_data(std::slice::from_ref(&leaf_rows_u32))?;

    // Get pipeline references
    let transpose_kernel_name = if paired {
        "transpose_bitrev_paired"
    } else {
        "transpose_bitrev"
    };
    let transpose_pipeline = transpose_state
        .state
        .get_pipeline_ref(transpose_kernel_name)
        .ok_or_else(|| MetalError::FunctionError(transpose_kernel_name.to_string()))?;
    let leaf_hash_pipeline = keccak_state
        .state
        .get_pipeline_ref("keccak256_hash_leaves")
        .ok_or_else(|| MetalError::FunctionError("keccak256_hash_leaves".to_string()))?;
    let pair_hash_pipeline = keccak_state
        .state
        .get_pipeline_ref("keccak256_hash_pairs")
        .ok_or_else(|| MetalError::FunctionError("keccak256_hash_pairs".to_string()))?;

    // === ONE command buffer for everything ===
    let command_buffer = keccak_state.state.command_queue().new_command_buffer();

    // Phase 1: Blit-copy column buffers into flat column-major buffer
    let blit = command_buffer.new_blit_command_encoder();
    for (i, col_buf) in column_buffers.iter().enumerate() {
        blit.copy_from_buffer(
            col_buf,
            0,
            &flat_cols,
            (i * num_rows * std::mem::size_of::<u64>()) as u64,
            col_buf.length(),
        );
    }
    blit.end_encoding();

    // Phase 2: Transpose + bit-reverse compute kernel
    {
        let transpose_threads = if paired {
            transpose_state.paired_max_threads
        } else {
            transpose_state.unpaired_max_threads
        };
        let threads_per_group = transpose_threads.min(256);
        let dispatch_count = if paired {
            num_rows as u64 / 2
        } else {
            num_rows as u64
        };
        let thread_groups = dispatch_count.div_ceil(threads_per_group);

        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(transpose_pipeline);
        encoder.set_buffer(0, Some(&flat_cols), 0);
        encoder.set_buffer(1, Some(&transpose_out), 0);
        encoder.set_buffer(2, Some(&buf_num_cols), 0);
        encoder.set_buffer(3, Some(&buf_num_rows), 0);
        encoder.set_buffer(4, Some(&buf_log_n), 0);
        encoder.dispatch_thread_groups(
            MTLSize::new(thread_groups, 1, 1),
            MTLSize::new(threads_per_group, 1, 1),
        );
        encoder.end_encoding();
    }

    // Phase 3: Hash leaves
    {
        let leaf_byte_offset = ((leaves_len - 1) * 32) as u64;
        let threads_per_group = keccak_state.hash_leaves_max_threads.min(256);
        let thread_groups = (leaf_rows as u64).div_ceil(threads_per_group);

        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(leaf_hash_pipeline);
        encoder.set_buffer(0, Some(&transpose_out), 0);
        encoder.set_buffer(1, Some(&tree_buf), leaf_byte_offset);
        encoder.set_buffer(2, Some(&buf_leaf_cols), 0);
        encoder.set_buffer(3, Some(&buf_leaf_rows), 0);
        encoder.dispatch_thread_groups(
            MTLSize::new(thread_groups, 1, 1),
            MTLSize::new(threads_per_group, 1, 1),
        );
        encoder.end_encoding();
    }

    // Phase 4: Build tree levels (ALL on GPU, no CPU fallback)
    let threads_per_group = keccak_state.hash_pairs_max_threads.min(256);
    let mut level_begin = leaves_len - 1;
    let mut level_end = 2 * level_begin;

    while level_begin != level_end {
        let new_level_begin = level_begin / 2;
        let num_pairs = (level_end - level_begin).div_ceil(2);

        let children_byte_offset = (level_begin * 32) as u64;
        let parents_byte_offset = (new_level_begin * 32) as u64;
        let num_pairs_u32 = num_pairs as u32;
        let param_buf = keccak_state.state.device().new_buffer_with_data(
            &num_pairs_u32 as *const u32 as *const _,
            std::mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pair_hash_pipeline);
        encoder.set_buffer(0, Some(&tree_buf), children_byte_offset);
        encoder.set_buffer(1, Some(&tree_buf), parents_byte_offset);
        encoder.set_buffer(2, Some(&param_buf), 0);
        let thread_groups_count = (num_pairs as u64).div_ceil(threads_per_group);
        encoder.dispatch_thread_groups(
            MTLSize::new(thread_groups_count, 1, 1),
            MTLSize::new(threads_per_group, 1, 1),
        );
        encoder.end_encoding();

        level_end = level_begin - 1;
        level_begin = new_level_begin;
    }

    // === ONE sync point ===
    command_buffer.commit();
    command_buffer.wait_until_completed();

    if command_buffer.status() == MTLCommandBufferStatus::Error {
        return Err(MetalError::ExecutionError(
            "GPU transpose+hash+tree command error".to_string(),
        ));
    }

    // Read tree from UMA-shared buffer
    let mut nodes = vec![[0u8; 32]; total_nodes];
    unsafe {
        let ptr = tree_buf.contents() as *const u8;
        std::ptr::copy_nonoverlapping(ptr, nodes.as_mut_ptr() as *mut u8, total_nodes * 32);
    }

    // Pad leaf hashes if num_rows is not a power of two
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

    let root = nodes[0];
    Ok((nodes, root))
}

/// GPU FRI-layer Merkle commit from a GPU buffer.
///
/// Takes a Metal Buffer containing bit-reversed FFT evaluations (u64 layout)
/// and builds a FRI-compatible paired Merkle tree without CPU readback.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn gpu_fri_layer_commit_from_buffer(
    eval_buffer: &metal::Buffer,
    eval_len: usize,
    keccak_state: &GpuKeccakMerkleState,
) -> Result<(BatchedMerkleTree<Goldilocks64Field>, Commitment), MetalError> {
    let num_leaves = eval_len / 2;
    let num_cols = 2;

    // The buffer already contains u64 values in the right order for paired leaves:
    // leaf[i] = [eval[2*i], eval[2*i+1]]
    let (nodes, root) =
        gpu_hash_and_build_tree_from_buffer(eval_buffer, num_leaves, num_cols, keccak_state)?;
    let tree = BatchedMerkleTree::<Goldilocks64Field>::from_nodes(nodes)
        .ok_or_else(|| MetalError::ExecutionError("Failed to build FRI Merkle tree".into()))?;
    Ok((tree, root))
}

/// Build a FRI-compatible Merkle tree from bit-reversed evaluations using GPU.
///
/// The evaluations are already bit-reversed. Groups consecutive pairs as leaves
/// (matching the CPU `new_fri_layer` layout) and builds the Merkle tree on GPU.
///
/// Returns `(MerkleTree, root)` compatible with `FriLayer` and `query_phase`.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn gpu_fri_layer_commit(
    evaluation: &[FieldElement<Goldilocks64Field>],
    keccak_state: &GpuKeccakMerkleState,
) -> Result<(BatchedMerkleTree<Goldilocks64Field>, Commitment), MetalError> {
    let num_leaves = evaluation.len() / 2;
    let num_cols = 2;

    // Convert evaluations to flat u64 — natural order maps to paired rows:
    // leaf[i] = [eval[2*i], eval[2*i+1]]
    let flat_data: Vec<u64> = evaluation
        .iter()
        .map(|fe| Goldilocks64Field::canonical(fe.value()))
        .collect();

    let (nodes, root) = gpu_hash_and_build_tree(&flat_data, num_leaves, num_cols, keccak_state)?;
    let tree = BatchedMerkleTree::<Goldilocks64Field>::from_nodes(nodes)
        .ok_or_else(|| MetalError::ExecutionError("Failed to build FRI Merkle tree".into()))?;
    Ok((tree, root))
}

/// Top-level GPU Merkle commit for Goldilocks field.
///
/// Replaces the CPU path of `columns2rows_bit_reversed() + cpu_batch_commit()`.
/// Takes column-major LDE evaluations, returns a MerkleTree + root commitment.
///
/// Computes the bit-reversed row-major flat data directly without intermediate
/// Vec-of-Vec allocations, then hashes and builds tree on GPU in a single
/// command buffer (zero intermediate readback).
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn gpu_batch_commit_goldilocks(
    lde_columns: &[Vec<FieldElement<Goldilocks64Field>>],
    keccak_state: &GpuKeccakMerkleState,
) -> Option<(BatchedMerkleTree<Goldilocks64Field>, Commitment)> {
    let num_cols = lde_columns.len();
    let num_rows = lde_columns[0].len();
    let log_n = num_rows.trailing_zeros();

    // Build flat u64 data directly: row[i] = [col0[bitrev(i)], col1[bitrev(i)], ...]
    let mut flat_data: Vec<u64> = Vec::with_capacity(num_rows * num_cols);
    for i in 0..num_rows {
        let src_idx = i.reverse_bits() >> (usize::BITS - log_n);
        for col in lde_columns {
            flat_data.push(Goldilocks64Field::canonical(col[src_idx].value()));
        }
    }

    // Hash + build tree in single GPU command buffer
    let (nodes, root) =
        gpu_hash_and_build_tree(&flat_data, num_rows, num_cols, keccak_state).ok()?;

    let tree = BatchedMerkleTree::<Goldilocks64Field>::from_nodes(nodes)?;
    Some((tree, root))
}

/// Top-level GPU Merkle commit for composition polynomial (paired rows variant).
///
/// The composition polynomial commit transposes, bit-reverses, then pairs
/// consecutive rows before hashing. Computes flat u64 data directly without
/// intermediate Vec-of-Vec allocations, and hashes + builds tree in a single
/// GPU command buffer.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn gpu_batch_commit_paired_goldilocks(
    lde_evaluations: &[Vec<FieldElement<Goldilocks64Field>>],
    keccak_state: &GpuKeccakMerkleState,
) -> Option<(BatchedMerkleTree<Goldilocks64Field>, Commitment)> {
    let num_cols = lde_evaluations.len();
    let lde_len = lde_evaluations[0].len();
    let num_merged_rows = lde_len / 2;
    let cols_per_merged_row = 2 * num_cols;
    let log_n = lde_len.trailing_zeros();

    // Build flat u64 data directly: merged_row[i] = row[bitrev(2i)] ++ row[bitrev(2i+1)]
    let mut flat_data: Vec<u64> = Vec::with_capacity(num_merged_rows * cols_per_merged_row);
    for i in 0..num_merged_rows {
        let idx0 = (2 * i).reverse_bits() >> (usize::BITS - log_n);
        let idx1 = (2 * i + 1).reverse_bits() >> (usize::BITS - log_n);
        for col in lde_evaluations {
            flat_data.push(Goldilocks64Field::canonical(col[idx0].value()));
        }
        for col in lde_evaluations {
            flat_data.push(Goldilocks64Field::canonical(col[idx1].value()));
        }
    }

    // Hash + build tree in single GPU command buffer
    let (nodes, root) = gpu_hash_and_build_tree(
        &flat_data,
        num_merged_rows,
        cols_per_merged_row,
        keccak_state,
    )
    .ok()?;

    let tree = BatchedMerkleTree::<Goldilocks64Field>::from_nodes(nodes)?;
    Some((tree, root))
}

// =============================================================================
// GPU Grinding: parallel nonce search using Keccak256
// =============================================================================

/// Search for a valid grinding nonce on the GPU.
///
/// Equivalent to `grinding::generate_nonce()` but runs massively parallel
/// on the Metal GPU. Each thread tests one nonce candidate.
///
/// The `inner_hash` is pre-computed on CPU as `Keccak256(PREFIX || seed || grinding_factor)`.
/// Each GPU thread computes `Keccak256(inner_hash || nonce_be)` and checks
/// if the first 8 bytes (big-endian u64) < `limit`.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn gpu_generate_nonce(
    seed: &[u8; 32],
    grinding_factor: u8,
    keccak_state: &GpuKeccakMerkleState,
) -> Option<u64> {
    use lambdaworks_gpu::metal::abstractions::state::MetalState;
    use sha3::{Digest, Keccak256};

    if grinding_factor == 0 {
        return Some(0);
    }

    // Step 1: Compute inner_hash on CPU (single hash, not worth GPU)
    let prefix: [u8; 8] = [0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xed];
    let mut inner_data = [0u8; 41];
    inner_data[0..8].copy_from_slice(&prefix);
    inner_data[8..40].copy_from_slice(seed);
    inner_data[40] = grinding_factor;
    let digest = Keccak256::digest(inner_data);
    let inner_hash: [u8; 32] = digest[..32].try_into().unwrap();

    let limit: u64 = 1u64 << (64 - grinding_factor);

    // Step 2: Prepare GPU buffers
    let inner_hash_buf = keccak_state.state.alloc_buffer_with_data(&inner_hash).ok()?;
    let limit_buf = keccak_state.state.alloc_buffer_with_data(&[limit]).ok()?;

    // Dispatch in batches of 2^20 (1M) threads per dispatch.
    // For grinding_factor=20, expected ~1M attempts on average.
    let batch_size: u64 = 1 << 20;

    let mut batch_offset: u64 = 0;
    loop {
        // Result = single u32 initialized to UINT_MAX (no valid nonce found).
        // The kernel uses atomic_fetch_min to track the smallest valid gid.
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
                "keccak256_grind_nonce",
                &[&inner_hash_buf, &result_buf, &limit_buf, &offset_buf],
                batch_size,
                keccak_state.grind_max_threads,
            )
            .ok()?;

        // Check if any thread found a valid nonce
        let result: Vec<u32> = MetalState::retrieve_contents(&result_buf);
        let min_gid = result[0];
        if min_gid != u32::MAX {
            return Some(batch_offset + min_gid as u64);
        }

        batch_offset += batch_size;
        if batch_offset >= u64::MAX - batch_size {
            return None; // Exhausted search space
        }
    }
}

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

    // =========================================================================
    // Task 4: GPU Keccak256 differential tests
    // =========================================================================

    /// Helper: compute CPU Keccak256 hash of a row of field elements
    fn cpu_hash_row(row: &[FpE]) -> [u8; 32] {
        let mut hasher = Keccak256::new();
        for element in row.iter() {
            hasher.update(element.as_bytes());
        }
        let mut result = [0u8; 32];
        result.copy_from_slice(&hasher.finalize());
        result
    }

    /// Helper: compute CPU Keccak256 hash of two 32-byte children
    fn cpu_hash_pair(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
        let mut hasher = Keccak256::new();
        hasher.update(left);
        hasher.update(right);
        let mut result = [0u8; 32];
        result.copy_from_slice(&hasher.finalize());
        result
    }

    #[test]
    fn gpu_keccak256_leaf_hashes_match_cpu() {
        let keccak_state = GpuKeccakMerkleState::new().unwrap();

        // 3 columns × 256 rows of field elements
        let num_cols = 3;
        let num_rows = 256;
        let rows: Vec<Vec<FpE>> = (0..num_rows)
            .map(|r| {
                (0..num_cols)
                    .map(|c| FpE::from((r * num_cols + c + 1) as u64))
                    .collect()
            })
            .collect();

        let gpu_hashes = gpu_hash_leaves_goldilocks(&rows, &keccak_state).unwrap();
        assert_eq!(gpu_hashes.len(), num_rows);

        for (i, row) in rows.iter().enumerate() {
            let cpu_hash = cpu_hash_row(row);
            assert_eq!(
                gpu_hashes[i],
                cpu_hash,
                "Leaf hash mismatch at row {i}: GPU={:?} CPU={:?}",
                &gpu_hashes[i][..4],
                &cpu_hash[..4]
            );
        }
    }

    #[test]
    fn gpu_keccak256_leaf_hashes_single_column() {
        let keccak_state = GpuKeccakMerkleState::new().unwrap();

        // 1 column × 128 rows
        let rows: Vec<Vec<FpE>> = (0..128).map(|r| vec![FpE::from(r as u64 + 42)]).collect();

        let gpu_hashes = gpu_hash_leaves_goldilocks(&rows, &keccak_state).unwrap();

        for (i, row) in rows.iter().enumerate() {
            let cpu_hash = cpu_hash_row(row);
            assert_eq!(
                gpu_hashes[i], cpu_hash,
                "Single-column leaf hash mismatch at row {i}"
            );
        }
    }

    #[test]
    fn gpu_keccak256_leaf_hashes_wide_rows() {
        let keccak_state = GpuKeccakMerkleState::new().unwrap();

        // 8 columns × 64 rows (simulating composition poly with paired rows)
        let num_cols = 8;
        let num_rows = 64;
        let rows: Vec<Vec<FpE>> = (0..num_rows)
            .map(|r| {
                (0..num_cols)
                    .map(|c| FpE::from((r * 1000 + c * 7 + 13) as u64))
                    .collect()
            })
            .collect();

        let gpu_hashes = gpu_hash_leaves_goldilocks(&rows, &keccak_state).unwrap();

        for (i, row) in rows.iter().enumerate() {
            let cpu_hash = cpu_hash_row(row);
            assert_eq!(
                gpu_hashes[i], cpu_hash,
                "Wide-row leaf hash mismatch at row {i}"
            );
        }
    }

    #[test]
    fn gpu_keccak256_pair_hashes_match_cpu() {
        let keccak_state = GpuKeccakMerkleState::new().unwrap();

        // Create 256 random-ish hashes (enough to exercise GPU path, threshold is 64)
        let children: Vec<[u8; 32]> = (0..256)
            .map(|i| {
                let mut hash = [0u8; 32];
                // Fill with deterministic pattern
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
            assert_eq!(
                gpu_parents[i], cpu_parent,
                "Pair hash mismatch at index {i}"
            );
        }
    }

    #[test]
    fn gpu_keccak256_full_tree_matches_cpu_batch_commit() {
        let keccak_state = GpuKeccakMerkleState::new().unwrap();

        // 3 columns × 256 rows (typical LDE data)
        let num_cols = 3;
        let num_rows = 256;
        let columns: Vec<Vec<FpE>> = (0..num_cols)
            .map(|c| {
                (0..num_rows)
                    .map(|r| FpE::from((c * num_rows + r + 1) as u64))
                    .collect()
            })
            .collect();

        // CPU path: columns2rows_bit_reversed + cpu_batch_commit
        let cpu_rows = columns2rows_bit_reversed(&columns);
        let (cpu_tree, cpu_root) = cpu_batch_commit(&cpu_rows).unwrap();

        // GPU path: gpu_batch_commit_goldilocks
        let (gpu_tree, gpu_root) = gpu_batch_commit_goldilocks(&columns, &keccak_state).unwrap();

        // Roots must match
        assert_eq!(
            gpu_root,
            cpu_root,
            "Merkle root mismatch: GPU={:?} CPU={:?}",
            &gpu_root[..8],
            &cpu_root[..8]
        );

        // Verify proof extraction compatibility: proofs at multiple positions must match
        for pos in [0, 1, num_rows / 2, num_rows - 1] {
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
    fn gpu_keccak256_tree_non_power_of_two_leaves() {
        let keccak_state = GpuKeccakMerkleState::new().unwrap();

        // 2 columns × 200 rows (non-power-of-two, triggers padding)
        let num_cols = 2;
        let num_rows = 200;
        let columns: Vec<Vec<FpE>> = (0..num_cols)
            .map(|c| {
                (0..num_rows)
                    .map(|r| FpE::from((c * 1000 + r + 1) as u64))
                    .collect()
            })
            .collect();

        let cpu_rows = columns2rows_bit_reversed(&columns);
        let (_, cpu_root) = cpu_batch_commit(&cpu_rows).unwrap();

        // For non-power-of-two, gpu_transpose_bitrev requires power-of-two rows.
        // So we use the CPU transpose + GPU hashing path instead.
        let leaf_hashes = gpu_hash_leaves_goldilocks(&cpu_rows, &keccak_state).unwrap();
        let (_, gpu_root) = gpu_build_merkle_tree(&leaf_hashes, &keccak_state).unwrap();

        assert_eq!(gpu_root, cpu_root, "Non-power-of-two tree root mismatch");
    }

    // =========================================================================
    // Task 5: GPU transpose + bit-reverse differential tests
    // =========================================================================

    #[test]
    fn gpu_transpose_bitrev_matches_cpu() {
        let keccak_state = GpuKeccakMerkleState::new().unwrap();

        // 3 columns × 256 rows (power of two)
        let num_cols = 3;
        let num_rows = 256;
        let columns: Vec<Vec<FpE>> = (0..num_cols)
            .map(|c| {
                (0..num_rows)
                    .map(|r| FpE::from((c * num_rows + r + 1) as u64))
                    .collect()
            })
            .collect();

        let cpu_rows = columns2rows_bit_reversed(&columns);
        let gpu_rows = gpu_transpose_bitrev(&columns, &keccak_state).unwrap();

        assert_eq!(gpu_rows.len(), cpu_rows.len(), "Row count mismatch");
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

        // 2 columns × 8 rows (small case)
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

    // =========================================================================
    // Task 5b: GPU transpose+bitrev from Metal buffers differential tests
    // =========================================================================

    #[test]
    fn gpu_transpose_bitrev_buffer_matches_cpu() {
        use lambdaworks_math::field::traits::IsPrimeField;
        use metal::MTLResourceOptions;

        let keccak_state = GpuKeccakMerkleState::new().unwrap();
        let transpose_state = keccak_state.transpose_state().expect("transpose state");

        // 3 columns x 256 rows (power of two)
        let num_cols = 3;
        let num_rows: usize = 256;
        let columns: Vec<Vec<FpE>> = (0..num_cols)
            .map(|c| {
                (0..num_rows)
                    .map(|r| FpE::from((c * num_rows + r + 1) as u64))
                    .collect()
            })
            .collect();

        // CPU reference: columns2rows_bit_reversed
        let cpu_rows = columns2rows_bit_reversed(&columns);

        // Create Metal buffers for each column (simulating FFT output)
        let device = transpose_state.state.device();
        let col_buffers: Vec<metal::Buffer> = columns
            .iter()
            .map(|col| {
                let data: Vec<u64> = col
                    .iter()
                    .map(|fe| Goldilocks64Field::canonical(fe.value()))
                    .collect();
                device.new_buffer_with_data(
                    data.as_ptr() as *const _,
                    (data.len() * std::mem::size_of::<u64>()) as u64,
                    MTLResourceOptions::StorageModeShared,
                )
            })
            .collect();
        let col_buf_refs: Vec<&metal::Buffer> = col_buffers.iter().collect();

        // GPU transpose from buffers
        let result_buf =
            gpu_transpose_bitrev_to_buffer(&col_buf_refs, num_rows, transpose_state).unwrap();

        // Read back result
        let gpu_raw: Vec<u64> = unsafe {
            let ptr = result_buf.contents() as *const u64;
            std::slice::from_raw_parts(ptr, num_rows * num_cols).to_vec()
        };

        // Compare with CPU reference
        for (row_idx, cpu_row) in cpu_rows.iter().enumerate() {
            for (col_idx, cpu_val) in cpu_row.iter().enumerate() {
                let gpu_val = gpu_raw[row_idx * num_cols + col_idx];
                let expected = Goldilocks64Field::canonical(cpu_val.value());
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
        use lambdaworks_math::field::traits::IsPrimeField;
        use metal::MTLResourceOptions;

        let keccak_state = GpuKeccakMerkleState::new().unwrap();
        let transpose_state = keccak_state.transpose_state().expect("transpose state");

        // 2 columns x 128 rows
        let num_cols = 2;
        let num_rows: usize = 128;
        let columns: Vec<Vec<FpE>> = (0..num_cols)
            .map(|c| {
                (0..num_rows)
                    .map(|r| FpE::from((c * num_rows + r + 1) as u64))
                    .collect()
            })
            .collect();

        // CPU reference: transpose + bit-reverse + pair
        let mut cpu_rows: Vec<Vec<FpE>> = (0..num_rows)
            .map(|i| columns.iter().map(|col| col[i].clone()).collect())
            .collect();
        in_place_bit_reverse_permute(&mut cpu_rows);
        let mut merged_rows = Vec::with_capacity(num_rows / 2);
        let mut iter = cpu_rows.into_iter();
        while let (Some(mut chunk0), Some(chunk1)) = (iter.next(), iter.next()) {
            chunk0.extend(chunk1);
            merged_rows.push(chunk0);
        }

        // Create Metal buffers for each column
        let device = transpose_state.state.device();
        let col_buffers: Vec<metal::Buffer> = columns
            .iter()
            .map(|col| {
                let data: Vec<u64> = col
                    .iter()
                    .map(|fe| Goldilocks64Field::canonical(fe.value()))
                    .collect();
                device.new_buffer_with_data(
                    data.as_ptr() as *const _,
                    (data.len() * std::mem::size_of::<u64>()) as u64,
                    MTLResourceOptions::StorageModeShared,
                )
            })
            .collect();
        let col_buf_refs: Vec<&metal::Buffer> = col_buffers.iter().collect();

        // GPU paired transpose from buffers
        let result_buf =
            gpu_transpose_bitrev_paired_to_buffer(&col_buf_refs, num_rows, transpose_state)
                .unwrap();

        // Read back result
        let num_merged_rows = num_rows / 2;
        let cols_per_merged = 2 * num_cols;
        let gpu_raw: Vec<u64> = unsafe {
            let ptr = result_buf.contents() as *const u64;
            std::slice::from_raw_parts(ptr, num_merged_rows * cols_per_merged).to_vec()
        };

        // Compare with CPU reference
        for (row_idx, cpu_row) in merged_rows.iter().enumerate() {
            for (col_idx, cpu_val) in cpu_row.iter().enumerate() {
                let gpu_val = gpu_raw[row_idx * cols_per_merged + col_idx];
                let expected = Goldilocks64Field::canonical(cpu_val.value());
                assert_eq!(
                    gpu_val, expected,
                    "Paired buffer transpose mismatch at row {row_idx}, col {col_idx}"
                );
            }
        }
    }

    #[test]
    fn gpu_transpose_bitrev_buffer_small() {
        use lambdaworks_math::field::traits::IsPrimeField;
        use metal::MTLResourceOptions;

        let keccak_state = GpuKeccakMerkleState::new().unwrap();
        let transpose_state = keccak_state.transpose_state().expect("transpose state");

        // 2 columns x 8 rows (small case)
        let columns: Vec<Vec<FpE>> = vec![
            (0..8).map(|i| FpE::from(i as u64)).collect(),
            (0..8).map(|i| FpE::from(i as u64 + 100)).collect(),
        ];

        let cpu_rows = columns2rows_bit_reversed(&columns);

        // Create Metal buffers
        let device = transpose_state.state.device();
        let col_buffers: Vec<metal::Buffer> = columns
            .iter()
            .map(|col| {
                let data: Vec<u64> = col
                    .iter()
                    .map(|fe| Goldilocks64Field::canonical(fe.value()))
                    .collect();
                device.new_buffer_with_data(
                    data.as_ptr() as *const _,
                    (data.len() * std::mem::size_of::<u64>()) as u64,
                    MTLResourceOptions::StorageModeShared,
                )
            })
            .collect();
        let col_buf_refs: Vec<&metal::Buffer> = col_buffers.iter().collect();

        let result_buf =
            gpu_transpose_bitrev_to_buffer(&col_buf_refs, 8, transpose_state).unwrap();

        let num_cols = 2;
        let num_rows = 8;
        let gpu_raw: Vec<u64> = unsafe {
            let ptr = result_buf.contents() as *const u64;
            std::slice::from_raw_parts(ptr, num_rows * num_cols).to_vec()
        };

        for (row_idx, cpu_row) in cpu_rows.iter().enumerate() {
            for (col_idx, cpu_val) in cpu_row.iter().enumerate() {
                let gpu_val = gpu_raw[row_idx * num_cols + col_idx];
                let expected = Goldilocks64Field::canonical(cpu_val.value());
                assert_eq!(
                    gpu_val, expected,
                    "Small buffer transpose mismatch at row {row_idx}, col {col_idx}"
                );
            }
        }
    }

    // =========================================================================
    // Task 6: Integrated gpu_batch_commit differential test
    // =========================================================================

    #[test]
    fn gpu_batch_commit_matches_cpu() {
        let keccak_state = GpuKeccakMerkleState::new().unwrap();

        // Simulate realistic LDE data: 3 columns × 512 rows
        let num_cols = 3;
        let num_rows = 512;
        let columns: Vec<Vec<FpE>> = (0..num_cols)
            .map(|c| {
                (0..num_rows)
                    .map(|r| FpE::from((c * num_rows + r + 1) as u64))
                    .collect()
            })
            .collect();

        // CPU path
        let cpu_rows = columns2rows_bit_reversed(&columns);
        let (cpu_tree, cpu_root) = cpu_batch_commit(&cpu_rows).unwrap();

        // GPU path
        let (gpu_tree, gpu_root) = gpu_batch_commit_goldilocks(&columns, &keccak_state).unwrap();

        assert_eq!(gpu_root, cpu_root, "Batch commit root mismatch");

        // Verify proof compatibility at multiple positions
        for pos in [0, 1, num_rows / 4, num_rows / 2, num_rows - 1] {
            let cpu_proof = cpu_tree.get_proof_by_pos(pos);
            let gpu_proof = gpu_tree.get_proof_by_pos(pos);
            assert_eq!(
                cpu_proof.is_some(),
                gpu_proof.is_some(),
                "Batch commit proof availability mismatch at pos {pos}"
            );
            if let (Some(cp), Some(gp)) = (cpu_proof, gpu_proof) {
                assert_eq!(
                    cp.merkle_path, gp.merkle_path,
                    "Batch commit proof path mismatch at pos {pos}"
                );
            }
        }
    }

    #[test]
    fn gpu_batch_commit_paired_matches_cpu() {
        let keccak_state = GpuKeccakMerkleState::new().unwrap();

        use lambdaworks_math::fft::cpu::bit_reversing::in_place_bit_reverse_permute;

        // Simulate composition poly LDE: 2 parts × 128 evaluations
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
            .map(|i| lde_evaluations.iter().map(|col| col[i].clone()).collect())
            .collect();
        in_place_bit_reverse_permute(&mut cpu_rows);
        let mut merged_rows = Vec::with_capacity(lde_len / 2);
        let mut iter = cpu_rows.into_iter();
        while let (Some(mut chunk0), Some(chunk1)) = (iter.next(), iter.next()) {
            chunk0.extend(chunk1);
            merged_rows.push(chunk0);
        }
        let (cpu_tree, cpu_root) = cpu_batch_commit(&merged_rows).unwrap();

        // GPU path
        let (gpu_tree, gpu_root) =
            gpu_batch_commit_paired_goldilocks(&lde_evaluations, &keccak_state).unwrap();

        assert_eq!(gpu_root, cpu_root, "Paired commit root mismatch");

        // Verify proof compatibility
        let num_leaves = lde_len / 2;
        for pos in [0, 1, num_leaves / 2, num_leaves - 1] {
            let cpu_proof = cpu_tree.get_proof_by_pos(pos);
            let gpu_proof = gpu_tree.get_proof_by_pos(pos);
            assert_eq!(
                cpu_proof.is_some(),
                gpu_proof.is_some(),
                "Paired commit proof availability mismatch at pos {pos}"
            );
            if let (Some(cp), Some(gp)) = (cpu_proof, gpu_proof) {
                assert_eq!(
                    cp.merkle_path, gp.merkle_path,
                    "Paired commit proof path mismatch at pos {pos}"
                );
            }
        }
    }

    // =========================================================================
    // GPU grinding differential tests
    // =========================================================================

    #[test]
    fn gpu_grinding_matches_cpu() {
        use stark_platinum_prover::grinding;

        let keccak_state = GpuKeccakMerkleState::new().unwrap();

        // Test with known seeds and grinding factors from the CPU test suite
        let seed1 = [
            37, 68, 26, 150, 139, 142, 66, 175, 33, 47, 199, 160, 9, 109, 79, 234, 135, 254, 39,
            11, 225, 219, 206, 108, 224, 165, 25, 72, 189, 96, 218, 95,
        ];

        // grinding_factor=0: should return Some(0)
        assert_eq!(gpu_generate_nonce(&seed1, 0, &keccak_state), Some(0));

        // grinding_factor=1: very easy, first valid nonce should match CPU
        let cpu_nonce = grinding::generate_nonce(&seed1, 1).unwrap();
        let gpu_nonce = gpu_generate_nonce(&seed1, 1, &keccak_state).unwrap();
        assert_eq!(gpu_nonce, cpu_nonce, "grinding_factor=1 mismatch");
        assert!(grinding::is_valid_nonce(&seed1, gpu_nonce, 1));

        // grinding_factor=10: known valid nonce is 0x5ba
        let cpu_nonce = grinding::generate_nonce(&seed1, 10).unwrap();
        let gpu_nonce = gpu_generate_nonce(&seed1, 10, &keccak_state).unwrap();
        assert_eq!(gpu_nonce, cpu_nonce, "grinding_factor=10 mismatch");
        assert!(grinding::is_valid_nonce(&seed1, gpu_nonce, 10));

        // Test with a different seed
        let seed2 = [
            174, 187, 26, 134, 6, 43, 222, 151, 140, 48, 52, 67, 69, 181, 177, 165, 111, 222,
            148, 92, 130, 241, 171, 2, 62, 34, 95, 159, 37, 116, 155, 217,
        ];
        let cpu_nonce = grinding::generate_nonce(&seed2, 1).unwrap();
        let gpu_nonce = gpu_generate_nonce(&seed2, 1, &keccak_state).unwrap();
        assert_eq!(gpu_nonce, cpu_nonce, "seed2 grinding_factor=1 mismatch");
    }
}
