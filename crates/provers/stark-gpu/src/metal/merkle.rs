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
        Ok(Self {
            state,
            hash_leaves_max_threads,
            hash_pairs_max_threads,
            transpose_max_threads,
        })
    }
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

    let buf_data = keccak_state.state.alloc_buffer_with_data(&flat_data)?;
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
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn gpu_build_merkle_tree(
    leaf_hashes: &[[u8; 32]],
    keccak_state: &GpuKeccakMerkleState,
) -> Result<(Vec<[u8; 32]>, [u8; 32]), MetalError> {
    if leaf_hashes.is_empty() {
        return Err(MetalError::ExecutionError("Empty leaf hashes".to_string()));
    }

    // Pad to power of two
    let mut leaves: Vec<[u8; 32]> = leaf_hashes.to_vec();
    while !leaves.len().is_power_of_two() {
        leaves.push(*leaves.last().unwrap());
    }

    let leaves_len = leaves.len();

    // Allocate flat node array: [inner_nodes (leaves_len - 1) | leaves (leaves_len)]
    let total_nodes = 2 * leaves_len - 1;
    let mut nodes = vec![[0u8; 32]; total_nodes];

    // Copy leaves into the second half
    nodes[leaves_len - 1..].copy_from_slice(&leaves);

    // Build tree bottom-up, level by level
    // Same indexing as utils::build: inner nodes at [0..leaves_len-1], leaves at [leaves_len-1..]
    let mut level_begin = leaves_len - 1;
    let mut level_end = 2 * level_begin; // inclusive

    while level_begin != level_end {
        let new_level_begin = level_begin / 2;
        // Children for this level: nodes[level_begin..=level_end]
        let children_slice = &nodes[level_begin..=level_end];
        let parents = gpu_hash_tree_level(children_slice, keccak_state)?;

        // Copy parents into their positions
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

/// Top-level GPU Merkle commit for Goldilocks field.
///
/// Replaces the CPU path of `columns2rows_bit_reversed() + cpu_batch_commit()`.
/// Takes column-major LDE evaluations, returns a MerkleTree + root commitment.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn gpu_batch_commit_goldilocks(
    lde_columns: &[Vec<FieldElement<Goldilocks64Field>>],
    keccak_state: &GpuKeccakMerkleState,
) -> Option<(BatchedMerkleTree<Goldilocks64Field>, Commitment)> {
    // Step 1: Transpose + bit-reverse on GPU
    let rows = gpu_transpose_bitrev(lde_columns, keccak_state).ok()?;

    // Step 2: Hash leaves on GPU
    let leaf_hashes = gpu_hash_leaves_goldilocks(&rows, keccak_state).ok()?;

    // Step 3: Build tree on GPU
    let (nodes, root) = gpu_build_merkle_tree(&leaf_hashes, keccak_state).ok()?;

    // Step 4: Construct MerkleTree from pre-computed nodes
    let tree = BatchedMerkleTree::<Goldilocks64Field>::from_nodes(nodes)?;

    Some((tree, root))
}

/// Top-level GPU Merkle commit for composition polynomial (paired rows variant).
///
/// The composition polynomial commit transposes, bit-reverses, then pairs
/// consecutive rows before hashing. This variant handles that layout.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn gpu_batch_commit_paired_goldilocks(
    lde_evaluations: &[Vec<FieldElement<Goldilocks64Field>>],
    keccak_state: &GpuKeccakMerkleState,
) -> Option<(BatchedMerkleTree<Goldilocks64Field>, Commitment)> {
    use lambdaworks_math::fft::cpu::bit_reversing::in_place_bit_reverse_permute;

    let lde_len = lde_evaluations[0].len();

    // Transpose: column-major to row-major
    let mut rows: Vec<Vec<FieldElement<Goldilocks64Field>>> = (0..lde_len)
        .map(|i| lde_evaluations.iter().map(|col| col[i]).collect())
        .collect();

    // Bit-reverse permute
    in_place_bit_reverse_permute(&mut rows);

    // Pair consecutive rows (merge row 2i and row 2i+1 into one row)
    let mut merged_rows = Vec::with_capacity(lde_len / 2);
    let mut iter = rows.into_iter();
    while let (Some(mut chunk0), Some(chunk1)) = (iter.next(), iter.next()) {
        chunk0.extend(chunk1);
        merged_rows.push(chunk0);
    }

    // Hash merged rows on GPU
    let leaf_hashes = gpu_hash_leaves_goldilocks(&merged_rows, keccak_state).ok()?;

    // Build tree on GPU
    let (nodes, root) = gpu_build_merkle_tree(&leaf_hashes, keccak_state).ok()?;

    let tree = BatchedMerkleTree::<Goldilocks64Field>::from_nodes(nodes)?;

    Some((tree, root))
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
}
