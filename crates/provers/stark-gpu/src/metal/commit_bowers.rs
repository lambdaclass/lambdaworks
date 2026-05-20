//! End-to-end Bowers-G LDE -> GPU Keccak Merkle commitment.
//!
//! `commit_matrix_bowers` takes a matrix of polynomial coefficient columns
//! (already padded to the LDE output size), runs the fused Bowers-G NTT + coset
//! LDE on the GPU, then builds a batched Keccak Merkle tree.
//!
//! The Bowers LDE runs on a `MetalState` and the Keccak Merkle on a
//! `GpuKeccakMerkleState`; the natural-order evaluations are read back to the
//! host between the two stages so the two Metal states stay decoupled. Fusing
//! the two stages into a single device buffer is a future optimization.

#![cfg(all(target_os = "macos", feature = "metal"))]

use lambdaworks_gpu::metal::abstractions::{errors::MetalError, state::MetalState};
use lambdaworks_math::fft::gpu::metal::bowers::dispatcher::metal_bowers_lde;
use lambdaworks_math::fft::gpu::metal::bowers::twiddles::{
    cached_bowers_twiddles_goldilocks, cached_coset_powers_goldilocks,
};
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::u64_goldilocks_field::Goldilocks64Field;
use stark_platinum_prover::config::{BatchedMerkleTree, Commitment};

use crate::metal::merkle::{gpu_batch_commit_from_column_buffers, GpuKeccakMerkleState};

type FpE = FieldElement<Goldilocks64Field>;

/// Bit-reverse a slice in place (`log2(len)` bits).
fn bit_reverse<T>(data: &mut [T]) {
    let n = data.len();
    if n <= 1 {
        return;
    }
    let log_n = n.trailing_zeros();
    for i in 0..n {
        let j = ((i as u32).reverse_bits() >> (u32::BITS - log_n)) as usize;
        if i < j {
            data.swap(i, j);
        }
    }
}

/// Commit to a matrix of Goldilocks polynomial-coefficient columns via a fused
/// Bowers-G coset LDE followed by a GPU Keccak Merkle tree build.
///
/// - `coeff_cols`: `num_cols` columns, each exactly `n = 1 << log_n` coefficients
///   (the caller is responsible for zero-padding to the LDE output size).
/// - `coset_offset`: multiplicative coset offset for the LDE.
/// - `lde_state`: Metal state used for the Bowers NTT.
/// - `keccak_state`: Metal state used for the Keccak Merkle tree.
///
/// Returns the batched Merkle tree and its root commitment. The leaf order
/// matches the existing GPU prover's column-buffer commit path (bit-reversed
/// rows), so the resulting tree is interchangeable with `evaluate_offset_fft`
/// + `gpu_batch_commit_from_column_buffers`.
pub fn commit_matrix_bowers(
    coeff_cols: &[Vec<FpE>],
    coset_offset: FpE,
    lde_state: &MetalState,
    keccak_state: &GpuKeccakMerkleState,
) -> Result<(BatchedMerkleTree<Goldilocks64Field>, Commitment), MetalError> {
    if coeff_cols.is_empty() {
        return Err(MetalError::InputError(0));
    }
    let n = coeff_cols[0].len();
    if !n.is_power_of_two() || n < 4 {
        return Err(MetalError::InputError(n));
    }
    if coeff_cols.iter().any(|c| c.len() != n) {
        return Err(MetalError::InputError(n));
    }
    let num_cols = coeff_cols.len();
    let log_n = n.trailing_zeros();

    // Pack coefficient columns column-major into a single u64 buffer.
    let mut packed: Vec<u64> = Vec::with_capacity(n * num_cols);
    for col in coeff_cols {
        packed.extend(col.iter().map(|x| *x.value()));
    }
    let cols_buf = lde_state.alloc_buffer_data(&packed);

    // Bit-reversed Bowers twiddles (cached).
    let twiddles = cached_bowers_twiddles_goldilocks(log_n);
    let tw_u64: Vec<u64> = twiddles.iter().map(|x| *x.value()).collect();
    let tw_buf = lde_state.alloc_buffer_data(&tw_u64);

    // Coset powers, supplied bit-reversed (see `metal_bowers_lde` docs).
    let coset = cached_coset_powers_goldilocks(log_n, coset_offset);
    let mut coset_u64: Vec<u64> = coset.iter().map(|x| *x.value()).collect();
    bit_reverse(&mut coset_u64);
    let coset_buf = lde_state.alloc_buffer_data(&coset_u64);

    // Fused Bowers-G NTT + coset LDE -> natural-order evaluations.
    let mut out_buf = lde_state.alloc_buffer::<u64>(n * num_cols);
    metal_bowers_lde(
        &cols_buf,
        &tw_buf,
        &coset_buf,
        &mut out_buf,
        log_n,
        num_cols as u32,
        lde_state,
    )?;
    let evals: Vec<u64> = MetalState::retrieve_contents(&out_buf);

    // Re-upload each column onto the Keccak Merkle device and commit.
    let column_buffers: Vec<metal::Buffer> = (0..num_cols)
        .map(|c| {
            let slice = &evals[c * n..(c + 1) * n];
            keccak_state.state.alloc_buffer_with_data(slice)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let column_refs: Vec<&metal::Buffer> = column_buffers.iter().collect();

    gpu_batch_commit_from_column_buffers(&column_refs, n, keccak_state)
        .ok_or_else(|| MetalError::ExecutionError("Bowers Merkle commit failed".to_string()))
}
