//! Rust dispatcher for the Bowers-G fused LDE Metal kernels.
//!
//! Stage convention matches the CPU reference `ntt_bowers_butterflies`:
//!   stage 0   -> smallest stride (half = 1)
//!   stage k   -> half = 2^k
//!   stage log_n - 1 -> largest stride (half = n/2)
//!
//! Pipeline:
//!   - Bit-reverse the input columns (`cols` -> `out`). The Bowers-G network
//!     requires bit-reversed input and yields natural-order output.
//!   - Fused HEAD kernel covers the first K stages (smallest strides) in
//!     threadgroup memory, applying the coset multiply on load.
//!   - Per-stage MIDDLE dispatches handle stages K..log_n.
//!
//! Output is in natural order, so `coset_powers` must be supplied in
//! bit-reversed order: the head kernel multiplies the (already bit-reversed)
//! data element-wise by `coset_powers`, and `bitrev(c) * bitrev(g)` =
//! `bitrev(c * g)` gives the correct pre-image for the butterfly network.
//!
//! Twiddles and coset powers are always base-field Goldilocks (`u64`), even on
//! the Fp3 path (extension-field FFT with base-field twiddles).

use lambdaworks_gpu::metal::abstractions::{errors::MetalError, state::*};
use metal::{Buffer, MTLSize};

use core::mem;

/// Threadgroup memory budget in bytes (32 KB).
const FFT_TG_MEM_BUDGET: usize = 32768;

/// Maximum fused stages cap (avoids excessive register pressure).
const FFT_MAX_FUSED_STAGES: u32 = 12;

/// Minimum log2 of input size.
const BOWERS_MIN_LOG_N: u32 = 2;

/// Bytes per Goldilocks base-field element (u64).
const GOLDILOCKS_ELEM_SIZE: usize = 8;

/// Bytes per Goldilocks Fp3 element (3 x u64).
const GOLDILOCKS_FP3_ELEM_SIZE: usize = 24;

/// Kernel name set for one field variant.
struct BowersKernels {
    bitrev: &'static str,
    head: &'static str,
    middle: &'static str,
}

const GOLDILOCKS_KERNELS: BowersKernels = BowersKernels {
    bitrev: "bitrev_permutation_Goldilocks",
    head: "bowers_lde_head_Goldilocks",
    middle: "bowers_lde_middle_Goldilocks",
};

const GOLDILOCKS_FP3_KERNELS: BowersKernels = BowersKernels {
    bitrev: "bitrev_permutation_Goldilocks_fp3",
    head: "bowers_lde_head_Goldilocks_fp3",
    middle: "bowers_lde_middle_Goldilocks_fp3",
};

/// Computes the optimal number of fused head stages for a given element size.
///
/// Capped at `log_n - 1` so at least one middle dispatch remains (covering the
/// largest-stride stage).
fn optimal_fused_head_stages(log_n: u32, elem_size: usize) -> u32 {
    let max_block = FFT_TG_MEM_BUDGET / elem_size;
    let max_fused = max_block.ilog2();
    max_fused
        .min(log_n.saturating_sub(1))
        .min(FFT_MAX_FUSED_STAGES)
}

/// Generic Bowers-G NTT + fused LDE dispatch, parameterised by element size and
/// the kernel name set. `elem_size` is the size in bytes of one column element
/// (8 for base Goldilocks, 24 for Fp3).
#[allow(clippy::too_many_arguments)]
fn dispatch_bowers_lde(
    cols: &Buffer,
    twiddles_bitrev: &Buffer,
    coset_powers: &Buffer,
    out: &mut Buffer,
    log_n: u32,
    num_cols: u32,
    elem_size: usize,
    kernels: &BowersKernels,
    state: &MetalState,
) -> Result<(), MetalError> {
    if log_n < BOWERS_MIN_LOG_N {
        return Err(MetalError::InputError(1 << log_n));
    }

    let n = 1u32 << log_n;
    let col_bytes = (n as usize) * elem_size;
    let k = optimal_fused_head_stages(log_n, elem_size);
    let k_val: u32 = k;

    let pipeline_bitrev = state.get_pipeline(kernels.bitrev)?;
    let pipeline_head = state.get_pipeline(kernels.head)?;
    let pipeline_middle = state.get_pipeline(kernels.middle)?;

    let full_grid = MTLSize::new(n as u64, 1, 1);
    let half_grid = MTLSize::new(n as u64 / 2, 1, 1);
    let bitrev_tg = MTLSize::new(
        pipeline_bitrev
            .max_total_threads_per_threadgroup()
            .min(n as u64),
        1,
        1,
    );
    let middle_tg = MTLSize::new(pipeline_middle.thread_execution_width(), 1, 1);

    let head_block_size = 1u32 << k;
    let head_groups = MTLSize::new(n as u64 / head_block_size as u64, 1, 1);
    let head_threads = 256u64
        .min(pipeline_head.max_total_threads_per_threadgroup())
        .min((head_block_size as u64 / 2).max(1));
    let head_tg = MTLSize::new(head_threads, 1, 1);
    let head_shmem = (head_block_size as u64) * (elem_size as u64);

    objc::rc::autoreleasepool(|| -> Result<(), MetalError> {
        let command_buffer = state.queue.new_command_buffer();
        let enc = command_buffer.new_compute_command_encoder();

        // ---- Bit-reverse the input columns: cols -> out ----
        enc.set_compute_pipeline_state(&pipeline_bitrev);
        for col in 0..num_cols {
            let offset = (col as usize * col_bytes) as u64;
            enc.set_buffer(0, Some(cols), offset);
            enc.set_buffer(1, Some(out), offset);
            enc.dispatch_threads(full_grid, bitrev_tg);
        }

        // ---- HEAD: stages 0..k in threadgroup memory (coset fold on load) ----
        enc.set_compute_pipeline_state(&pipeline_head);
        enc.set_buffer(1, Some(twiddles_bitrev), 0);
        enc.set_buffer(2, Some(coset_powers), 0);
        enc.set_bytes(3, mem::size_of::<u32>() as u64, void_ptr(&k_val));
        enc.set_threadgroup_memory_length(0, head_shmem);
        for col in 0..num_cols {
            let offset = (col as usize * col_bytes) as u64;
            enc.set_buffer(0, Some(out), offset);
            enc.dispatch_thread_groups(head_groups, head_tg);
        }

        // ---- MIDDLE stages: k..log_n ----
        if k < log_n {
            enc.set_compute_pipeline_state(&pipeline_middle);
            enc.set_buffer(1, Some(twiddles_bitrev), 0);
            for stage in k..log_n {
                let stage_val: u32 = stage;
                enc.set_bytes(2, mem::size_of::<u32>() as u64, void_ptr(&stage_val));
                for col in 0..num_cols {
                    let offset = (col as usize * col_bytes) as u64;
                    enc.set_buffer(0, Some(out), offset);
                    enc.dispatch_threads(half_grid, middle_tg);
                }
            }
        }

        enc.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        Ok(())
    })?;

    Ok(())
}

/// Execute the Bowers-G NTT + fused LDE pipeline for the Goldilocks base field.
///
/// - `cols`: `num_cols * n` Goldilocks `u64` elements in column-major order.
/// - `twiddles_bitrev`: `n/2` bit-reversed Bowers twiddles (`u64`).
/// - `coset_powers`: `n` bit-reversed coset powers (`u64`).
/// - `out`: receives `num_cols * n` evaluations in natural order.
pub fn metal_bowers_lde(
    cols: &Buffer,
    twiddles_bitrev: &Buffer,
    coset_powers: &Buffer,
    out: &mut Buffer,
    log_n: u32,
    num_cols: u32,
    state: &MetalState,
) -> Result<(), MetalError> {
    dispatch_bowers_lde(
        cols,
        twiddles_bitrev,
        coset_powers,
        out,
        log_n,
        num_cols,
        GOLDILOCKS_ELEM_SIZE,
        &GOLDILOCKS_KERNELS,
        state,
    )
}

/// Execute the Bowers-G NTT + fused LDE pipeline for the Goldilocks Fp3
/// extension field. Data elements are Fp3 (3 x u64); twiddles and coset powers
/// remain base-field Goldilocks `u64`.
///
/// - `cols`: `num_cols * n` Fp3 elements (`3 * u64` each) in column-major order.
/// - `twiddles_bitrev`: `n/2` bit-reversed Bowers twiddles (base-field `u64`).
/// - `coset_powers`: `n` bit-reversed coset powers (base-field `u64`).
/// - `out`: receives `num_cols * n` Fp3 evaluations in natural order.
pub fn metal_bowers_lde_fp3(
    cols: &Buffer,
    twiddles_bitrev: &Buffer,
    coset_powers: &Buffer,
    out: &mut Buffer,
    log_n: u32,
    num_cols: u32,
    state: &MetalState,
) -> Result<(), MetalError> {
    dispatch_bowers_lde(
        cols,
        twiddles_bitrev,
        coset_powers,
        out,
        log_n,
        num_cols,
        GOLDILOCKS_FP3_ELEM_SIZE,
        &GOLDILOCKS_FP3_KERNELS,
        state,
    )
}

/// Test-only: run the Bowers-G NTT without coset multiply (identity coset).
#[cfg(test)]
pub(crate) fn metal_bowers_fft_no_coset(
    cols: &Buffer,
    twiddles_bitrev: &Buffer,
    out: &mut Buffer,
    log_n: u32,
    num_cols: u32,
    state: &MetalState,
) -> Result<(), MetalError> {
    let n = 1usize << log_n;
    let ones: Vec<u64> = vec![1u64; n];
    let ones_buf = state.alloc_buffer_data(&ones);
    metal_bowers_lde(
        cols,
        twiddles_bitrev,
        &ones_buf,
        out,
        log_n,
        num_cols,
        state,
    )
}
