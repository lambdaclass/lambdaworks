//! Rust dispatcher for the Bowers-G fused LDE Metal kernels.
//!
//! Drives three kernel variants (stage0, middle, tail) over a multi-column
//! NTT + LDE pipeline using a single command buffer per call.

use lambdaworks_gpu::metal::abstractions::{errors::MetalError, state::*};
use metal::{Buffer, MTLSize};

use core::mem;

/// Threadgroup memory budget in bytes (32 KB).
const FFT_TG_MEM_BUDGET: usize = 32768;

/// Maximum fused stages cap (avoids excessive register pressure).
const FFT_MAX_FUSED_STAGES: u32 = 12;

/// Minimum log2 of input size.
const BOWERS_MIN_LOG_N: u32 = 2;

/// Bytes per Goldilocks field element (u64).
const GOLDILOCKS_ELEM_SIZE: usize = 8;

/// Computes the optimal number of fused tail stages.
fn optimal_fused_stages_for_bowers(log_n: u32) -> u32 {
    let max_block = FFT_TG_MEM_BUDGET / GOLDILOCKS_ELEM_SIZE;
    let max_fused = max_block.ilog2();
    max_fused.min(log_n).min(FFT_MAX_FUSED_STAGES)
}

/// Execute the Bowers-G NTT + fused LDE pipeline on the GPU for Goldilocks base field.
///
/// # Parameters
/// - `cols`: GPU buffer holding `num_cols * n` Goldilocks field elements in column-major order.
/// - `twiddles_bitrev`: GPU buffer holding `n` twiddle factors in bit-reversed Bowers layout.
/// - `coset_powers`: GPU buffer holding `n` coset powers for the stage-0 LDE fold.
/// - `out`: GPU buffer (same size as `cols`) that receives the NTT output.
/// - `log_n`: Log2 of the column length; must be >= 2.
/// - `num_cols`: Number of independent columns to process.
/// - `state`: Metal state with device and compiled shader library.
pub fn metal_bowers_lde(
    cols: &Buffer,
    twiddles_bitrev: &Buffer,
    coset_powers: &Buffer,
    out: &mut Buffer,
    log_n: u32,
    num_cols: u32,
    state: &MetalState,
) -> Result<(), MetalError> {
    if log_n < BOWERS_MIN_LOG_N {
        return Err(MetalError::InputError(1 << log_n));
    }

    let n = 1u32 << log_n;
    let col_bytes = (n as usize) * GOLDILOCKS_ELEM_SIZE;

    // Compute optimal fused tail stages.
    let k = optimal_fused_stages_for_bowers(log_n);

    // Stage 0 is always dispatched first (coset-fold + first DIF butterfly).
    // Middle stages cover 1..(log_n - k) inclusive.
    // Tail fuses the last k stages.
    // When log_n <= k there are no middle stages; start_stage = log_n - k still holds.
    let middle_stage_count = if log_n > k { log_n - k - 1 } else { 0 };

    let pipeline_stage0 = state.get_pipeline("bowers_lde_stage0_Goldilocks")?;
    let pipeline_middle = state.get_pipeline("bowers_lde_middle_Goldilocks")?;
    let pipeline_tail = state.get_pipeline("bowers_lde_tail_Goldilocks")?;

    // Half-sized grid: each thread processes one butterfly (n/2 butterflies per stage).
    let half_grid = MTLSize::new(n as u64 / 2, 1, 1);
    let half_tg = MTLSize::new(pipeline_stage0.thread_execution_width(), 1, 1);

    // Tail dispatch: one threadgroup per block of (1 << k) elements.
    let block_size = 1u32 << k;
    let tail_groups = MTLSize::new(n as u64 / block_size as u64, 1, 1);
    let tail_tg = MTLSize::new(
        256u64.min(pipeline_tail.max_total_threads_per_threadgroup()),
        1,
        1,
    );
    let tail_shmem = (block_size as u64) * (GOLDILOCKS_ELEM_SIZE as u64);

    // Scalar kernel parameters.
    let n_val: u32 = n;
    let start_stage: u32 = log_n - k;
    let k_val: u32 = k;

    objc::rc::autoreleasepool(|| {
        let command_buffer = state.queue.new_command_buffer();

        // Blit cols -> out so all NTT stages operate in-place on out.
        {
            let blit = command_buffer.new_blit_command_encoder();
            blit.copy_from_buffer(cols, 0, out, 0, cols.length());
            blit.end_encoding();
        }

        // Single compute encoder for all stages and all columns.
        let enc = command_buffer.new_compute_command_encoder();

        // ---- Stage 0: coset-fold + first DIF butterfly ----
        enc.set_compute_pipeline_state(&pipeline_stage0);
        enc.set_buffer(1, Some(twiddles_bitrev), 0);
        enc.set_buffer(2, Some(coset_powers), 0);
        enc.set_bytes(3, mem::size_of::<u32>() as u64, void_ptr(&n_val));

        for col in 0..num_cols {
            let offset = (col as usize * col_bytes) as u64;
            enc.set_buffer(0, Some(out), offset);
            enc.dispatch_threads(half_grid, half_tg);
        }

        // ---- Middle stages: generic DIF butterfly ----
        if middle_stage_count > 0 {
            let middle_tg = MTLSize::new(pipeline_middle.thread_execution_width(), 1, 1);

            enc.set_compute_pipeline_state(&pipeline_middle);
            enc.set_buffer(1, Some(twiddles_bitrev), 0);
            enc.set_bytes(2, mem::size_of::<u32>() as u64, void_ptr(&n_val));

            for stage in 1..=middle_stage_count {
                let stage_val: u32 = stage;
                enc.set_bytes(3, mem::size_of::<u32>() as u64, void_ptr(&stage_val));

                for col in 0..num_cols {
                    let offset = (col as usize * col_bytes) as u64;
                    enc.set_buffer(0, Some(out), offset);
                    enc.dispatch_threads(half_grid, middle_tg);
                }
            }
        }

        // ---- Tail: last k stages fused in threadgroup memory ----
        enc.set_compute_pipeline_state(&pipeline_tail);
        enc.set_buffer(1, Some(twiddles_bitrev), 0);
        enc.set_bytes(2, mem::size_of::<u32>() as u64, void_ptr(&n_val));
        enc.set_bytes(3, mem::size_of::<u32>() as u64, void_ptr(&start_stage));
        enc.set_bytes(4, mem::size_of::<u32>() as u64, void_ptr(&k_val));
        enc.set_threadgroup_memory_length(0, tail_shmem);

        for col in 0..num_cols {
            let offset = (col as usize * col_bytes) as u64;
            enc.set_buffer(0, Some(out), offset);
            enc.dispatch_thread_groups(tail_groups, tail_tg);
        }

        enc.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
    });

    Ok(())
}

/// Placeholder for the Goldilocks Fp3 extension field Bowers-G NTT + LDE pipeline.
///
/// To be implemented in Task 12.
pub fn metal_bowers_lde_fp3(
    _cols: &Buffer,
    _twiddles_bitrev: &Buffer,
    _coset_powers: &Buffer,
    _out: &mut Buffer,
    _log_n: u32,
    _num_cols: u32,
    _state: &MetalState,
) -> Result<(), MetalError> {
    unimplemented!("filled in by Task 12")
}
