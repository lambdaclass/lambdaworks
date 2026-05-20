//! Rust dispatcher for the Bowers-G fused LDE Metal kernels.
//!
//! Stage convention matches the CPU reference `ntt_bowers_butterflies`:
//!   stage 0   -> smallest stride (half = 1)
//!   stage k   -> half = 2^k
//!   stage log_n - 1 -> largest stride (half = n/2)
//!
//! Pipeline:
//!   - Fused HEAD kernel covers the first K stages (smallest strides) in
//!     threadgroup memory, applying the coset multiply on load.
//!   - Per-stage MIDDLE dispatches handle stages K..log_n.
//!
//! No bit-reverse pass: DIF butterflies on natural-order input produce
//! bit-reversed-order output (intentional; the Merkle leaf indexer absorbs
//! the reordering via `LeafOrder::Bitrev`).
//!
//! If `K == 0` (very small log_n), the dispatcher falls back to a separate
//! stage-0 dispatch (coset + butterfly with per-block twiddle) followed by
//! middle stages.

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

/// Computes the optimal number of fused head stages.
///
/// Capped at `log_n - 1` so at least one middle dispatch remains (covering the
/// largest-stride stage).
fn optimal_fused_head_stages(log_n: u32) -> u32 {
    let max_block = FFT_TG_MEM_BUDGET / GOLDILOCKS_ELEM_SIZE;
    let max_fused = max_block.ilog2(); // 12 for 32 KB / 8 bytes
    max_fused
        .min(log_n.saturating_sub(1))
        .min(FFT_MAX_FUSED_STAGES)
}

/// Execute the Bowers-G NTT + fused LDE pipeline on the GPU for Goldilocks base field.
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

    let k = optimal_fused_head_stages(log_n);
    let k_val: u32 = k;

    let pipeline_middle = state.get_pipeline("bowers_lde_middle_Goldilocks")?;
    let middle_tg = MTLSize::new(pipeline_middle.thread_execution_width(), 1, 1);
    let half_grid = MTLSize::new(n as u64 / 2, 1, 1);

    objc::rc::autoreleasepool(|| -> Result<(), MetalError> {
        let command_buffer = state.queue.new_command_buffer();

        // Blit cols -> out so all NTT stages operate in-place on out.
        {
            let blit = command_buffer.new_blit_command_encoder();
            blit.copy_from_buffer(cols, 0, out, 0, cols.length());
            blit.end_encoding();
        }

        let enc = command_buffer.new_compute_command_encoder();

        // ---- HEAD: stages 0..k in threadgroup memory (coset fold on load) ----
        if k > 0 {
            let pipeline_head = state.get_pipeline("bowers_lde_head_Goldilocks")?;
            let head_block_size = 1u32 << k;
            let head_groups = MTLSize::new(n as u64 / head_block_size as u64, 1, 1);
            let head_threads = 256u64
                .min(pipeline_head.max_total_threads_per_threadgroup())
                .min((head_block_size as u64 / 2).max(1));
            let head_tg = MTLSize::new(head_threads, 1, 1);
            let head_shmem = (head_block_size as u64) * (GOLDILOCKS_ELEM_SIZE as u64);

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
        } else {
            // k == 0 path: standalone stage0 kernel (coset + stage-0 butterfly).
            let pipeline_stage0 = state.get_pipeline("bowers_lde_stage0_Goldilocks")?;
            let stage0_tg = MTLSize::new(pipeline_stage0.thread_execution_width(), 1, 1);

            enc.set_compute_pipeline_state(&pipeline_stage0);
            enc.set_buffer(1, Some(twiddles_bitrev), 0);
            enc.set_buffer(2, Some(coset_powers), 0);

            for col in 0..num_cols {
                let offset = (col as usize * col_bytes) as u64;
                enc.set_buffer(0, Some(out), offset);
                enc.dispatch_threads(half_grid, stage0_tg);
            }
        }

        // ---- MIDDLE stages: k..log_n (or 1..log_n if k==0 path was used) ----
        let middle_start = if k == 0 { 1 } else { k };
        if middle_start < log_n {
            enc.set_compute_pipeline_state(&pipeline_middle);
            enc.set_buffer(1, Some(twiddles_bitrev), 0);

            for stage in middle_start..log_n {
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
