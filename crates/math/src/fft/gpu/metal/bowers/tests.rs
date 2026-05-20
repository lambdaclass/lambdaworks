//! Unit tests for the Bowers-G Metal pipeline.

use super::dispatcher::metal_bowers_fft_no_coset;
use super::twiddles::cached_bowers_twiddles_goldilocks;
use crate::fft::cpu::ntt_bowers_goldilocks::{compute_bowers_twiddles, ntt_bowers_butterflies, P};
use lambdaworks_gpu::metal::abstractions::state::MetalState;

/// Compares the GPU Bowers butterfly network against the CPU reference.
///
/// Both run the butterfly schedule directly on natural-order input (no
/// bit-reverse on either side), so the outputs must be element-for-element
/// equal. The GPU uses correct field arithmetic; the CPU reference uses raw
/// u64 Goldilocks arithmetic — they must agree.
fn run_metal_bowers_vs_cpu(log_n: u32) {
    let state = MetalState::new(None).expect("metal");
    let n = 1usize << log_n;

    let twiddles = cached_bowers_twiddles_goldilocks(log_n);
    let tw_u64: Vec<u64> = twiddles.iter().map(|x| *x.value()).collect();
    let tw_buf = state.alloc_buffer_data(&tw_u64);

    let input: Vec<u64> = (0..n as u64)
        .map(|i| i.wrapping_mul(0x9E37_79B9_7F4A_7C15) % P)
        .collect();
    let in_buf = state.alloc_buffer_data(&input);
    let mut out_buf = state.alloc_buffer::<u64>(n);

    metal_bowers_fft_no_coset(&in_buf, &tw_buf, &mut out_buf, log_n, 1, &state).unwrap();
    let gpu_out: Vec<u64> = MetalState::retrieve_contents(&out_buf);

    // CPU reference: butterfly network only (no bit-reverse), same as the GPU.
    let cpu_tw = compute_bowers_twiddles(n, 7);
    let mut cpu_out = input.clone();
    ntt_bowers_butterflies(&mut cpu_out, &cpu_tw);

    assert_eq!(gpu_out.len(), cpu_out.len());
    for i in 0..n {
        assert_eq!(gpu_out[i], cpu_out[i], "mismatch at i={i}; log_n={log_n}");
    }
}

#[test]
fn metal_bowers_matches_cpu_log4() {
    run_metal_bowers_vs_cpu(4);
}

#[test]
fn metal_bowers_matches_cpu_log10() {
    run_metal_bowers_vs_cpu(10);
}

#[test]
fn metal_bowers_matches_cpu_log16() {
    run_metal_bowers_vs_cpu(16);
}

#[test]
fn metal_bowers_matches_cpu_log18() {
    run_metal_bowers_vs_cpu(18);
}

#[test]
fn metal_bowers_matches_cpu_log20() {
    run_metal_bowers_vs_cpu(20);
}
