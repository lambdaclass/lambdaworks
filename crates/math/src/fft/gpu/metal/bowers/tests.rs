//! Unit tests for the Bowers-G Metal pipeline.

use super::dispatcher::{metal_bowers_fft_no_coset, metal_bowers_lde};
use super::twiddles::{cached_bowers_twiddles_goldilocks, cached_coset_powers_goldilocks};
use crate::fft::cpu::ntt_bowers_goldilocks::{compute_bowers_twiddles, ntt_bowers, P};
use crate::field::element::FieldElement;
use crate::field::fields::u64_goldilocks_field::Goldilocks64Field;
use crate::polynomial::Polynomial;
use lambdaworks_gpu::metal::abstractions::state::MetalState;

type Fp = FieldElement<Goldilocks64Field>;

/// Bit-reverse a slice in place (log2(len) bits).
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

/// The GPU `metal_bowers_lde` bit-reverses its input internally and produces
/// natural-order output. With an all-ones coset this is a plain forward NTT,
/// which must equal the CPU reference `ntt_bowers` (bit-reverse + butterflies).
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

    let cpu_tw = compute_bowers_twiddles(n, 7);
    let mut cpu_out = input.clone();
    ntt_bowers(&mut cpu_out, &cpu_tw);

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

/// End-to-end coset LDE: `metal_bowers_lde` must reproduce the trusted
/// `Polynomial::evaluate_offset_fft` (coset evaluations in natural order).
///
/// The GPU consumes natural-order coefficients and a *bit-reversed* coset
/// powers buffer; it bit-reverses the coefficients internally, folds the
/// coset on load, and outputs `DFT(coeffs * coset)` in natural order.
fn run_metal_bowers_lde_vs_cpu(log_n: u32, coset_offset: u64) {
    let state = MetalState::new(None).expect("metal");
    let n = 1usize << log_n;
    let g = Fp::from(coset_offset);

    let coeffs: Vec<Fp> = (0..n as u64)
        .map(|i| Fp::from(i.wrapping_mul(0x9E37_79B9_7F4A_7C15) % P))
        .collect();
    let coeffs_u64: Vec<u64> = coeffs.iter().map(|x| *x.value()).collect();

    let twiddles = cached_bowers_twiddles_goldilocks(log_n);
    let tw_u64: Vec<u64> = twiddles.iter().map(|x| *x.value()).collect();

    // Coset powers must be supplied bit-reversed (see dispatcher docs).
    let coset = cached_coset_powers_goldilocks(log_n, g);
    let mut coset_u64: Vec<u64> = coset.iter().map(|x| *x.value()).collect();
    bit_reverse(&mut coset_u64);

    let in_buf = state.alloc_buffer_data(&coeffs_u64);
    let tw_buf = state.alloc_buffer_data(&tw_u64);
    let coset_buf = state.alloc_buffer_data(&coset_u64);
    let mut out_buf = state.alloc_buffer::<u64>(n);

    metal_bowers_lde(&in_buf, &tw_buf, &coset_buf, &mut out_buf, log_n, 1, &state).unwrap();
    let gpu_out: Vec<u64> = MetalState::retrieve_contents(&out_buf);

    let poly = Polynomial::new(&coeffs);
    let cpu_evals = Polynomial::evaluate_offset_fft::<Goldilocks64Field>(&poly, 1, None, &g)
        .expect("evaluate_offset_fft");

    assert_eq!(gpu_out.len(), cpu_evals.len());
    for i in 0..n {
        assert_eq!(
            gpu_out[i],
            *cpu_evals[i].value(),
            "coset LDE mismatch at i={i}; log_n={log_n}"
        );
    }
}

#[test]
fn metal_bowers_lde_matches_cpu_log4() {
    run_metal_bowers_lde_vs_cpu(4, 3);
}

#[test]
fn metal_bowers_lde_matches_cpu_log10() {
    run_metal_bowers_lde_vs_cpu(10, 3);
}

#[test]
fn metal_bowers_lde_matches_cpu_log16() {
    run_metal_bowers_lde_vs_cpu(16, 7);
}
