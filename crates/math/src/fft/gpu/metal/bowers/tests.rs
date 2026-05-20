//! Unit tests for the Bowers-G Metal pipeline.

use super::dispatcher::{metal_bowers_fft_no_coset, metal_bowers_lde, metal_bowers_lde_fp3};
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

/// The pipeline must be deterministic: identical inputs produce byte-equal
/// output across independently constructed `MetalState`s.
#[test]
fn metal_bowers_lde_deterministic_across_states() {
    let log_n = 12u32;
    let n = 1usize << log_n;
    let g = Fp::from(5u64);

    let coeffs: Vec<u64> = (0..n as u64)
        .map(|i| i.wrapping_mul(0xDEAD_BEEF_CAFE_F00D) % P)
        .collect();
    let twiddles = cached_bowers_twiddles_goldilocks(log_n);
    let tw_u64: Vec<u64> = twiddles.iter().map(|x| *x.value()).collect();
    let coset = cached_coset_powers_goldilocks(log_n, g);
    let mut coset_u64: Vec<u64> = coset.iter().map(|x| *x.value()).collect();
    bit_reverse(&mut coset_u64);

    let mut outs: Vec<Vec<u64>> = Vec::new();
    for _ in 0..2 {
        let state = MetalState::new(None).expect("metal");
        let in_buf = state.alloc_buffer_data(&coeffs);
        let tw_buf = state.alloc_buffer_data(&tw_u64);
        let coset_buf = state.alloc_buffer_data(&coset_u64);
        let mut out_buf = state.alloc_buffer::<u64>(n);
        metal_bowers_lde(&in_buf, &tw_buf, &coset_buf, &mut out_buf, log_n, 1, &state).unwrap();
        outs.push(MetalState::retrieve_contents(&out_buf));
    }
    assert_eq!(outs[0], outs[1]);
}

/// Batched dispatch: 64 independent columns processed in one call must each
/// match the per-column `evaluate_offset_fft`.
#[test]
fn metal_bowers_lde_multicol_64() {
    let state = MetalState::new(None).expect("metal");
    let log_n = 10u32;
    let n = 1usize << log_n;
    let num_cols = 64u32;
    let g = Fp::from(7u64);

    let twiddles = cached_bowers_twiddles_goldilocks(log_n);
    let tw_u64: Vec<u64> = twiddles.iter().map(|x| *x.value()).collect();
    let coset = cached_coset_powers_goldilocks(log_n, g);
    let mut coset_u64: Vec<u64> = coset.iter().map(|x| *x.value()).collect();
    bit_reverse(&mut coset_u64);

    let mut all_in: Vec<u64> = Vec::with_capacity(n * num_cols as usize);
    let mut cpu_outs: Vec<Vec<Fp>> = Vec::with_capacity(num_cols as usize);
    for c in 0..num_cols {
        let coeffs: Vec<Fp> = (0..n as u64)
            .map(|i| Fp::from((i + c as u64 * 7919).wrapping_mul(0x9E37_79B9_7F4A_7C15) % P))
            .collect();
        all_in.extend(coeffs.iter().map(|x| *x.value()));
        let poly = Polynomial::new(&coeffs);
        cpu_outs.push(
            Polynomial::evaluate_offset_fft::<Goldilocks64Field>(&poly, 1, None, &g)
                .expect("evaluate_offset_fft"),
        );
    }

    let in_buf = state.alloc_buffer_data(&all_in);
    let tw_buf = state.alloc_buffer_data(&tw_u64);
    let coset_buf = state.alloc_buffer_data(&coset_u64);
    let mut out_buf = state.alloc_buffer::<u64>(n * num_cols as usize);
    metal_bowers_lde(
        &in_buf,
        &tw_buf,
        &coset_buf,
        &mut out_buf,
        log_n,
        num_cols,
        &state,
    )
    .unwrap();
    let gpu_out: Vec<u64> = MetalState::retrieve_contents(&out_buf);

    for c in 0..num_cols as usize {
        for i in 0..n {
            assert_eq!(
                gpu_out[c * n + i],
                *cpu_outs[c][i].value(),
                "multicol mismatch at col={c} i={i}"
            );
        }
    }
}

/// Fp3 golden vectors: the GPU extension-field Bowers NTT must match the CPU
/// reference `ntt_bowers_fp3` element-for-element. With an all-ones coset this
/// is a plain forward NTT over Fp3.
fn run_metal_bowers_fp3_vs_cpu(log_n: u32) {
    use crate::fft::cpu::ntt_bowers_fp3::ntt_bowers_fp3;
    use crate::field::fields::u64_goldilocks_field::Degree3GoldilocksExtensionField;
    use crate::field::traits::IsPrimeField;

    type Fp3 = FieldElement<Degree3GoldilocksExtensionField>;

    let state = MetalState::new(None).expect("metal");
    let n = 1usize << log_n;

    // Base-field twiddles, shared with the Goldilocks path.
    let twiddles = cached_bowers_twiddles_goldilocks(log_n);
    let tw_u64: Vec<u64> = twiddles.iter().map(|x| *x.value()).collect();
    let tw_buf = state.alloc_buffer_data(&tw_u64);

    // Identity coset (all ones), base-field, bit-reversed (still all ones).
    let ones: Vec<u64> = vec![1u64; n];
    let coset_buf = state.alloc_buffer_data(&ones);

    // Fp3 input with all three limbs populated.
    let coeffs: Vec<Fp3> = (0..n as u64)
        .map(|i| {
            let c0 = Fp::from(i.wrapping_mul(0x9E37_79B9_7F4A_7C15) % P);
            let c1 = Fp::from(i.wrapping_mul(0xC2B2_AE3D_27D4_EB4F) % P);
            let c2 = Fp::from(i.wrapping_mul(0x1656_67B1_9E37_79F9) % P);
            FieldElement::new([c0, c1, c2])
        })
        .collect();

    // Pack Fp3 -> [c0, c1, c2] u64 triples for the GPU buffer.
    let mut packed: Vec<u64> = Vec::with_capacity(n * 3);
    for e in &coeffs {
        let limbs = e.value();
        packed.push(*limbs[0].value());
        packed.push(*limbs[1].value());
        packed.push(*limbs[2].value());
    }
    let in_buf = state.alloc_buffer_data(&packed);
    let mut out_buf = state.alloc_buffer::<u64>(n * 3);

    metal_bowers_lde_fp3(&in_buf, &tw_buf, &coset_buf, &mut out_buf, log_n, 1, &state).unwrap();
    let gpu_out: Vec<u64> = MetalState::retrieve_contents(&out_buf);

    // CPU reference: bit-reverse + butterflies over Fp3.
    let cpu_tw: Vec<Fp> = compute_bowers_twiddles(n, 7)
        .iter()
        .map(|&x| Fp::from(x))
        .collect();
    let mut cpu_data = coeffs.clone();
    ntt_bowers_fp3(&mut cpu_data, &cpu_tw);

    for i in 0..n {
        let cpu_limbs = cpu_data[i].value();
        for limb in 0..3 {
            let gpu_v = Goldilocks64Field::canonical(&gpu_out[i * 3 + limb]);
            let cpu_v = Goldilocks64Field::canonical(cpu_limbs[limb].value());
            assert_eq!(
                gpu_v, cpu_v,
                "Fp3 mismatch at i={i} limb={limb}; log_n={log_n}"
            );
        }
    }
}

#[test]
fn metal_bowers_fp3_matches_cpu_log4() {
    run_metal_bowers_fp3_vs_cpu(4);
}

#[test]
fn metal_bowers_fp3_matches_cpu_log10() {
    run_metal_bowers_fp3_vs_cpu(10);
}

#[test]
fn metal_bowers_fp3_matches_cpu_log16() {
    run_metal_bowers_fp3_vs_cpu(16);
}
