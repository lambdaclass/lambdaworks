//! Differential fuzzer: GPU STARK LDE (evaluate_offset_fft) vs CPU.
#![no_main]

use libfuzzer_sys::fuzz_target;
use std::sync::LazyLock;

use lambdaworks_math::{
    field::{element::FieldElement, fields::u64_goldilocks_field::Goldilocks64Field},
    polynomial::Polynomial,
};
use lambdaworks_stark_gpu::metal::state::StarkMetalState;
use lambdaworks_stark_gpu::metal::fft::gpu_evaluate_offset_fft;

type F = Goldilocks64Field;
type FpE = FieldElement<F>;

static STATE: LazyLock<Option<StarkMetalState>> = LazyLock::new(|| StarkMetalState::new().ok());

fuzz_target!(|data: (Vec<u64>, u64)| {
    let Some(state) = STATE.as_ref() else {
        return;
    };

    let (coeffs_raw, offset_raw) = data;
    if coeffs_raw.is_empty() {
        return;
    }

    // Power-of-2 length, cap at 2^14 for speed
    let len = coeffs_raw.len().next_power_of_two().min(1 << 14);
    let mut padded = coeffs_raw;
    padded.resize(len, 0u64);

    let coeffs: Vec<FpE> = padded.iter().map(|&v| FpE::from(v)).collect();
    let offset = FpE::from(offset_raw.max(1)); // non-zero offset
    let blowup_factor = 4; // standard STARK blowup

    // Goldilocks two-adicity = 32; domain_size = len * blowup_factor
    let domain_order = (len * blowup_factor).trailing_zeros();
    if domain_order > 32 {
        return;
    }

    let poly = Polynomial::new(&coeffs);

    // CPU LDE
    let cpu_evals = match Polynomial::evaluate_offset_fft::<F>(&poly, blowup_factor, None, &offset) {
        Ok(e) => e,
        Err(_) => return,
    };

    // GPU LDE
    let gpu_evals = match gpu_evaluate_offset_fft::<F>(&coeffs, blowup_factor, &offset, state.inner()) {
        Ok(e) => e,
        Err(_) => panic!("GPU LDE failed but CPU succeeded for len {}", coeffs.len()),
    };

    assert_eq!(cpu_evals.len(), gpu_evals.len(), "LDE eval count mismatch for len {}", coeffs.len());
    for (i, (cpu, gpu)) in cpu_evals.iter().zip(&gpu_evals).enumerate() {
        assert_eq!(cpu, gpu, "LDE eval {} differs for len {}", i, coeffs.len());
    }
});
