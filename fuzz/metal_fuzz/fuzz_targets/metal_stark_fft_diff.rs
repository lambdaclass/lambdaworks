//! Differential fuzzer: GPU STARK FFT interpolation vs CPU FFT interpolation.
#![no_main]

use libfuzzer_sys::fuzz_target;
use std::sync::LazyLock;

use lambdaworks_math::{
    field::{element::FieldElement, fields::u64_goldilocks_field::Goldilocks64Field},
    polynomial::Polynomial,
};
use lambdaworks_stark_gpu::metal::state::StarkMetalState;
use lambdaworks_stark_gpu::metal::fft::gpu_interpolate_fft;

type F = Goldilocks64Field;
type FpE = FieldElement<F>;

static STATE: LazyLock<Option<StarkMetalState>> = LazyLock::new(|| StarkMetalState::new().ok());

fuzz_target!(|data: Vec<u64>| {
    let Some(state) = STATE.as_ref() else {
        return;
    };

    let mut input_raw = data;
    if input_raw.is_empty() {
        input_raw.push(1u64);
    }

    // Ensure power-of-2 length, cap at 2^16 to keep fuzzing fast
    let len = input_raw.len().next_power_of_two().min(1 << 16);
    input_raw.resize(len, 0u64);

    let inputs: Vec<FpE> = input_raw.iter().map(|&v| FpE::from(v)).collect();

    // CPU interpolation
    let cpu_poly = match Polynomial::interpolate_fft::<F>(&inputs) {
        Ok(p) => p,
        Err(_) => return,
    };

    // GPU interpolation
    let gpu_coeffs = match gpu_interpolate_fft::<F>(&inputs, state.inner()) {
        Ok(c) => c,
        Err(_) => panic!("GPU FFT failed but CPU succeeded for len {}", inputs.len()),
    };

    assert_eq!(
        cpu_poly.coefficients().len(),
        gpu_coeffs.len(),
        "coefficient count mismatch for len {}",
        inputs.len()
    );
    for (i, (cpu, gpu)) in cpu_poly.coefficients().iter().zip(&gpu_coeffs).enumerate() {
        assert_eq!(cpu, gpu, "coefficient {} differs for len {}", i, inputs.len());
    }
});
