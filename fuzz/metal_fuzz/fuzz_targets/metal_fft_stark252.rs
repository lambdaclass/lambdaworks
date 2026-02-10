//! Differential fuzzer for Metal FFT vs CPU FFT on Stark252 field.
//!
//! This fuzzer compares Metal GPU FFT results against CPU FFT results
//! to ensure the Metal implementation is correct.

#![no_main]

use libfuzzer_sys::fuzz_target;
use std::sync::LazyLock;

use lambdaworks_gpu::metal::abstractions::state::MetalState;
use lambdaworks_math::{
    fft::{
        cpu::{ops::fft as fft_cpu, roots_of_unity::get_twiddles},
        gpu::metal::ops::fft as fft_metal,
    },
    field::{
        element::FieldElement, fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
        traits::RootsConfig,
    },
};

type F = Stark252PrimeField;
type FE = FieldElement<F>;

static METAL_STATE: LazyLock<Option<MetalState>> = LazyLock::new(|| MetalState::new(None).ok());

fuzz_target!(|data: Vec<u64>| {
    let Some(metal_state) = METAL_STATE.as_ref() else {
        return;
    };

    // Convert raw bytes to field elements
    let mut input_raw = data;
    if input_raw.is_empty() {
        input_raw.push(1u64);
    }

    // Ensure power of 2 length
    let len = input_raw.len().next_power_of_two();
    input_raw.resize(len, 0u64);

    let inputs: Vec<FE> = input_raw
        .iter()
        .map(|&v| {
            let hex = format!("{:x}", v);
            FE::from_hex_unchecked(&hex)
        })
        .collect();

    // Get twiddle factors
    let order = inputs.len().trailing_zeros() as u64;
    let twiddles = match get_twiddles::<F>(order, RootsConfig::BitReverse) {
        Ok(tw) => tw,
        Err(_) => return,
    };

    // Run CPU FFT
    let cpu_result = match fft_cpu(&inputs, &twiddles) {
        Ok(r) => r,
        Err(_) => return,
    };

    // Run Metal FFT
    let metal_result = match fft_metal(&inputs, &twiddles, metal_state) {
        Ok(r) => r,
        Err(e) => {
            panic!(
                "Metal FFT failed but CPU succeeded for input len {}: {:?}",
                inputs.len(),
                e
            );
        }
    };

    // Compare results
    assert_eq!(
        cpu_result,
        metal_result,
        "Metal FFT result differs from CPU FFT for input len {}",
        inputs.len()
    );
});
