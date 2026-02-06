//! Differential fuzzer for Metal FFT vs CPU FFT on Goldilocks 64-bit field.
//!
//! This fuzzer compares Metal GPU FFT results against CPU FFT results
//! to ensure the Metal implementation is correct for the Goldilocks prime.

#[macro_use]
extern crate honggfuzz;

use lambdaworks_gpu::metal::abstractions::state::MetalState;
use lambdaworks_math::{
    fft::{
        cpu::{ops::fft as fft_cpu, roots_of_unity::get_twiddles},
        gpu::metal::ops::fft as fft_metal,
    },
    field::{
        element::FieldElement, fields::u64_goldilocks_field::Goldilocks64Field,
        traits::RootsConfig,
    },
};

type F = Goldilocks64Field;
type FE = FieldElement<F>;

fn main() {
    // Initialize Metal state once outside the loop.
    // Skip the entire fuzzer if Metal is not available.
    let metal_state = match MetalState::new(None) {
        Ok(state) => state,
        Err(e) => {
            eprintln!("Metal device not available, skipping fuzzer: {:?}", e);
            return;
        }
    };

    loop {
        fuzz!(|data: Vec<u64>| {
            // Convert raw bytes to field elements
            let mut input_raw = data;
            if input_raw.is_empty() {
                input_raw.push(1u64);
            }

            // Ensure power of 2 length
            let len = input_raw.len().next_power_of_two();
            input_raw.resize(len, 0u64);

            // Create field elements directly (Goldilocks uses u64)
            let inputs: Vec<FE> = input_raw.iter().map(|&v| FE::from(v)).collect();

            // Get twiddle factors
            let order = inputs.len().trailing_zeros() as u64;

            // Goldilocks has TWO_ADICITY = 32, so order must be <= 32
            if order > 32 {
                return;
            }

            let twiddles = match get_twiddles::<F>(order, RootsConfig::BitReverse) {
                Ok(tw) => tw,
                Err(_) => return, // Skip if order not supported
            };

            // Run CPU FFT
            let cpu_result = match fft_cpu(&inputs, &twiddles) {
                Ok(r) => r,
                Err(_) => return, // Skip if CPU FFT fails
            };

            // Run Metal FFT
            let metal_result = match fft_metal(&inputs, &twiddles, &metal_state) {
                Ok(r) => r,
                Err(e) => {
                    // Metal FFT failed but CPU succeeded - this is a correctness bug
                    assert!(
                        false,
                        "Metal Goldilocks FFT failed but CPU succeeded for input len {}: {:?}",
                        inputs.len(),
                        e
                    );
                    return;
                }
            };

            // Compare results
            assert_eq!(
                cpu_result, metal_result,
                "Metal Goldilocks FFT result differs from CPU FFT for input len {}",
                inputs.len()
            );
        });
    }
}
