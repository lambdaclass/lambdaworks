//! Differential fuzzer for Metal CFFT vs CPU CFFT on Mersenne31 field.
//!
//! Compares Metal GPU circle FFT results against CPU circle FFT results
//! to ensure the Metal implementation is correct for the Mersenne31 prime.

#![no_main]

use libfuzzer_sys::fuzz_target;
use std::sync::LazyLock;

use lambdaworks_gpu::metal::abstractions::state::MetalState;
use lambdaworks_math::{
    circle::{
        cosets::Coset,
        gpu::metal::ops::cfft_gpu,
        cfft::cfft,
        twiddles::{get_twiddles, TwiddlesConfig},
    },
    field::{element::FieldElement, fields::mersenne31::field::Mersenne31Field},
};

type FE = FieldElement<Mersenne31Field>;

static METAL_STATE: LazyLock<Option<MetalState>> = LazyLock::new(|| MetalState::new(None).ok());

fuzz_target!(|data: Vec<u32>| {
    let Some(metal_state) = METAL_STATE.as_ref() else {
        return;
    };

    let mut input_raw = data;
    if input_raw.is_empty() {
        input_raw.push(1u32);
    }

    // Ensure power of 2 length (min 4 for meaningful CFFT)
    let len = input_raw.len().next_power_of_two().max(4);
    input_raw.resize(len, 0u32);

    // Mask to valid Mersenne31 range and create field elements
    let inputs: Vec<FE> = input_raw
        .iter()
        .map(|&v| FE::from(&(v & 0x7FFFFFFF)))
        .collect();

    // Generate evaluation twiddles
    let log_2_size = inputs.len().trailing_zeros();
    let coset = Coset::new_standard(log_2_size);
    let twiddles = get_twiddles(coset, TwiddlesConfig::Evaluation);

    // Run CPU CFFT
    let mut cpu_input = inputs.clone();
    cfft(&mut cpu_input, &twiddles);

    // Run Metal CFFT
    let metal_result = match cfft_gpu(&inputs, &twiddles, metal_state) {
        Ok(r) => r,
        Err(e) => {
            panic!(
                "Metal CFFT failed but CPU succeeded for input len {}: {:?}",
                inputs.len(),
                e
            );
        }
    };

    // Compare results
    assert_eq!(
        cpu_input.to_vec(),
        metal_result,
        "Metal CFFT result differs from CPU CFFT for input len {}",
        inputs.len()
    );
});
