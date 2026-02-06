//! Differential fuzzer for Metal twiddle generation vs CPU twiddle generation.
//!
//! This fuzzer compares Metal GPU twiddle factor generation against CPU
//! for both Stark252 and Goldilocks fields.

#![no_main]

use libfuzzer_sys::fuzz_target;
use std::sync::LazyLock;

use lambdaworks_gpu::metal::abstractions::state::MetalState;
use lambdaworks_math::{
    fft::{cpu::roots_of_unity::get_twiddles, gpu::metal::ops::gen_twiddles},
    field::{
        fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
        fields::u64_goldilocks_field::Goldilocks64Field, traits::RootsConfig,
    },
};

static METAL_STATE: LazyLock<Option<MetalState>> = LazyLock::new(|| MetalState::new(None).ok());

fuzz_target!(|data: (u8, u8)| {
    let Some(metal_state) = METAL_STATE.as_ref() else {
        return;
    };

    let (order_byte, config_byte) = data;

    // Limit order to reasonable range (1-20 for testing)
    let order = ((order_byte % 20) + 1) as u64;

    // Select configuration
    let config = match config_byte % 4 {
        0 => RootsConfig::Natural,
        1 => RootsConfig::NaturalInversed,
        2 => RootsConfig::BitReverse,
        _ => RootsConfig::BitReverseInversed,
    };

    // Test Stark252 twiddles
    {
        let cpu_twiddles = match get_twiddles::<Stark252PrimeField>(order, config) {
            Ok(tw) => tw,
            Err(_) => return,
        };

        let metal_twiddles = match gen_twiddles::<Stark252PrimeField>(order, config, metal_state) {
            Ok(tw) => tw,
            Err(e) => {
                panic!(
                    "Metal Stark252 twiddle gen failed for order={}, config={:?}: {:?}",
                    order, config, e
                );
            }
        };

        assert_eq!(
            cpu_twiddles, metal_twiddles,
            "Stark252 twiddles mismatch for order={}, config={:?}",
            order, config
        );
    }

    // Test Goldilocks twiddles (only if order <= 32 due to two-adicity)
    if order <= 32 {
        let cpu_twiddles = match get_twiddles::<Goldilocks64Field>(order, config) {
            Ok(tw) => tw,
            Err(_) => return,
        };

        let metal_twiddles = match gen_twiddles::<Goldilocks64Field>(order, config, metal_state) {
            Ok(tw) => tw,
            Err(e) => {
                panic!(
                    "Metal Goldilocks twiddle gen failed for order={}, config={:?}: {:?}",
                    order, config, e
                );
            }
        };

        assert_eq!(
            cpu_twiddles, metal_twiddles,
            "Goldilocks twiddles mismatch for order={}, config={:?}",
            order, config
        );
    }
});
