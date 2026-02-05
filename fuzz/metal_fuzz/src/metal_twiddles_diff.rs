//! Differential fuzzer for Metal twiddle generation vs CPU twiddle generation.
//!
//! This fuzzer compares Metal GPU twiddle factor generation against CPU
//! for both Stark252 and Goldilocks fields.

#[macro_use]
extern crate honggfuzz;

use lambdaworks_gpu::metal::abstractions::state::MetalState;
use lambdaworks_math::{
    fft::{cpu::roots_of_unity::get_twiddles, gpu::metal::ops::gen_twiddles},
    field::{
        fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
        fields::u64_goldilocks_field::Goldilocks64Field, traits::RootsConfig,
    },
};

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
        fuzz!(|data: (u8, u8)| {
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

                let metal_twiddles =
                    match gen_twiddles::<Stark252PrimeField>(order, config, &metal_state) {
                        Ok(tw) => tw,
                        Err(e) => {
                            // CPU succeeded but Metal failed - this is a correctness bug
                            // Panicking is intentional here to signal the fuzzer found an issue
                            assert!(
                                false,
                                "Metal Stark252 twiddle gen failed for order={}, config={:?}: {:?}",
                                order, config, e
                            );
                            return;
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

                let metal_twiddles =
                    match gen_twiddles::<Goldilocks64Field>(order, config, &metal_state) {
                        Ok(tw) => tw,
                        Err(e) => {
                            // CPU succeeded but Metal failed - this is a correctness bug
                            assert!(
                                false,
                                "Metal Goldilocks twiddle gen failed for order={}, config={:?}: {:?}",
                                order, config, e
                            );
                            return;
                        }
                    };

                assert_eq!(
                    cpu_twiddles, metal_twiddles,
                    "Goldilocks twiddles mismatch for order={}, config={:?}",
                    order, config
                );
            }
        });
    }
}
