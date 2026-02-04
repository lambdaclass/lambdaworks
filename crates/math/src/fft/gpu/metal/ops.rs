//! Metal FFT operations for lambdaworks.
//!
//! This module provides GPU-accelerated FFT operations using Apple's Metal framework.
//! The implementation mirrors the CPU FFT API but executes on the GPU for better
//! performance on large inputs.
//!
//! # Inspiration
//!
//! This implementation is based on:
//! - Original lambdaworks Metal implementation (pre-PR#993)
//! - ICICLE's multi-backend GPU architecture
//! - VkFFT's efficient FFT patterns

use crate::field::{
    element::FieldElement,
    traits::{IsFFTField, IsField, IsSubFieldOf, RootsConfig},
};
use lambdaworks_gpu::metal::abstractions::{errors::MetalError, state::*};

use metal::MTLSize;

use core::mem;

/// Executes parallel ordered FFT over a slice of field elements using Metal GPU.
///
/// "Ordered" means that the input is in natural order, and the output will be
/// in natural order too. Twiddle factors must be in bit-reverse order.
///
/// # Type Parameters
///
/// - `F`: The FFT-compatible field (provides primitive roots of unity)
/// - `E`: The field of input elements (can be an extension of F)
///
/// # Arguments
///
/// - `input`: Slice of field elements to transform (length must be power of 2)
/// - `twiddles`: Pre-computed twiddle factors in bit-reverse order
/// - `state`: Metal state containing device and shader library
///
/// # Errors
///
/// Returns `MetalError::InputError` if input length is not a power of two.
/// Returns `MetalError::PipelineError` if kernel setup fails.
pub fn fft<F, E>(
    input: &[FieldElement<E>],
    twiddles: &[FieldElement<F>],
    state: &MetalState,
) -> Result<Vec<FieldElement<E>>, MetalError>
where
    F: IsFFTField + IsSubFieldOf<E>,
    E: IsField,
{
    if !input.len().is_power_of_two() {
        return Err(MetalError::InputError(input.len()));
    }

    let pipeline = state.setup_pipeline(&format!("radix2_dit_butterfly_{}", F::field_name()))?;

    let input_buffer = state.alloc_buffer_data(input);
    let twiddles_buffer = state.alloc_buffer_data(twiddles);

    objc::rc::autoreleasepool(|| {
        let (command_buffer, command_encoder) = state.setup_command(
            &pipeline,
            Some(&[(0, &input_buffer), (1, &twiddles_buffer)]),
        );

        let order = input.len().trailing_zeros();
        for stage in 0..order {
            command_encoder.set_bytes(2, mem::size_of_val(&stage) as u64, void_ptr(&stage));

            let grid_size = MTLSize::new(input.len() as u64 / 2, 1, 1);
            let threadgroup_size = MTLSize::new(pipeline.thread_execution_width(), 1, 1);

            command_encoder.dispatch_threads(grid_size, threadgroup_size);
        }
        command_encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();
    });

    let result = MetalState::retrieve_contents(&input_buffer);
    let result = bitrev_permutation::<F, _>(&result, state)?;
    Ok(result.into_iter().map(FieldElement::from_raw).collect())
}

/// Generates twiddle factors in parallel on the GPU.
///
/// Generates 2^{order-1} twiddle factors with the specified configuration.
///
/// # Arguments
///
/// - `order`: Log2 of the FFT size (generates 2^order / 2 twiddles)
/// - `config`: Whether to generate natural/bit-reversed and/or inverted twiddles
/// - `state`: Metal state containing device and shader library
///
/// # Errors
///
/// Returns `MetalError::FunctionError` if order > 63.
/// Returns `MetalError::PipelineError` if kernel setup fails.
pub fn gen_twiddles<F: IsFFTField>(
    order: u64,
    config: RootsConfig,
    state: &MetalState,
) -> Result<Vec<FieldElement<F>>, MetalError> {
    if order > 63 {
        return Err(MetalError::FunctionError(
            "Order should be less than or equal to 63".to_string(),
        ));
    }

    let len = (1 << order) / 2;

    let kernel = match config {
        RootsConfig::Natural => format!("calc_twiddles_{}", F::field_name()),
        RootsConfig::NaturalInversed => format!("calc_twiddles_inv_{}", F::field_name()),
        RootsConfig::BitReverse => format!("calc_twiddles_bitrev_{}", F::field_name()),
        RootsConfig::BitReverseInversed => {
            format!("calc_twiddles_bitrev_inv_{}", F::field_name())
        }
    };

    let pipeline = state.setup_pipeline(&kernel)?;

    let result_buffer = state.alloc_buffer::<F::BaseType>(len);

    objc::rc::autoreleasepool(|| {
        let (command_buffer, command_encoder) =
            state.setup_command(&pipeline, Some(&[(0, &result_buffer)]));

        let root = F::get_primitive_root_of_unity(order)
            .map_err(|_| MetalError::FunctionError(format!("No root of unity for order {}", order)))
            .expect("Failed to get primitive root of unity");

        command_encoder.set_bytes(1, mem::size_of::<F::BaseType>() as u64, void_ptr(&root));

        let grid_size = MTLSize::new(len as u64, 1, 1);
        let threadgroup_size = MTLSize::new(pipeline.max_total_threads_per_threadgroup(), 1, 1);

        command_encoder.dispatch_threads(grid_size, threadgroup_size);
        command_encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();
    });

    let result = MetalState::retrieve_contents(&result_buffer);
    Ok(result.into_iter().map(FieldElement::from_raw).collect())
}

/// Performs bit-reverse permutation on the GPU.
///
/// Reorders array elements by reversing the bit-representation of their indices.
/// This is needed to convert between natural and bit-reversed orderings in FFT.
///
/// # Arguments
///
/// - `input`: Slice of elements to permute
/// - `state`: Metal state containing device and shader library
///
/// # Errors
///
/// Returns `MetalError::PipelineError` if kernel setup fails.
pub fn bitrev_permutation<F: IsFFTField, T: Clone>(
    input: &[T],
    state: &MetalState,
) -> Result<Vec<T>, MetalError> {
    let pipeline = state.setup_pipeline(&format!("bitrev_permutation_{}", F::field_name()))?;

    let input_buffer = state.alloc_buffer_data(input);
    let result_buffer = state.alloc_buffer::<T>(input.len());

    objc::rc::autoreleasepool(|| {
        let (command_buffer, command_encoder) =
            state.setup_command(&pipeline, Some(&[(0, &input_buffer), (1, &result_buffer)]));

        let grid_size = MTLSize::new(input.len() as u64, 1, 1);
        let threadgroup_size = MTLSize::new(pipeline.max_total_threads_per_threadgroup(), 1, 1);

        command_encoder.dispatch_threads(grid_size, threadgroup_size);
        command_encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();
    });

    Ok(MetalState::retrieve_contents::<T>(&result_buffer))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fft::cpu::roots_of_unity::get_twiddles;
    use crate::field::fields::fft_friendly::stark_252_prime_field::Stark252PrimeField;
    use crate::field::fields::u64_goldilocks_field::Goldilocks64Field;
    use proptest::{collection, prelude::*};

    type StarkF = Stark252PrimeField;
    type StarkFE = FieldElement<StarkF>;

    type GoldilocksF = Goldilocks64Field;
    type GoldilocksFE = FieldElement<GoldilocksF>;

    // ==================== Stark252 Tests ====================

    prop_compose! {
        fn powers_of_two(max_exp: u8)(exp in 1..max_exp) -> usize { 1 << exp }
    }

    prop_compose! {
        fn stark_field_element()(num in any::<u64>().prop_filter("Avoid null polynomial", |x| x != &0)) -> StarkFE {
            StarkFE::from(num)
        }
    }

    fn stark_field_vec(max_exp: u8) -> impl Strategy<Value = Vec<StarkFE>> {
        powers_of_two(max_exp).prop_flat_map(|size| collection::vec(stark_field_element(), size))
    }

    proptest! {
        /// Property-based test that ensures Metal parallel FFT matches sequential CPU FFT
        /// for the Stark252 prime field.
        #[test]
        fn test_metal_fft_stark252_matches_cpu(input in stark_field_vec(6)) {
            let metal_state = MetalState::new(None).expect("Metal device required for GPU tests");
            let order = input.len().trailing_zeros();
            let twiddles = get_twiddles::<StarkF>(order.into(), RootsConfig::BitReverse)
                .expect("Stark252 field supports all power-of-two orders");

            let metal_result = fft(&input, &twiddles, &metal_state)
                .expect("Metal FFT should succeed with valid inputs");
            let cpu_result = crate::fft::cpu::ops::fft(&input, &twiddles)
                .expect("CPU FFT should succeed with valid inputs");

            prop_assert_eq!(&metal_result, &cpu_result);
        }
    }

    #[test]
    fn test_metal_fft_stark252_large_input() {
        const ORDER: usize = 16; // 2^16 = 65536 elements
        let input = vec![StarkFE::one(); 1 << ORDER];

        let metal_state = MetalState::new(None).expect("Metal device required for GPU tests");
        let order = input.len().trailing_zeros();
        let twiddles = get_twiddles::<StarkF>(order.into(), RootsConfig::BitReverse)
            .expect("Stark252 field supports order 16");

        let metal_result = fft(&input, &twiddles, &metal_state)
            .expect("Metal FFT should succeed with valid inputs");
        let cpu_result = crate::fft::cpu::ops::fft(&input, &twiddles)
            .expect("CPU FFT should succeed with valid inputs");

        assert_eq!(&metal_result, &cpu_result);
    }

    // ==================== Goldilocks Tests ====================

    prop_compose! {
        fn goldilocks_field_element()(num in any::<u64>().prop_filter("Avoid null polynomial", |x| x != &0)) -> GoldilocksFE {
            GoldilocksFE::from(num)
        }
    }

    fn goldilocks_field_vec(max_exp: u8) -> impl Strategy<Value = Vec<GoldilocksFE>> {
        powers_of_two(max_exp).prop_flat_map(|size| collection::vec(goldilocks_field_element(), size))
    }

    proptest! {
        /// Property-based test that ensures Metal parallel FFT matches sequential CPU FFT
        /// for the Goldilocks 64-bit prime field.
        #[test]
        fn test_metal_fft_goldilocks_matches_cpu(input in goldilocks_field_vec(6)) {
            let metal_state = MetalState::new(None).expect("Metal device required for GPU tests");
            let order = input.len().trailing_zeros();
            let twiddles = get_twiddles::<GoldilocksF>(order.into(), RootsConfig::BitReverse)
                .expect("Goldilocks field supports all power-of-two orders up to 32");

            let metal_result = fft(&input, &twiddles, &metal_state)
                .expect("Metal FFT should succeed with valid inputs");
            let cpu_result = crate::fft::cpu::ops::fft(&input, &twiddles)
                .expect("CPU FFT should succeed with valid inputs");

            prop_assert_eq!(&metal_result, &cpu_result);
        }
    }

    #[test]
    fn test_metal_fft_goldilocks_large_input() {
        const ORDER: usize = 16; // 2^16 = 65536 elements
        let input = vec![GoldilocksFE::one(); 1 << ORDER];

        let metal_state = MetalState::new(None).expect("Metal device required for GPU tests");
        let order = input.len().trailing_zeros();
        let twiddles = get_twiddles::<GoldilocksF>(order.into(), RootsConfig::BitReverse)
            .expect("Goldilocks field supports order 16");

        let metal_result = fft(&input, &twiddles, &metal_state)
            .expect("Metal FFT should succeed with valid inputs");
        let cpu_result = crate::fft::cpu::ops::fft(&input, &twiddles)
            .expect("CPU FFT should succeed with valid inputs");

        assert_eq!(&metal_result, &cpu_result);
    }

    // ==================== Twiddle Generation Tests ====================

    #[test]
    fn test_metal_twiddles_stark252_match_cpu() {
        let metal_state = MetalState::new(None).expect("Metal device required for GPU tests");

        for order in 1..=10 {
            for config in [
                RootsConfig::Natural,
                RootsConfig::NaturalInversed,
                RootsConfig::BitReverse,
                RootsConfig::BitReverseInversed,
            ] {
                let metal_twiddles = gen_twiddles::<StarkF>(order, config, &metal_state)
                    .expect("Stark252 twiddle generation should succeed for valid orders");
                let cpu_twiddles = get_twiddles::<StarkF>(order, config)
                    .expect("CPU twiddle generation should succeed for valid orders");

                assert_eq!(
                    metal_twiddles, cpu_twiddles,
                    "Stark252 twiddles mismatch for order={}, config={:?}",
                    order, config
                );
            }
        }
    }

    #[test]
    fn test_metal_twiddles_goldilocks_match_cpu() {
        let metal_state = MetalState::new(None).expect("Metal device required for GPU tests");

        for order in 1..=10 {
            for config in [
                RootsConfig::Natural,
                RootsConfig::NaturalInversed,
                RootsConfig::BitReverse,
                RootsConfig::BitReverseInversed,
            ] {
                let metal_twiddles = gen_twiddles::<GoldilocksF>(order, config, &metal_state)
                    .expect("Goldilocks twiddle generation should succeed for valid orders");
                let cpu_twiddles = get_twiddles::<GoldilocksF>(order, config)
                    .expect("CPU twiddle generation should succeed for valid orders");

                assert_eq!(
                    metal_twiddles, cpu_twiddles,
                    "Goldilocks twiddles mismatch for order={}, config={:?}",
                    order, config
                );
            }
        }
    }

    // ==================== Error Handling Tests ====================

    #[test]
    fn gen_twiddles_with_order_greater_than_63_should_fail() {
        let metal_state = MetalState::new(None).expect("Metal device required for GPU tests");
        let twiddles = gen_twiddles::<StarkF>(64, RootsConfig::Natural, &metal_state);

        assert!(matches!(twiddles, Err(MetalError::FunctionError(_))));
    }

    #[test]
    fn fft_with_non_power_of_two_should_fail() {
        let metal_state = MetalState::new(None).expect("Metal device required for GPU tests");
        let input = vec![StarkFE::one(); 5]; // Not a power of 2
        let twiddles = get_twiddles::<StarkF>(3, RootsConfig::BitReverse)
            .expect("Order 3 is valid for Stark252");

        let result = fft(&input, &twiddles, &metal_state);
        assert!(matches!(result, Err(MetalError::InputError(5))));
    }

    // ==================== Differential Fuzzing Tests ====================

    proptest! {
        /// Differential fuzzing: random Stark252 inputs should produce identical
        /// results between Metal and CPU implementations.
        #[test]
        fn fuzz_metal_vs_cpu_stark252(
            input in stark_field_vec(10),
            use_natural_order in any::<bool>()
        ) {
            let metal_state = MetalState::new(None).expect("Metal device required for GPU tests");
            let order = input.len().trailing_zeros();
            let config = if use_natural_order {
                RootsConfig::BitReverse
            } else {
                RootsConfig::BitReverseInversed
            };
            let twiddles = get_twiddles::<StarkF>(order.into(), config)
                .expect("Stark252 supports all test orders");

            let metal_result = fft(&input, &twiddles, &metal_state)
                .expect("Metal FFT should succeed with valid inputs");
            let cpu_result = crate::fft::cpu::ops::fft(&input, &twiddles)
                .expect("CPU FFT should succeed with valid inputs");

            prop_assert_eq!(
                &metal_result,
                &cpu_result,
                "Stark252 FFT mismatch for input len={}, config={:?}",
                input.len(),
                config
            );
        }

        /// Differential fuzzing: random Goldilocks inputs should produce identical
        /// results between Metal and CPU implementations.
        #[test]
        fn fuzz_metal_vs_cpu_goldilocks(
            input in goldilocks_field_vec(10),
            use_natural_order in any::<bool>()
        ) {
            let metal_state = MetalState::new(None).expect("Metal device required for GPU tests");
            let order = input.len().trailing_zeros();
            let config = if use_natural_order {
                RootsConfig::BitReverse
            } else {
                RootsConfig::BitReverseInversed
            };
            let twiddles = get_twiddles::<GoldilocksF>(order.into(), config)
                .expect("Goldilocks supports all test orders");

            let metal_result = fft(&input, &twiddles, &metal_state)
                .expect("Metal FFT should succeed with valid inputs");
            let cpu_result = crate::fft::cpu::ops::fft(&input, &twiddles)
                .expect("CPU FFT should succeed with valid inputs");

            prop_assert_eq!(
                &metal_result,
                &cpu_result,
                "Goldilocks FFT mismatch for input len={}, config={:?}",
                input.len(),
                config
            );
        }

        /// Differential fuzzing: twiddle generation should match between Metal and CPU
        /// for random orders and all configurations.
        #[test]
        fn fuzz_twiddles_metal_vs_cpu(order in 1u64..20u64, config_idx in 0usize..4) {
            let metal_state = MetalState::new(None).expect("Metal device required for GPU tests");
            let config = match config_idx {
                0 => RootsConfig::Natural,
                1 => RootsConfig::NaturalInversed,
                2 => RootsConfig::BitReverse,
                _ => RootsConfig::BitReverseInversed,
            };

            // Test Stark252
            let metal_stark = gen_twiddles::<StarkF>(order, config, &metal_state)
                .expect("Stark252 twiddle generation should succeed");
            let cpu_stark = get_twiddles::<StarkF>(order, config)
                .expect("CPU Stark252 twiddle generation should succeed");
            prop_assert_eq!(
                &metal_stark,
                &cpu_stark,
                "Stark252 twiddles mismatch for order={}, config={:?}",
                order,
                config
            );

            // Test Goldilocks
            let metal_goldilocks = gen_twiddles::<GoldilocksF>(order, config, &metal_state)
                .expect("Goldilocks twiddle generation should succeed");
            let cpu_goldilocks = get_twiddles::<GoldilocksF>(order, config)
                .expect("CPU Goldilocks twiddle generation should succeed");
            prop_assert_eq!(
                &metal_goldilocks,
                &cpu_goldilocks,
                "Goldilocks twiddles mismatch for order={}, config={:?}",
                order,
                config
            );
        }
    }

    // ==================== Roundtrip Tests ====================

    proptest! {
        /// Roundtrip test: FFT followed by IFFT should recover the original input
        /// for Stark252 field.
        #[test]
        fn test_roundtrip_stark252(input in stark_field_vec(8)) {
            let metal_state = MetalState::new(None).expect("Metal device required for GPU tests");
            let order = input.len().trailing_zeros();

            // Forward FFT with bit-reverse twiddles
            let fwd_twiddles = get_twiddles::<StarkF>(order.into(), RootsConfig::BitReverse)
                .expect("Stark252 supports forward twiddles");
            let transformed = fft(&input, &fwd_twiddles, &metal_state)
                .expect("Forward FFT should succeed");

            // Inverse FFT with bit-reverse inversed twiddles
            let inv_twiddles = get_twiddles::<StarkF>(order.into(), RootsConfig::BitReverseInversed)
                .expect("Stark252 supports inverse twiddles");
            let recovered = fft(&transformed, &inv_twiddles, &metal_state)
                .expect("Inverse FFT should succeed");

            // Scale by 1/n
            let n_inv = StarkFE::from(input.len() as u64).inv()
                .expect("Power of two is invertible in Stark252");
            let recovered: Vec<_> = recovered.iter().map(|x| x * &n_inv).collect();

            prop_assert_eq!(&input, &recovered, "Stark252 roundtrip failed");
        }

        /// Roundtrip test: FFT followed by IFFT should recover the original input
        /// for Goldilocks field.
        #[test]
        fn test_roundtrip_goldilocks(input in goldilocks_field_vec(8)) {
            let metal_state = MetalState::new(None).expect("Metal device required for GPU tests");
            let order = input.len().trailing_zeros();

            // Forward FFT with bit-reverse twiddles
            let fwd_twiddles = get_twiddles::<GoldilocksF>(order.into(), RootsConfig::BitReverse)
                .expect("Goldilocks supports forward twiddles");
            let transformed = fft(&input, &fwd_twiddles, &metal_state)
                .expect("Forward FFT should succeed");

            // Inverse FFT with bit-reverse inversed twiddles
            let inv_twiddles = get_twiddles::<GoldilocksF>(order.into(), RootsConfig::BitReverseInversed)
                .expect("Goldilocks supports inverse twiddles");
            let recovered = fft(&transformed, &inv_twiddles, &metal_state)
                .expect("Inverse FFT should succeed");

            // Scale by 1/n
            let n_inv = GoldilocksFE::from(input.len() as u64).inv()
                .expect("Power of two is invertible in Goldilocks");
            let recovered: Vec<_> = recovered.iter().map(|x| x * &n_inv).collect();

            prop_assert_eq!(&input, &recovered, "Goldilocks roundtrip failed");
        }
    }
}
