use lambdaworks_math::{
    fft::bit_reversing::in_place_bit_reverse_permute,
    field::{
        element::FieldElement,
        traits::{IsTwoAdicField, RootsConfig},
    },
};

use crate::metal::{abstractions::state::*, fft::errors::FFTMetalError};

use super::helpers::{log2, void_ptr};
use metal::MTLSize;

use core::mem;

/// Executes parallel ordered FFT over a slice of two-adic field elements, in Metal.
/// Twiddle factors are required to be in bit-reverse order.
///
/// "Ordered" means that the input is required to be in natural order, and the output will be
/// in this order too. Natural order means that input[i] corresponds to the i-th coefficient,
/// as opposed to bit-reverse order in which input[bit_rev(i)] corresponds to the i-th
/// coefficient.
pub fn fft<F: IsTwoAdicField>(
    input: &[FieldElement<F>],
    twiddles: &[FieldElement<F>],
    state: &MetalState,
) -> Result<Vec<FieldElement<F>>, FFTMetalError> {
    let pipeline = state
        .setup_pipeline("radix2_dit_butterfly")
        .map_err(FFTMetalError::Metal)?;

    let input_buffer = state.alloc_buffer_data(input);
    let twiddles_buffer = state.alloc_buffer_data(twiddles);
    // TODO: twiddle factors security (right now anything can be passed as twiddle factors)

    let (command_buffer, command_encoder) = state.setup_command(
        &pipeline,
        Some(&[(0, &input_buffer), (1, &twiddles_buffer)]),
    );

    let order = log2(input.len()).map_err(FFTMetalError::FFT)?;
    for stage in 0..order {
        let group_count = stage + 1;
        let group_size = input.len() as u64 / (1 << stage);

        command_encoder.set_bytes(
            2,
            mem::size_of::<F::BaseType>() as u64,
            void_ptr(&group_size),
        );

        let threadgroup_size = MTLSize::new(group_size / 2, 1, 1);
        let threadgroup_count = MTLSize::new(group_count, 1, 1);
        command_encoder.dispatch_thread_groups(threadgroup_count, threadgroup_size);
    }
    command_encoder.end_encoding();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    let mut result = MetalState::retrieve_contents(&input_buffer);
    in_place_bit_reverse_permute(&mut result); // TODO: implement this in metal.
    Ok(result.iter().map(FieldElement::from).collect())
}

/// Generates 2^{`order`} naturally-ordered twiddle factors in parallel, in Metal.
pub fn gen_twiddles<F: IsTwoAdicField>(
    order: u64,
    config: RootsConfig,
    state: &MetalState,
) -> Result<Vec<FieldElement<F>>, FFTMetalError> {
    let len = (1 << order) / 2;

    let pipeline = match config {
        RootsConfig::Natural => state.setup_pipeline("calc_twiddle"),
        RootsConfig::NaturalInversed => state.setup_pipeline("calc_twiddle_inv"),
        RootsConfig::BitReverse => state.setup_pipeline("calc_twiddle_bitrev"),
        RootsConfig::BitReverseInversed => state.setup_pipeline("calc_twiddle_bitrev_inv"),
    }
    .map_err(FFTMetalError::Metal)?;

    let result_buffer = state.alloc_buffer::<F::BaseType>(len);

    let (command_buffer, command_encoder) =
        state.setup_command(&pipeline, Some(&[(0, &result_buffer)]));

    let root = F::get_primitive_root_of_unity(order).map_err(FFTMetalError::FFT)?;
    command_encoder.set_bytes(1, mem::size_of::<F::BaseType>() as u64, void_ptr(&root));

    let grid_size = MTLSize::new(len as u64, 1, 1);
    let threadgroup_size = MTLSize::new(pipeline.max_total_threads_per_threadgroup(), 1, 1);

    command_encoder.dispatch_threads(grid_size, threadgroup_size);
    command_encoder.end_encoding();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    let result = MetalState::retrieve_contents(&result_buffer);
    Ok(result.iter().map(FieldElement::from).collect())
}

#[cfg(test)]
mod tests {
    use crate::metal::abstractions::state::*;
    use lambdaworks_math::{
        field::{test_fields::u32_test_field::U32TestField, traits::RootsConfig},
        polynomial::Polynomial,
    };
    use proptest::prelude::*;

    use super::*;

    type F = U32TestField;
    type FE = FieldElement<F>;

    prop_compose! {
        fn powers_of_two(max_exp: u8)(exp in 1..max_exp) -> usize { 1 << exp }
        // max_exp cannot be multiple of the bits that represent a usize, generally 64 or 32.
        // also it can't exceed the test field's two-adicity.
    }
    prop_compose! {
        fn field_element()(num in any::<u64>().prop_filter("Avoid null polynomial", |x| x != &0)) -> FE {
            FE::from(num)
        }
    }
    prop_compose! {
        fn field_vec(max_exp: u8)(elem in field_element(), size in powers_of_two(max_exp)) -> Vec<FE> {
            vec![elem; size]
        }
    }
    prop_compose! {
        fn poly(max_exp: u8)(coeffs in field_vec(max_exp)) -> Polynomial<FE> {
            Polynomial::new(&coeffs)
        }
    }

    proptest! {
        // Property-based test that ensures Metal parallel FFT gives same result as a sequential one.
        // These tests actually pass, but we ignore them because they fail in the CI due to a lack of GPU
        #[test]
        fn test_metal_fft_matches_sequential(poly in poly(8)) {
            objc::rc::autoreleasepool(|| {
                let expected = poly.evaluate_fft().unwrap();
                let order = poly.coefficients().len().trailing_zeros() as u64;

                let metal_state = MetalState::new(None).unwrap();
                let twiddles = F::get_twiddles(order, RootsConfig::BitReverse).unwrap();
                let result = fft(poly.coefficients(), &twiddles, &metal_state).unwrap();

                prop_assert_eq!(&result, &expected);

                Ok(())
            }).unwrap();
        }
    }

    proptest! {
        // These tests actually pass, but we ignore them because they fail in the CI due to a lack of GPU
        #[test]
        fn test_gpu_twiddles_match_cpu(order in 2..16) {
            objc::rc::autoreleasepool(|| {
                let configs = [
                    RootsConfig::Natural,
                    RootsConfig::NaturalInversed,
                    RootsConfig::BitReverse,
                    RootsConfig::BitReverseInversed,
                ];

                for config in configs {
                    let cpu_twiddles = F::get_twiddles(order as u64, config).unwrap();

                    let metal_state = MetalState::new(None).unwrap();
                    let gpu_twiddles = gen_twiddles::<F>(order as u64, config, &metal_state).unwrap();

                    prop_assert_eq!(cpu_twiddles, gpu_twiddles);
                }
                Ok(())
            }).unwrap();
        }
    }
}
