use crate::circle::cfft::{order_cfft_result_in_place, order_icfft_input_in_place};
use crate::circle::cosets::Coset;
use crate::circle::twiddles::{get_twiddles, TwiddlesConfig};
use crate::fft::cpu::bit_reversing::in_place_bit_reverse_permute;
use crate::field::element::FieldElement;
use crate::field::fields::mersenne31::field::Mersenne31Field;

use lambdaworks_gpu::metal::abstractions::{errors::MetalError, state::*};
use metal::MTLSize;

use core::mem;

type FE = FieldElement<Mersenne31Field>;

/// Flatten evaluation twiddles (reversed order from get_twiddles) into a single buffer.
///
/// Evaluation twiddles after get_twiddles(_, Evaluation) are reversed:
/// - twiddles[0] has 1 element
/// - twiddles[1] has 2 elements
/// - twiddles[i] has 2^i elements
///
/// Flattened layout: [tw[0], tw[1], tw[2], ...]
/// Offset for layer i = 2^i - 1
fn flatten_eval_twiddles(twiddles: &[Vec<FE>]) -> Vec<FE> {
    let total: usize = twiddles.iter().map(|v| v.len()).sum();
    let mut flat = Vec::with_capacity(total);
    for layer in twiddles {
        flat.extend_from_slice(layer);
    }
    flat
}

/// Flatten interpolation twiddles into a single buffer.
///
/// Interpolation twiddles (not reversed):
/// - twiddles[0] has n/2 elements
/// - twiddles[1] has n/4 elements
/// - twiddles[i] has n/2^(i+1) elements
///
/// Flattened layout: [tw[0], tw[1], tw[2], ...]
/// Offset for layer i = n - n/2^i
fn flatten_interp_twiddles(twiddles: &[Vec<FE>]) -> Vec<FE> {
    let total: usize = twiddles.iter().map(|v| v.len()).sum();
    let mut flat = Vec::with_capacity(total);
    for layer in twiddles {
        flat.extend_from_slice(layer);
    }
    flat
}

/// Raw CFFT butterflies on GPU. Does NOT include bit-reverse or ordering permutation.
///
/// `twiddles` must be evaluation twiddles (reversed order): layer i has 2^i elements.
pub fn cfft_gpu(
    input: &[FE],
    twiddles: &[Vec<FE>],
    state: &MetalState,
) -> Result<Vec<FE>, MetalError> {
    let n = input.len();
    if !n.is_power_of_two() {
        return Err(MetalError::InputError(n));
    }

    let log_n = n.trailing_zeros();
    let flat_twiddles = flatten_eval_twiddles(twiddles);

    let pipeline = state.setup_pipeline("cfft_butterfly_mersenne31")?;
    let input_buffer = state.alloc_buffer_data(input);
    let twiddle_buffer = state.alloc_buffer_data(&flat_twiddles);

    objc::rc::autoreleasepool(|| {
        let (command_buffer, command_encoder) =
            state.setup_command(&pipeline, Some(&[(0, &input_buffer), (1, &twiddle_buffer)]));

        for i in 0..log_n {
            let half_chunk_shift: u32 = i;
            let tw_offset: u32 = (1u32 << i) - 1;

            command_encoder.set_bytes(2, mem::size_of::<u32>() as u64, void_ptr(&half_chunk_shift));
            command_encoder.set_bytes(3, mem::size_of::<u32>() as u64, void_ptr(&tw_offset));

            let grid_size = MTLSize::new(n as u64 / 2, 1, 1);
            let threadgroup_size = MTLSize::new(pipeline.thread_execution_width(), 1, 1);
            command_encoder.dispatch_threads(grid_size, threadgroup_size);
        }

        command_encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
    });

    let raw: Vec<u32> = MetalState::retrieve_contents(&input_buffer);
    Ok(raw.into_iter().map(FieldElement::from_raw).collect())
}

/// Raw ICFFT butterflies on GPU. Does NOT include ordering or bit-reverse permutation.
///
/// `twiddles` must be interpolation twiddles (not reversed): layer 0 has n/2 elements.
pub fn icfft_gpu(
    input: &[FE],
    twiddles: &[Vec<FE>],
    state: &MetalState,
) -> Result<Vec<FE>, MetalError> {
    let n = input.len();
    if !n.is_power_of_two() {
        return Err(MetalError::InputError(n));
    }

    let log_n = n.trailing_zeros();
    let flat_twiddles = flatten_interp_twiddles(twiddles);

    let pipeline = state.setup_pipeline("icfft_butterfly_mersenne31")?;
    let input_buffer = state.alloc_buffer_data(input);
    let twiddle_buffer = state.alloc_buffer_data(&flat_twiddles);

    objc::rc::autoreleasepool(|| {
        let (command_buffer, command_encoder) =
            state.setup_command(&pipeline, Some(&[(0, &input_buffer), (1, &twiddle_buffer)]));

        for i in 0..log_n {
            // ICFFT: chunk_size = 2^(log_n - i), half_chunk = 2^(log_n - 1 - i)
            let half_chunk_shift: u32 = log_n - 1 - i;

            // Offset into flattened twiddles for layer i:
            // Layer 0 has n/2 elements at offset 0
            // Layer i offset = n - n/2^i
            let tw_offset: u32 = if i == 0 {
                0
            } else {
                n as u32 - (n as u32 >> i)
            };

            command_encoder.set_bytes(2, mem::size_of::<u32>() as u64, void_ptr(&half_chunk_shift));
            command_encoder.set_bytes(3, mem::size_of::<u32>() as u64, void_ptr(&tw_offset));

            let grid_size = MTLSize::new(n as u64 / 2, 1, 1);
            let threadgroup_size = MTLSize::new(pipeline.thread_execution_width(), 1, 1);
            command_encoder.dispatch_threads(grid_size, threadgroup_size);
        }

        command_encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
    });

    let raw: Vec<u32> = MetalState::retrieve_contents(&input_buffer);
    Ok(raw.into_iter().map(FieldElement::from_raw).collect())
}

/// Full evaluate: bit-reverse → CFFT butterflies (GPU) → ordering permutation → return
///
/// Matches the CPU implementation in circle/polynomial.rs evaluate_cfft().
pub fn evaluate_cfft_gpu(mut coeff: Vec<FE>, state: &MetalState) -> Result<Vec<FE>, MetalError> {
    let domain_log_2_size: u32 = coeff.len().trailing_zeros();
    let coset = Coset::new_standard(domain_log_2_size);
    let twiddles = get_twiddles(coset, TwiddlesConfig::Evaluation);

    in_place_bit_reverse_permute::<FE>(&mut coeff);
    let mut result = cfft_gpu(&coeff, &twiddles, state)?;
    order_cfft_result_in_place(&mut result);
    Ok(result)
}

/// Full interpolate: ordering → ICFFT butterflies (GPU) → bit-reverse → scale by 1/n → return
///
/// Matches the CPU implementation in circle/polynomial.rs interpolate_cfft().
pub fn interpolate_cfft_gpu(mut eval: Vec<FE>, state: &MetalState) -> Result<Vec<FE>, MetalError> {
    if eval.is_empty() {
        return Ok(Vec::new());
    }

    let domain_log_2_size: u32 = eval.len().trailing_zeros();
    let coset = Coset::new_standard(domain_log_2_size);
    let twiddles = get_twiddles(coset, TwiddlesConfig::Interpolation);

    order_icfft_input_in_place(&mut eval);
    let mut result = icfft_gpu(&eval, &twiddles, state)?;
    in_place_bit_reverse_permute::<FE>(&mut result);

    let factor = FE::from(result.len() as u64)
        .inv()
        .expect("evaluation length is non-zero, so its inverse exists");
    Ok(result.into_iter().map(|c| c * factor).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circle::cfft::{cfft, icfft};
    use crate::circle::polynomial::{evaluate_cfft, interpolate_cfft};
    use proptest::{collection, prelude::*};

    fn metal_state() -> MetalState {
        MetalState::new(None).expect("Metal device required for GPU tests")
    }

    prop_compose! {
        fn powers_of_two(max_exp: u8)(exp in 2..max_exp) -> usize { 1 << exp }
    }

    prop_compose! {
        fn mersenne31_field_element()(num in 1u32..0x7FFFFFFFu32) -> FE {
            FE::from(&num)
        }
    }

    fn mersenne31_field_vec(max_exp: u8) -> impl Strategy<Value = Vec<FE>> {
        powers_of_two(max_exp)
            .prop_flat_map(|size| collection::vec(mersenne31_field_element(), size))
    }

    // ==================== Raw CFFT Tests ====================

    proptest! {
        #[test]
        fn test_metal_cfft_matches_cpu(input in mersenne31_field_vec(10)) {
            let state = metal_state();
            let domain_log_2_size = input.len().trailing_zeros();
            let coset = Coset::new_standard(domain_log_2_size);
            let twiddles = get_twiddles(coset, TwiddlesConfig::Evaluation);

            let gpu_result = cfft_gpu(&input, &twiddles, &state)
                .expect("GPU CFFT should succeed");

            let mut cpu_input = input;
            cfft(&mut cpu_input, &twiddles);

            prop_assert_eq!(&gpu_result, &cpu_input.to_vec());
        }
    }

    proptest! {
        #[test]
        fn test_metal_icfft_matches_cpu(input in mersenne31_field_vec(10)) {
            let state = metal_state();
            let domain_log_2_size = input.len().trailing_zeros();
            let coset = Coset::new_standard(domain_log_2_size);
            let twiddles = get_twiddles(coset, TwiddlesConfig::Interpolation);

            let gpu_result = icfft_gpu(&input, &twiddles, &state)
                .expect("GPU ICFFT should succeed");

            let mut cpu_input = input;
            icfft(&mut cpu_input, &twiddles);

            prop_assert_eq!(&gpu_result, &cpu_input.to_vec());
        }
    }

    // ==================== Full Evaluate/Interpolate Tests ====================

    proptest! {
        #[test]
        fn test_evaluate_cfft_gpu_matches_cpu(input in mersenne31_field_vec(10)) {
            let state = metal_state();

            let gpu_result = evaluate_cfft_gpu(input.clone(), &state)
                .expect("GPU evaluate should succeed");
            let cpu_result = evaluate_cfft(input);

            prop_assert_eq!(&gpu_result, &cpu_result);
        }
    }

    proptest! {
        #[test]
        fn test_interpolate_cfft_gpu_matches_cpu(input in mersenne31_field_vec(10)) {
            let state = metal_state();

            let gpu_result = interpolate_cfft_gpu(input.clone(), &state)
                .expect("GPU interpolate should succeed");
            let cpu_result = interpolate_cfft(input);

            prop_assert_eq!(&gpu_result, &cpu_result);
        }
    }

    // ==================== Roundtrip Tests ====================

    proptest! {
        #[test]
        fn test_roundtrip_evaluate_interpolate(input in mersenne31_field_vec(10)) {
            let state = metal_state();

            let evals = evaluate_cfft_gpu(input.clone(), &state)
                .expect("GPU evaluate should succeed");
            let recovered = interpolate_cfft_gpu(evals, &state)
                .expect("GPU interpolate should succeed");

            prop_assert_eq!(&input, &recovered);
        }
    }

    // ==================== Deterministic Tests ====================

    #[test]
    fn test_cfft_gpu_size_4() {
        let state = metal_state();
        let input: Vec<FE> = (1..=4).map(FE::from).collect();

        let gpu_result =
            evaluate_cfft_gpu(input.clone(), &state).expect("GPU evaluate should succeed");
        let cpu_result = evaluate_cfft(input);

        assert_eq!(gpu_result, cpu_result);
    }

    #[test]
    fn test_cfft_gpu_size_8() {
        let state = metal_state();
        let input: Vec<FE> = (1..=8).map(FE::from).collect();

        let gpu_result =
            evaluate_cfft_gpu(input.clone(), &state).expect("GPU evaluate should succeed");
        let cpu_result = evaluate_cfft(input);

        assert_eq!(gpu_result, cpu_result);
    }

    #[test]
    fn test_cfft_gpu_size_16() {
        let state = metal_state();
        let input: Vec<FE> = (1..=16).map(FE::from).collect();

        let gpu_result =
            evaluate_cfft_gpu(input.clone(), &state).expect("GPU evaluate should succeed");
        let cpu_result = evaluate_cfft(input);

        assert_eq!(gpu_result, cpu_result);
    }

    #[test]
    fn test_roundtrip_size_8() {
        let state = metal_state();
        let input: Vec<FE> = (1..=8).map(FE::from).collect();

        let evals = evaluate_cfft_gpu(input.clone(), &state).expect("GPU evaluate should succeed");
        let recovered =
            interpolate_cfft_gpu(evals, &state).expect("GPU interpolate should succeed");

        assert_eq!(input, recovered);
    }

    #[test]
    fn test_roundtrip_size_2_pow_16() {
        let state = metal_state();
        let input: Vec<FE> = (0..65536u32).map(|i| FE::from(&i)).collect();

        let evals = evaluate_cfft_gpu(input.clone(), &state).expect("GPU evaluate should succeed");
        let recovered =
            interpolate_cfft_gpu(evals, &state).expect("GPU interpolate should succeed");

        assert_eq!(input, recovered);
    }

    // ==================== Error Handling Tests ====================

    #[test]
    fn cfft_gpu_non_power_of_two_fails() {
        let state = metal_state();
        let input = vec![FE::one(); 5];
        let coset = Coset::new_standard(3);
        let twiddles = get_twiddles(coset, TwiddlesConfig::Evaluation);

        let result = cfft_gpu(&input, &twiddles, &state);
        assert!(matches!(result, Err(MetalError::InputError(5))));
    }
}
