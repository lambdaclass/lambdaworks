use lambdaworks_math::field::{
    element::FieldElement,
    traits::{IsTwoAdicField, RootsConfig},
};

use crate::metal::{
    abstractions::{errors::MetalError, state::*},
    fft::errors::FFTMetalError,
};

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
    let pipeline = state.setup_pipeline("radix2_dit_butterfly")?;

    let input_buffer = state.alloc_buffer_data(input);
    let twiddles_buffer = state.alloc_buffer_data(twiddles);
    // TODO: twiddle factors security (right now anything can be passed as twiddle factors)

    let (command_buffer, command_encoder) = state.setup_command(
        &pipeline,
        Some(&[(0, &input_buffer), (1, &twiddles_buffer)]),
    );

    let order = log2(input.len())?;
    for stage in 0..order {
        let group_count = 1 << stage;
        let group_size = input.len() as u64 / group_count;

        let threadgroup_size = MTLSize::new(group_size / 2, 1, 1);
        let threadgroup_count = MTLSize::new(group_count, 1, 1);
        command_encoder.dispatch_thread_groups(threadgroup_count, threadgroup_size);
    }
    command_encoder.end_encoding();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    let result = MetalState::retrieve_contents(&input_buffer);
    let result = bitrev_permutation(&result, state)?;
    Ok(result.iter().map(FieldElement::from_raw).collect())
}

/// Executes parallel ordered FFT in a bigger domain over a slice of two-adic field elements, in Metal.
///
/// "Ordered" means that the input is required to be in natural order, and the output will be
/// in this order too. Natural order means that input[i] corresponds to the i-th coefficient,
/// as opposed to bit-reverse order in which input[bit_rev(i)] corresponds to the i-th
/// coefficient.
pub fn fft_with_blowup<F: IsTwoAdicField>(
    input: &[FieldElement<F>],
    blowup_factor: usize,
    state: &MetalState,
) -> Result<Vec<FieldElement<F>>, FFTMetalError> {
    let domain_size = input.len() * blowup_factor;
    let order = log2(domain_size)?;
    let twiddles = gen_twiddles(order, RootsConfig::BitReverse, state)?;
    let mut resized = input.to_vec();
    resized.resize(domain_size, FieldElement::zero());

    fft(&resized, &twiddles, state)
}

/// Generates 2^{`order`} twiddle factors in parallel, with a certain `config`, in Metal.
pub fn gen_twiddles<F: IsTwoAdicField>(
    order: u64,
    config: RootsConfig,
    state: &MetalState,
) -> Result<Vec<FieldElement<F>>, FFTMetalError> {
    let len = (1 << order) / 2;

    let kernel = match config {
        RootsConfig::Natural => "calc_twiddles",
        RootsConfig::NaturalInversed => "calc_twiddles_inv",
        RootsConfig::BitReverse => "calc_twiddles_bitrev",
        RootsConfig::BitReverseInversed => "calc_twiddles_bitrev_inv",
    };

    let pipeline = state.setup_pipeline(kernel)?;

    let result_buffer = state.alloc_buffer::<F::BaseType>(len);

    let (command_buffer, command_encoder) =
        state.setup_command(&pipeline, Some(&[(0, &result_buffer)]));

    let root = F::get_primitive_root_of_unity::<F>(order).unwrap();
    command_encoder.set_bytes(1, mem::size_of::<F::BaseType>() as u64, void_ptr(&root));

    let grid_size = MTLSize::new(len as u64, 1, 1);
    let threadgroup_size = MTLSize::new(pipeline.max_total_threads_per_threadgroup(), 1, 1);

    command_encoder.dispatch_threads(grid_size, threadgroup_size);
    command_encoder.end_encoding();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    let result = MetalState::retrieve_contents(&result_buffer);
    Ok(result.iter().map(FieldElement::from_raw).collect())
}

/// Executes a parallel bit-reverse permutation with the elements of `input`, in Metal.
pub fn bitrev_permutation<T: Clone>(input: &[T], state: &MetalState) -> Result<Vec<T>, MetalError> {
    let pipeline = state.setup_pipeline("bitrev_permutation")?;

    let input_buffer = state.alloc_buffer_data(input);
    let result_buffer = state.alloc_buffer::<T>(input.len());

    let (command_buffer, command_encoder) =
        state.setup_command(&pipeline, Some(&[(0, &input_buffer), (1, &result_buffer)]));

    let grid_size = MTLSize::new(input.len() as u64, 1, 1);
    let threadgroup_size = MTLSize::new(pipeline.max_total_threads_per_threadgroup(), 1, 1);

    command_encoder.dispatch_threads(grid_size, threadgroup_size);
    command_encoder.end_encoding();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    Ok(MetalState::retrieve_contents::<T>(&result_buffer))
}
