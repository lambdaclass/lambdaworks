//! Metal GPU FFT wrapper functions for the STARK prover.

#[cfg(all(target_os = "macos", feature = "metal"))]
use lambdaworks_gpu::metal::abstractions::{
    errors::MetalError,
    state::{DynamicMetalState, MetalState},
};

#[cfg(all(target_os = "macos", feature = "metal"))]
use lambdaworks_math::{
    fft::gpu::metal::ops::{
        fft, fft_buffer_to_buffer, fft_buffer_to_buffer_reuse, fft_encode_to_command_buffer,
        fft_to_buffer, gen_twiddles, gen_twiddles_to_buffer,
    },
    field::{
        element::FieldElement,
        fields::u64_goldilocks_field::Goldilocks64Field,
        traits::{IsFFTField, IsSubFieldOf, RootsConfig},
    },
};

#[cfg(all(target_os = "macos", feature = "metal"))]
use crate::metal::{canonical, to_raw_u64};

/// Interpolates polynomial coefficients from evaluations via inverse FFT + 1/n normalization.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn gpu_interpolate_fft<F>(
    evaluations: &[FieldElement<F>],
    state: &MetalState,
) -> Result<Vec<FieldElement<F>>, MetalError>
where
    F: IsFFTField + IsSubFieldOf<F>,
    F::BaseType: Copy,
{
    let order = evaluations.len().trailing_zeros() as u64;
    let inv_twiddles = gen_twiddles::<F>(order, RootsConfig::BitReverseInversed, state)?;
    let coeffs = fft(evaluations, &inv_twiddles, state)?;

    let n_inv = FieldElement::<F>::from(evaluations.len() as u64)
        .inv()
        .expect("Power-of-two length is always invertible in an FFT field");

    Ok(coeffs.iter().map(|c| c * &n_inv).collect())
}

/// Evaluates a polynomial on the offset coset domain `{offset * w^i}` via Metal GPU FFT.
/// This is the Low-Degree Extension (LDE) operation.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn gpu_evaluate_offset_fft<F>(
    coefficients: &[FieldElement<F>],
    blowup_factor: usize,
    offset: &FieldElement<F>,
    state: &MetalState,
) -> Result<Vec<FieldElement<F>>, MetalError>
where
    F: IsFFTField + IsSubFieldOf<F>,
    F::BaseType: Copy,
{
    let shifted = coset_shift_and_pad(coefficients, blowup_factor, offset);
    let order = shifted.len().trailing_zeros() as u64;
    let twiddles = gen_twiddles::<F>(order, RootsConfig::BitReverse, state)?;
    fft(&shifted, &twiddles, state)
}

/// Applies coset shift (multiply coeff k by offset^k) and zero-pads to `len * blowup_factor`.
#[cfg(all(target_os = "macos", feature = "metal"))]
fn coset_shift_and_pad<F: IsFFTField + IsSubFieldOf<F>>(
    coefficients: &[FieldElement<F>],
    blowup_factor: usize,
    offset: &FieldElement<F>,
) -> Vec<FieldElement<F>> {
    let domain_size = coefficients.len() * blowup_factor;
    let mut shifted = Vec::with_capacity(domain_size);
    let mut offset_power = FieldElement::<F>::one();
    for coeff in coefficients {
        shifted.push(coeff * &offset_power);
        offset_power = &offset_power * offset;
    }
    shifted.resize(domain_size, FieldElement::zero());
    shifted
}

/// Batch-evaluates multiple polynomials on a coset domain, generating twiddles once.
/// Returns `Vec<(Buffer, element_count)>`.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn gpu_evaluate_offset_fft_to_buffers_batch<F>(
    polynomials: &[&[FieldElement<F>]],
    blowup_factor: usize,
    offset: &FieldElement<F>,
    state: &MetalState,
) -> Result<Vec<(metal::Buffer, usize)>, MetalError>
where
    F: IsFFTField + IsSubFieldOf<F>,
    F::BaseType: Copy,
{
    if polynomials.is_empty() {
        return Ok(Vec::new());
    }

    let domain_size = polynomials[0].len() * blowup_factor;
    let order = domain_size.trailing_zeros() as u64;
    let twiddles_buffer = gen_twiddles_to_buffer::<F>(order, RootsConfig::BitReverse, state)?;

    let mut results = Vec::with_capacity(polynomials.len());
    for coefficients in polynomials {
        let shifted = coset_shift_and_pad(coefficients, blowup_factor, offset);
        let result_buffer = fft_to_buffer::<F>(&shifted, &twiddles_buffer, state)?;
        results.push((result_buffer, shifted.len()));
    }

    Ok(results)
}

/// Batch-evaluates polynomials from GPU coefficient buffers on a coset domain.
/// Reads from existing Metal Buffers, keeping the entire pipeline on GPU.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn gpu_evaluate_offset_fft_buffer_to_buffers_batch(
    coeff_buffers: &[metal::Buffer],
    part_len: usize,
    blowup_factor: usize,
    offset: &FieldElement<Goldilocks64Field>,
    coset_state: &CosetShiftState,
    metal_state: &MetalState,
) -> Result<Vec<(metal::Buffer, usize)>, MetalError> {
    if coeff_buffers.is_empty() {
        return Ok(Vec::new());
    }

    let domain_size = part_len * blowup_factor;
    let order = domain_size.trailing_zeros() as u64;
    let twiddles_buffer =
        gen_twiddles_to_buffer::<Goldilocks64Field>(order, RootsConfig::BitReverse, metal_state)?;
    let working_buffer = metal_state.alloc_buffer::<u64>(domain_size);

    let mut results = Vec::with_capacity(coeff_buffers.len());
    for coeff_buf in coeff_buffers {
        let shifted_buffer = gpu_coset_shift_buffer_to_buffer(
            coeff_buf,
            part_len,
            offset,
            domain_size,
            coset_state,
        )?;
        let result_buffer = fft_buffer_to_buffer_reuse::<Goldilocks64Field>(
            &shifted_buffer,
            domain_size,
            &twiddles_buffer,
            &working_buffer,
            metal_state,
        )?;
        drop(shifted_buffer);
        results.push((result_buffer, domain_size));
    }

    Ok(results)
}

// =============================================================================
// GPU Coset Shift + Scale kernels (Goldilocks-specific)
// =============================================================================

#[cfg(all(target_os = "macos", feature = "metal"))]
const GOLDILOCKS_FIELD_HEADER: &str =
    include_str!("../../../../math/src/gpu/metal/shaders/field/fp_u64.h.metal");

#[cfg(all(target_os = "macos", feature = "metal"))]
const COSET_SHIFT_SHADER: &str = include_str!("shaders/coset_shift.metal");

/// Pre-compiled Metal state for coset shift and scale operations.
/// Caches compiled pipelines for `goldilocks_coset_shift`, `goldilocks_scale`,
/// and `goldilocks_cyclic_mul`. Create once and reuse across the entire prove call.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub struct CosetShiftState {
    state: DynamicMetalState,
    coset_shift_max_threads: u64,
    scale_max_threads: u64,
    cyclic_mul_max_threads: u64,
}

#[cfg(all(target_os = "macos", feature = "metal"))]
impl CosetShiftState {
    /// Compile the coset shift shader and prepare all pipelines.
    pub fn new() -> Result<Self, MetalError> {
        let combined_source = format!("{}\n{}", GOLDILOCKS_FIELD_HEADER, COSET_SHIFT_SHADER);
        let mut state = DynamicMetalState::new()?;
        state.load_library(&combined_source)?;
        Self::with_pipelines(state)
    }

    /// Compile the coset shift shader sharing a device and queue with an existing Metal state.
    pub fn from_device_and_queue(
        device: &metal::Device,
        queue: &metal::CommandQueue,
    ) -> Result<Self, MetalError> {
        let combined_source = format!("{}\n{}", GOLDILOCKS_FIELD_HEADER, COSET_SHIFT_SHADER);
        let mut state = DynamicMetalState::from_device_and_queue(device, queue);
        state.load_library(&combined_source)?;
        Self::with_pipelines(state)
    }

    fn with_pipelines(mut state: DynamicMetalState) -> Result<Self, MetalError> {
        let coset_shift_max_threads = state.prepare_pipeline("goldilocks_coset_shift")?;
        let scale_max_threads = state.prepare_pipeline("goldilocks_scale")?;
        let cyclic_mul_max_threads = state.prepare_pipeline("goldilocks_cyclic_mul")?;
        Ok(Self {
            state,
            coset_shift_max_threads,
            scale_max_threads,
            cyclic_mul_max_threads,
        })
    }
}

/// Dispatches a 1D compute kernel with the given pipeline, buffers, and element count.
/// Commits the command buffer and waits for completion.
#[cfg(all(target_os = "macos", feature = "metal"))]
fn dispatch_kernel(
    state: &CosetShiftState,
    pipeline_name: &str,
    max_threads: u64,
    buffers: &[&metal::Buffer],
    num_elements: usize,
) -> Result<(), MetalError> {
    use metal::MTLSize;

    let pipeline = state
        .state
        .get_pipeline_ref(pipeline_name)
        .ok_or_else(|| MetalError::FunctionError(pipeline_name.to_string()))?;

    let threads_per_group = max_threads.min(256);
    let thread_groups = (num_elements as u64).div_ceil(threads_per_group);

    let command_buffer = state.state.command_queue().new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    for (i, buf) in buffers.iter().enumerate() {
        encoder.set_buffer(i as u64, Some(buf), 0);
    }
    encoder.dispatch_thread_groups(
        MTLSize::new(thread_groups, 1, 1),
        MTLSize::new(threads_per_group, 1, 1),
    );
    encoder.end_encoding();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    if command_buffer.status() == metal::MTLCommandBufferStatus::Error {
        return Err(MetalError::ExecutionError(format!(
            "GPU {pipeline_name} command buffer error"
        )));
    }

    Ok(())
}

/// Allocates a u32 scalar as a GPU buffer.
#[cfg(all(target_os = "macos", feature = "metal"))]
fn alloc_u32(state: &CosetShiftState, val: u32) -> Result<metal::Buffer, MetalError> {
    state
        .state
        .alloc_buffer_with_data(std::slice::from_ref(&val))
}

/// Allocates a u64 scalar as a GPU buffer.
#[cfg(all(target_os = "macos", feature = "metal"))]
fn alloc_u64(state: &CosetShiftState, val: u64) -> Result<metal::Buffer, MetalError> {
    state
        .state
        .alloc_buffer_with_data(std::slice::from_ref(&val))
}

/// Perform coset shift on GPU: `output[k] = coeffs[k] * offset^k`, zero-padded to `output_len`.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn gpu_coset_shift_to_buffer(
    coeffs: &[FieldElement<Goldilocks64Field>],
    offset: &FieldElement<Goldilocks64Field>,
    output_len: usize,
    coset_state: &CosetShiftState,
) -> Result<metal::Buffer, MetalError> {
    let input_len = coeffs.len();
    let coeffs_u64 = to_raw_u64(coeffs);
    let offset_u64 = canonical(offset);

    let buf_input = coset_state.state.alloc_buffer_with_data(&coeffs_u64)?;
    let buf_output = coset_state
        .state
        .alloc_buffer(output_len * std::mem::size_of::<u64>())?;
    let buf_offset = alloc_u64(coset_state, offset_u64)?;
    let buf_input_len = alloc_u32(coset_state, input_len as u32)?;
    let buf_output_len = alloc_u32(coset_state, output_len as u32)?;

    dispatch_kernel(
        coset_state,
        "goldilocks_coset_shift",
        coset_state.coset_shift_max_threads,
        &[
            &buf_input,
            &buf_output,
            &buf_offset,
            &buf_input_len,
            &buf_output_len,
        ],
        output_len,
    )?;

    Ok(buf_output)
}

/// Scale all elements of a GPU buffer: `output[k] = input[k] * scalar`.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn gpu_scale_buffer(
    buffer: &metal::Buffer,
    len: usize,
    scalar: &FieldElement<Goldilocks64Field>,
    coset_state: &CosetShiftState,
) -> Result<metal::Buffer, MetalError> {
    let scalar_u64 = canonical(scalar);

    let buf_output = coset_state
        .state
        .alloc_buffer(len * std::mem::size_of::<u64>())?;
    let buf_scalar = alloc_u64(coset_state, scalar_u64)?;
    let buf_len = alloc_u32(coset_state, len as u32)?;

    dispatch_kernel(
        coset_state,
        "goldilocks_scale",
        coset_state.scale_max_threads,
        &[buffer, &buf_output, &buf_scalar, &buf_len],
        len,
    )?;

    Ok(buf_output)
}

/// Scale all elements of a GPU buffer in-place: `buffer[k] *= scalar`.
/// Safe because each thread accesses a unique element.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn gpu_scale_buffer_inplace(
    buffer: &metal::Buffer,
    len: usize,
    scalar: &FieldElement<Goldilocks64Field>,
    coset_state: &CosetShiftState,
) -> Result<(), MetalError> {
    let scalar_u64 = canonical(scalar);

    let buf_scalar = alloc_u64(coset_state, scalar_u64)?;
    let buf_len = alloc_u32(coset_state, len as u32)?;

    dispatch_kernel(
        coset_state,
        "goldilocks_scale",
        coset_state.scale_max_threads,
        &[buffer, buffer, &buf_scalar, &buf_len],
        len,
    )
}

/// Coset shift from GPU buffer to GPU buffer: `output[k] = input[k] * offset^k`,
/// zero-padded to `output_len`.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn gpu_coset_shift_buffer_to_buffer(
    input_buffer: &metal::Buffer,
    input_len: usize,
    offset: &FieldElement<Goldilocks64Field>,
    output_len: usize,
    state: &CosetShiftState,
) -> Result<metal::Buffer, MetalError> {
    let offset_u64 = canonical(offset);

    let buf_output = state
        .state
        .alloc_buffer(output_len * std::mem::size_of::<u64>())?;
    let buf_offset = alloc_u64(state, offset_u64)?;
    let buf_input_len = alloc_u32(state, input_len as u32)?;
    let buf_output_len = alloc_u32(state, output_len as u32)?;

    dispatch_kernel(
        state,
        "goldilocks_coset_shift",
        state.coset_shift_max_threads,
        &[
            input_buffer,
            &buf_output,
            &buf_offset,
            &buf_input_len,
            &buf_output_len,
        ],
        output_len,
    )?;

    Ok(buf_output)
}

/// Element-wise cyclic multiply: `output[i] = input[i] * pattern[i % pattern_len]`.
/// Used for combining base zerofier with end-exemptions evaluations on GPU.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn gpu_cyclic_mul_buffer(
    input: &metal::Buffer,
    len: usize,
    pattern: &[u64],
    state: &CosetShiftState,
) -> Result<metal::Buffer, MetalError> {
    let buf_output = state.state.alloc_buffer(len * std::mem::size_of::<u64>())?;
    let buf_pattern = state.state.alloc_buffer_with_data(pattern)?;
    let buf_len = alloc_u32(state, len as u32)?;
    let buf_pattern_len = alloc_u32(state, pattern.len() as u32)?;

    dispatch_kernel(
        state,
        "goldilocks_cyclic_mul",
        state.cyclic_mul_max_threads,
        &[input, &buf_output, &buf_pattern, &buf_len, &buf_pattern_len],
        len,
    )?;

    Ok(buf_output)
}

/// GPU inverse FFT returning a Metal Buffer (no CPU readback).
/// Generates inverse twiddles, runs FFT, and normalizes by 1/n via `gpu_scale_buffer`.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn gpu_ifft_to_buffer(
    evaluations: &[FieldElement<Goldilocks64Field>],
    coset_state: &CosetShiftState,
    metal_state: &MetalState,
) -> Result<metal::Buffer, MetalError> {
    type FpE = FieldElement<Goldilocks64Field>;

    let len = evaluations.len();
    let order = len.trailing_zeros() as u64;

    let inv_twiddles = gen_twiddles_to_buffer::<Goldilocks64Field>(
        order,
        RootsConfig::BitReverseInversed,
        metal_state,
    )?;

    let evals_u64 = to_raw_u64(evaluations);
    let eval_buffer = metal_state.alloc_buffer_data(&evals_u64);

    let result_buffer =
        fft_buffer_to_buffer::<Goldilocks64Field>(&eval_buffer, len, &inv_twiddles, metal_state)?;

    let n_inv = FpE::from(len as u64)
        .inv()
        .expect("Power-of-two length is always invertible in an FFT field");
    gpu_scale_buffer(&result_buffer, len, &n_inv, coset_state)
}

/// Coset IFFT on GPU: recovers polynomial coefficients from evaluations on
/// `{offset * w^i}` by performing IFFT + inverse coset shift.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn gpu_interpolate_offset_fft_to_buffer(
    evaluations: &[FieldElement<Goldilocks64Field>],
    coset_offset: &FieldElement<Goldilocks64Field>,
    coset_state: &CosetShiftState,
    metal_state: &MetalState,
) -> Result<metal::Buffer, MetalError> {
    let len = evaluations.len();
    let coeffs_buffer = gpu_ifft_to_buffer(evaluations, coset_state, metal_state)?;
    let offset_inv = coset_offset.inv().expect("Coset offset must be invertible");
    gpu_coset_shift_buffer_to_buffer(&coeffs_buffer, len, &offset_inv, len, coset_state)
}

/// Coset IFFT on a GPU buffer: like `gpu_interpolate_offset_fft_to_buffer` but reads
/// from an existing Metal Buffer, avoiding CPU-to-GPU transfer.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn gpu_interpolate_offset_fft_buffer_to_buffer(
    eval_buffer: &metal::Buffer,
    len: usize,
    coset_offset: &FieldElement<Goldilocks64Field>,
    coset_state: &CosetShiftState,
    metal_state: &MetalState,
) -> Result<metal::Buffer, MetalError> {
    type FpE = FieldElement<Goldilocks64Field>;

    let order = len.trailing_zeros() as u64;

    let inv_twiddles = gen_twiddles_to_buffer::<Goldilocks64Field>(
        order,
        RootsConfig::BitReverseInversed,
        metal_state,
    )?;

    let result_buffer =
        fft_buffer_to_buffer::<Goldilocks64Field>(eval_buffer, len, &inv_twiddles, metal_state)?;

    let n_inv = FpE::from(len as u64)
        .inv()
        .expect("Power-of-two length is always invertible in an FFT field");
    let normalized = gpu_scale_buffer(&result_buffer, len, &n_inv, coset_state)?;

    let offset_inv = coset_offset.inv().expect("Coset offset must be invertible");
    gpu_coset_shift_buffer_to_buffer(&normalized, len, &offset_inv, len, coset_state)
}

/// Fused IFFT -> scale -> coset shift -> FFT in a single Metal command buffer.
/// Eliminates 3 command buffer submissions per column (4 stages, 1 commit).
#[cfg(all(target_os = "macos", feature = "metal"))]
#[allow(clippy::too_many_arguments)]
fn gpu_lde_column_fused(
    eval_buffer: &metal::Buffer,
    trace_len: usize,
    domain_size: usize,
    inv_twiddles: &metal::Buffer,
    fwd_twiddles: &metal::Buffer,
    n_inv: &FieldElement<Goldilocks64Field>,
    coset_offset: &FieldElement<Goldilocks64Field>,
    working_buffer: &metal::Buffer,
    coset_state: &CosetShiftState,
    metal_state: &MetalState,
) -> Result<metal::Buffer, MetalError> {
    use metal::MTLSize;

    let coeff_buffer = metal_state.alloc_buffer::<u64>(trace_len);
    let shifted_buffer = coset_state
        .state
        .alloc_buffer(domain_size * std::mem::size_of::<u64>())?;
    let lde_buffer = metal_state.alloc_buffer::<u64>(domain_size);

    let buf_scalar = alloc_u64(coset_state, canonical(n_inv))?;
    let buf_scale_len = alloc_u32(coset_state, trace_len as u32)?;
    let buf_offset = alloc_u64(coset_state, canonical(coset_offset))?;
    let buf_input_len = alloc_u32(coset_state, trace_len as u32)?;
    let buf_output_len = alloc_u32(coset_state, domain_size as u32)?;

    objc::rc::autoreleasepool(|| {
        let command_buffer = metal_state.queue.new_command_buffer();

        // Stage 1: IFFT (eval_buffer -> working -> coeff_buffer)
        fft_encode_to_command_buffer::<Goldilocks64Field>(
            command_buffer,
            eval_buffer,
            trace_len,
            inv_twiddles,
            working_buffer,
            &coeff_buffer,
            metal_state,
        )?;

        // Stage 2: Scale by 1/N (coeff_buffer in-place)
        {
            let scale_pipeline = coset_state
                .state
                .get_pipeline_ref("goldilocks_scale")
                .ok_or_else(|| MetalError::FunctionError("goldilocks_scale".to_string()))?;

            let threads_per_group = coset_state.scale_max_threads.min(256);
            let thread_groups = (trace_len as u64).div_ceil(threads_per_group);

            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(scale_pipeline);
            encoder.set_buffer(0, Some(&coeff_buffer), 0);
            encoder.set_buffer(1, Some(&coeff_buffer), 0);
            encoder.set_buffer(2, Some(&buf_scalar), 0);
            encoder.set_buffer(3, Some(&buf_scale_len), 0);
            encoder.dispatch_thread_groups(
                MTLSize::new(thread_groups, 1, 1),
                MTLSize::new(threads_per_group, 1, 1),
            );
            encoder.end_encoding();
        }

        // Stage 3: Coset shift + zero-pad (coeff_buffer -> shifted_buffer)
        {
            let coset_pipeline = coset_state
                .state
                .get_pipeline_ref("goldilocks_coset_shift")
                .ok_or_else(|| MetalError::FunctionError("goldilocks_coset_shift".to_string()))?;

            let threads_per_group = coset_state.coset_shift_max_threads.min(256);
            let thread_groups = (domain_size as u64).div_ceil(threads_per_group);

            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(coset_pipeline);
            encoder.set_buffer(0, Some(&coeff_buffer), 0);
            encoder.set_buffer(1, Some(&shifted_buffer), 0);
            encoder.set_buffer(2, Some(&buf_offset), 0);
            encoder.set_buffer(3, Some(&buf_input_len), 0);
            encoder.set_buffer(4, Some(&buf_output_len), 0);
            encoder.dispatch_thread_groups(
                MTLSize::new(thread_groups, 1, 1),
                MTLSize::new(threads_per_group, 1, 1),
            );
            encoder.end_encoding();
        }

        // Stage 4: FFT (shifted_buffer -> working -> lde_buffer)
        fft_encode_to_command_buffer::<Goldilocks64Field>(
            command_buffer,
            &shifted_buffer,
            domain_size,
            fwd_twiddles,
            working_buffer,
            &lde_buffer,
            metal_state,
        )?;

        command_buffer.commit();
        command_buffer.wait_until_completed();

        if command_buffer.status() == metal::MTLCommandBufferStatus::Error {
            return Err(MetalError::ExecutionError(
                "GPU fused LDE command buffer error".to_string(),
            ));
        }

        Ok::<(), MetalError>(())
    })?;

    Ok(lde_buffer)
}

/// Fused IFFT -> scale -> decoset -> trim -> coset -> FFT pipeline for composition polynomial.
///
/// When `number_of_parts == 1`, the entire composition polynomial interpolation
/// and LDE runs in a single Metal command buffer (6 stages, 1 commit).
///
/// Returns `(coeffs_buffer, vec![lde_buffer], lde_domain_size)`.
#[cfg(all(target_os = "macos", feature = "metal"))]
#[allow(clippy::too_many_arguments)]
pub fn gpu_composition_ifft_and_lde_fused(
    eval_buffer: &metal::Buffer,
    eval_len: usize,
    meaningful_coeffs: usize,
    blowup_factor: usize,
    coset_offset: &FieldElement<Goldilocks64Field>,
    coset_state: &CosetShiftState,
    metal_state: &MetalState,
) -> Result<(metal::Buffer, Vec<metal::Buffer>, usize), MetalError> {
    use metal::MTLSize;

    type F = Goldilocks64Field;
    type FpE = FieldElement<F>;

    let lde_domain_size = meaningful_coeffs * blowup_factor;
    let ifft_order = eval_len.trailing_zeros() as u64;
    let fft_order = lde_domain_size.trailing_zeros() as u64;

    let inv_twiddles =
        gen_twiddles_to_buffer::<F>(ifft_order, RootsConfig::BitReverseInversed, metal_state)?;
    let fwd_twiddles =
        gen_twiddles_to_buffer::<F>(fft_order, RootsConfig::BitReverse, metal_state)?;

    let n_inv = FpE::from(eval_len as u64)
        .inv()
        .expect("Power-of-two length is always invertible in an FFT field");
    let offset_inv = coset_offset.inv().expect("Coset offset must be invertible");

    let buf_n_inv = alloc_u64(coset_state, canonical(&n_inv))?;
    let buf_eval_len = alloc_u32(coset_state, eval_len as u32)?;
    let buf_offset_inv = alloc_u64(coset_state, canonical(&offset_inv))?;
    let buf_offset_fwd = alloc_u64(coset_state, canonical(coset_offset))?;
    let buf_meaningful_len = alloc_u32(coset_state, meaningful_coeffs as u32)?;
    let buf_lde_len = alloc_u32(coset_state, lde_domain_size as u32)?;

    let ifft_buffer = metal_state.alloc_buffer::<u64>(eval_len);
    let coeffs_buffer = coset_state
        .state
        .alloc_buffer(eval_len * std::mem::size_of::<u64>())?;
    let trimmed_buffer = coset_state
        .state
        .alloc_buffer(meaningful_coeffs * std::mem::size_of::<u64>())?;
    let shifted_buffer = coset_state
        .state
        .alloc_buffer(lde_domain_size * std::mem::size_of::<u64>())?;
    let lde_buffer = metal_state.alloc_buffer::<u64>(lde_domain_size);
    let working_size = eval_len.max(lde_domain_size);
    let working_buffer = metal_state.alloc_buffer::<u64>(working_size);

    let scale_pipeline = coset_state
        .state
        .get_pipeline_ref("goldilocks_scale")
        .ok_or_else(|| MetalError::FunctionError("goldilocks_scale".to_string()))?
        .to_owned();
    let coset_pipeline = coset_state
        .state
        .get_pipeline_ref("goldilocks_coset_shift")
        .ok_or_else(|| MetalError::FunctionError("goldilocks_coset_shift".to_string()))?
        .to_owned();

    objc::rc::autoreleasepool(|| {
        let command_buffer = metal_state.queue.new_command_buffer();

        // Stage 1: IFFT
        fft_encode_to_command_buffer::<F>(
            command_buffer,
            eval_buffer,
            eval_len,
            &inv_twiddles,
            &working_buffer,
            &ifft_buffer,
            metal_state,
        )?;

        // Stage 2: Scale by 1/N (in-place)
        {
            let threads_per_group = coset_state.scale_max_threads.min(256);
            let thread_groups = (eval_len as u64).div_ceil(threads_per_group);
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&scale_pipeline);
            encoder.set_buffer(0, Some(&ifft_buffer), 0);
            encoder.set_buffer(1, Some(&ifft_buffer), 0);
            encoder.set_buffer(2, Some(&buf_n_inv), 0);
            encoder.set_buffer(3, Some(&buf_eval_len), 0);
            encoder.dispatch_thread_groups(
                MTLSize::new(thread_groups, 1, 1),
                MTLSize::new(threads_per_group, 1, 1),
            );
            encoder.end_encoding();
        }

        // Stage 3: Inverse coset shift
        {
            let threads_per_group = coset_state.coset_shift_max_threads.min(256);
            let thread_groups = (eval_len as u64).div_ceil(threads_per_group);
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&coset_pipeline);
            encoder.set_buffer(0, Some(&ifft_buffer), 0);
            encoder.set_buffer(1, Some(&coeffs_buffer), 0);
            encoder.set_buffer(2, Some(&buf_offset_inv), 0);
            encoder.set_buffer(3, Some(&buf_eval_len), 0);
            encoder.set_buffer(4, Some(&buf_eval_len), 0);
            encoder.dispatch_thread_groups(
                MTLSize::new(thread_groups, 1, 1),
                MTLSize::new(threads_per_group, 1, 1),
            );
            encoder.end_encoding();
        }

        // Stage 4: Trim via blit
        {
            let blit = command_buffer.new_blit_command_encoder();
            blit.copy_from_buffer(
                &coeffs_buffer,
                0,
                &trimmed_buffer,
                0,
                (meaningful_coeffs * std::mem::size_of::<u64>()) as u64,
            );
            blit.end_encoding();
        }

        // Stage 5: Forward coset shift + zero-pad
        {
            let threads_per_group = coset_state.coset_shift_max_threads.min(256);
            let thread_groups = (lde_domain_size as u64).div_ceil(threads_per_group);
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&coset_pipeline);
            encoder.set_buffer(0, Some(&trimmed_buffer), 0);
            encoder.set_buffer(1, Some(&shifted_buffer), 0);
            encoder.set_buffer(2, Some(&buf_offset_fwd), 0);
            encoder.set_buffer(3, Some(&buf_meaningful_len), 0);
            encoder.set_buffer(4, Some(&buf_lde_len), 0);
            encoder.dispatch_thread_groups(
                MTLSize::new(thread_groups, 1, 1),
                MTLSize::new(threads_per_group, 1, 1),
            );
            encoder.end_encoding();
        }

        // Stage 6: Forward FFT
        fft_encode_to_command_buffer::<F>(
            command_buffer,
            &shifted_buffer,
            lde_domain_size,
            &fwd_twiddles,
            &working_buffer,
            &lde_buffer,
            metal_state,
        )?;

        command_buffer.commit();
        command_buffer.wait_until_completed();

        if command_buffer.status() == metal::MTLCommandBufferStatus::Error {
            return Err(MetalError::ExecutionError(
                "GPU fused composition IFFT+LDE command buffer error".to_string(),
            ));
        }

        Ok::<(), MetalError>(())
    })?;

    Ok((coeffs_buffer, vec![lde_buffer], lde_domain_size))
}

/// Computes LDE directly from trace evaluations, fusing IFFT + coset LDE per column.
///
/// For each column: upload -> IFFT -> 1/N -> coset shift + zero-pad -> FFT.
/// Generates both inverse and forward twiddles once and reuses a single working buffer.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn gpu_lde_from_evaluations(
    columns: &[Vec<FieldElement<Goldilocks64Field>>],
    blowup_factor: usize,
    coset_offset: &FieldElement<Goldilocks64Field>,
    coset_state: &CosetShiftState,
    metal_state: &MetalState,
) -> Result<(Vec<metal::Buffer>, usize), MetalError> {
    type FpE = FieldElement<Goldilocks64Field>;

    if columns.is_empty() {
        return Ok((Vec::new(), 0));
    }

    let trace_len = columns[0].len();
    let domain_size = trace_len * blowup_factor;
    let ifft_order = trace_len.trailing_zeros() as u64;
    let fft_order = domain_size.trailing_zeros() as u64;

    let inv_twiddles = gen_twiddles_to_buffer::<Goldilocks64Field>(
        ifft_order,
        RootsConfig::BitReverseInversed,
        metal_state,
    )?;
    let fwd_twiddles = gen_twiddles_to_buffer::<Goldilocks64Field>(
        fft_order,
        RootsConfig::BitReverse,
        metal_state,
    )?;

    let n_inv = FpE::from(trace_len as u64)
        .inv()
        .expect("Power-of-two length is always invertible in an FFT field");

    let working_buffer = metal_state.alloc_buffer::<u64>(domain_size);

    let mut lde_buffers = Vec::with_capacity(columns.len());
    for col in columns {
        let evals_u64 = to_raw_u64(col);
        let eval_buffer = metal_state.alloc_buffer_data(&evals_u64);

        let lde_buffer = gpu_lde_column_fused(
            &eval_buffer,
            trace_len,
            domain_size,
            &inv_twiddles,
            &fwd_twiddles,
            &n_inv,
            coset_offset,
            &working_buffer,
            coset_state,
            metal_state,
        )?;

        lde_buffers.push(lde_buffer);
    }

    Ok((lde_buffers, domain_size))
}

/// Embedded Metal shader source for the coefficient striding kernel.
#[cfg(all(target_os = "macos", feature = "metal"))]
const STRIDE_SHADER: &str = include_str!("shaders/stride.metal");

#[cfg(all(target_os = "macos", feature = "metal"))]
#[repr(C)]
#[derive(Copy, Clone)]
struct StrideParams {
    num_coeffs: u32,
    num_parts: u32,
}

/// GPU coefficient striding: breaks a coefficient buffer into `num_parts` sub-buffers
/// where part `i` contains coefficients at indices `[i, i+k, i+2k, ...]`.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn gpu_break_in_parts_buffer_to_buffers(
    coeffs_buffer: &metal::Buffer,
    num_coeffs: usize,
    num_parts: usize,
    state: &MetalState,
) -> Result<Vec<metal::Buffer>, MetalError> {
    assert!(
        num_parts > 0 && num_coeffs.is_multiple_of(num_parts),
        "num_coeffs ({num_coeffs}) must be divisible by num_parts ({num_parts})"
    );

    let part_len = num_coeffs / num_parts;
    let params = StrideParams {
        num_coeffs: num_coeffs as u32,
        num_parts: num_parts as u32,
    };

    let mut dyn_state = DynamicMetalState::new()?;
    dyn_state.load_library(STRIDE_SHADER)?;
    let max_threads = dyn_state.prepare_pipeline("goldilocks_stride_coefficients")?;

    let buf_params = dyn_state.alloc_buffer_with_data(std::slice::from_ref(&params))?;
    let buf_output = dyn_state.alloc_buffer(num_coeffs * std::mem::size_of::<u64>())?;

    dyn_state.execute_compute(
        "goldilocks_stride_coefficients",
        &[coeffs_buffer, &buf_output, &buf_params],
        num_coeffs as u64,
        max_threads,
    )?;

    // Split concatenated output into per-part buffers via Metal blit (stays on GPU).
    let mut part_buffers = Vec::with_capacity(num_parts);
    for _ in 0..num_parts {
        part_buffers.push(state.alloc_buffer_data(&vec![0u64; part_len]));
    }

    let command_buffer = state.queue.new_command_buffer();
    let blit_encoder = command_buffer.new_blit_command_encoder();
    for (i, part_buf) in part_buffers.iter().enumerate() {
        let src_offset = (i * part_len * std::mem::size_of::<u64>()) as u64;
        let size = (part_len * std::mem::size_of::<u64>()) as u64;
        blit_encoder.copy_from_buffer(&buf_output, src_offset, part_buf, 0, size);
    }
    blit_encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    Ok(part_buffers)
}

#[cfg(all(test, target_os = "macos", feature = "metal"))]
mod tests {
    use super::*;
    use lambdaworks_math::field::element::FieldElement;
    use lambdaworks_math::field::fields::u64_goldilocks_field::Goldilocks64Field;
    use lambdaworks_math::polynomial::Polynomial;

    type FpE = FieldElement<Goldilocks64Field>;

    #[test]
    fn gpu_fft_interpolation_matches_cpu() {
        let state = crate::metal::state::StarkMetalState::new().unwrap();
        let values: Vec<FpE> = (0..8).map(|i| FpE::from(i as u64 + 1)).collect();

        let cpu_poly = Polynomial::interpolate_fft::<Goldilocks64Field>(&values).unwrap();
        let gpu_coeffs = gpu_interpolate_fft::<Goldilocks64Field>(&values, state.inner()).unwrap();

        assert_eq!(cpu_poly.coefficients().len(), gpu_coeffs.len());
        for (cpu, gpu) in cpu_poly.coefficients().iter().zip(&gpu_coeffs) {
            assert_eq!(cpu, gpu, "FFT coefficient mismatch");
        }
    }

    #[test]
    fn gpu_fft_evaluate_offset_matches_cpu() {
        let state = crate::metal::state::StarkMetalState::new().unwrap();
        let coeffs: Vec<FpE> = (0..8).map(|i| FpE::from(i as u64 + 1)).collect();
        let poly = Polynomial::new(&coeffs);
        let offset = FpE::from(7u64);
        let blowup_factor = 4;

        let cpu_evals = Polynomial::evaluate_offset_fft::<Goldilocks64Field>(
            &poly,
            blowup_factor,
            None,
            &offset,
        )
        .unwrap();

        let gpu_evals = gpu_evaluate_offset_fft::<Goldilocks64Field>(
            &coeffs,
            blowup_factor,
            &offset,
            state.inner(),
        )
        .unwrap();

        assert_eq!(cpu_evals.len(), gpu_evals.len());
        for (cpu, gpu) in cpu_evals.iter().zip(&gpu_evals) {
            assert_eq!(cpu, gpu, "Offset FFT evaluation mismatch");
        }
    }

    #[test]
    fn gpu_fft_roundtrip() {
        let state = crate::metal::state::StarkMetalState::new().unwrap();
        let original_coeffs: Vec<FpE> = (0..16).map(|i| FpE::from(i as u64 + 1)).collect();
        let poly = Polynomial::new(&original_coeffs);

        let evals = Polynomial::evaluate_fft::<Goldilocks64Field>(&poly, 1, None).unwrap();
        let recovered_coeffs =
            gpu_interpolate_fft::<Goldilocks64Field>(&evals, state.inner()).unwrap();

        assert_eq!(original_coeffs.len(), recovered_coeffs.len());
        for (orig, recov) in original_coeffs.iter().zip(&recovered_coeffs) {
            assert_eq!(orig, recov, "Roundtrip coefficient mismatch");
        }
    }

    #[test]
    fn gpu_coset_shift_matches_cpu() {
        let coset_state = CosetShiftState::new().unwrap();
        let coeffs: Vec<FpE> = (0..16).map(|i| FpE::from(i as u64 + 1)).collect();
        let offset = FpE::from(7u64);
        let output_len = coeffs.len() * 4;

        let mut cpu_shifted = Vec::with_capacity(output_len);
        let mut offset_power = FpE::one();
        for coeff in &coeffs {
            cpu_shifted.push(coeff * offset_power);
            offset_power *= offset;
        }
        cpu_shifted.resize(output_len, FpE::zero());

        let gpu_buffer =
            gpu_coset_shift_to_buffer(&coeffs, &offset, output_len, &coset_state).unwrap();
        let gpu_u64: Vec<u64> = unsafe { coset_state.state.read_buffer(&gpu_buffer, output_len) };
        let gpu_shifted: Vec<FpE> = gpu_u64.iter().map(|&v| FpE::from(v)).collect();

        assert_eq!(cpu_shifted.len(), gpu_shifted.len());
        for (i, (cpu, gpu)) in cpu_shifted.iter().zip(&gpu_shifted).enumerate() {
            assert_eq!(cpu, gpu, "Coset shift mismatch at index {i}");
        }
    }

    #[test]
    fn gpu_coset_shift_matches_cpu_large() {
        let coset_state = CosetShiftState::new().unwrap();
        let coeffs: Vec<FpE> = (0..256).map(|i| FpE::from(i as u64 * 31 + 17)).collect();
        let offset = FpE::from(42u64);
        let output_len = coeffs.len() * 4;

        let mut cpu_shifted = Vec::with_capacity(output_len);
        let mut offset_power = FpE::one();
        for coeff in &coeffs {
            cpu_shifted.push(coeff * offset_power);
            offset_power *= offset;
        }
        cpu_shifted.resize(output_len, FpE::zero());

        let gpu_buffer =
            gpu_coset_shift_to_buffer(&coeffs, &offset, output_len, &coset_state).unwrap();
        let gpu_u64: Vec<u64> = unsafe { coset_state.state.read_buffer(&gpu_buffer, output_len) };
        let gpu_shifted: Vec<FpE> = gpu_u64.iter().map(|&v| FpE::from(v)).collect();

        assert_eq!(cpu_shifted.len(), gpu_shifted.len());
        for (i, (cpu, gpu)) in cpu_shifted.iter().zip(&gpu_shifted).enumerate() {
            assert_eq!(cpu, gpu, "Coset shift mismatch at index {i}");
        }
    }

    #[test]
    fn gpu_scale_matches_cpu() {
        let coset_state = CosetShiftState::new().unwrap();
        let values: Vec<FpE> = (0..32).map(|i| FpE::from(i as u64 + 1)).collect();
        let scalar = FpE::from(13u64);

        let cpu_scaled: Vec<FpE> = values.iter().map(|v| v * scalar).collect();

        let values_u64 = to_raw_u64(&values);
        let input_buffer = coset_state
            .state
            .alloc_buffer_with_data(&values_u64)
            .unwrap();
        let gpu_buffer =
            gpu_scale_buffer(&input_buffer, values.len(), &scalar, &coset_state).unwrap();

        let gpu_u64: Vec<u64> = unsafe { coset_state.state.read_buffer(&gpu_buffer, values.len()) };
        let gpu_scaled: Vec<FpE> = gpu_u64.iter().map(|&v| FpE::from(v)).collect();

        assert_eq!(cpu_scaled.len(), gpu_scaled.len());
        for (i, (cpu, gpu)) in cpu_scaled.iter().zip(&gpu_scaled).enumerate() {
            assert_eq!(cpu, gpu, "Scale mismatch at index {i}");
        }
    }

    #[test]
    fn gpu_scale_matches_cpu_large() {
        let coset_state = CosetShiftState::new().unwrap();
        let values: Vec<FpE> = (0..1024).map(|i| FpE::from(i as u64 * 7 + 3)).collect();
        let scalar = FpE::from(999u64);

        let cpu_scaled: Vec<FpE> = values.iter().map(|v| v * scalar).collect();

        let values_u64 = to_raw_u64(&values);
        let input_buffer = coset_state
            .state
            .alloc_buffer_with_data(&values_u64)
            .unwrap();
        let gpu_buffer =
            gpu_scale_buffer(&input_buffer, values.len(), &scalar, &coset_state).unwrap();

        let gpu_u64: Vec<u64> = unsafe { coset_state.state.read_buffer(&gpu_buffer, values.len()) };
        let gpu_scaled: Vec<FpE> = gpu_u64.iter().map(|&v| FpE::from(v)).collect();

        assert_eq!(cpu_scaled.len(), gpu_scaled.len());
        for (i, (cpu, gpu)) in cpu_scaled.iter().zip(&gpu_scaled).enumerate() {
            assert_eq!(cpu, gpu, "Scale mismatch at index {i}");
        }
    }

    #[test]
    fn gpu_ifft_matches_cpu() {
        let metal_state = crate::metal::state::StarkMetalState::new().unwrap();
        let coset_state = CosetShiftState::new().unwrap();
        let values: Vec<FpE> = (0..16).map(|i| FpE::from(i as u64 + 1)).collect();

        let cpu_coeffs =
            gpu_interpolate_fft::<Goldilocks64Field>(&values, metal_state.inner()).unwrap();

        let gpu_buffer = gpu_ifft_to_buffer(&values, &coset_state, metal_state.inner()).unwrap();
        let gpu_u64: Vec<u64> = unsafe { coset_state.state.read_buffer(&gpu_buffer, values.len()) };
        let gpu_coeffs: Vec<FpE> = gpu_u64.iter().map(|&v| FpE::from(v)).collect();

        assert_eq!(cpu_coeffs.len(), gpu_coeffs.len());
        for (i, (cpu, gpu)) in cpu_coeffs.iter().zip(&gpu_coeffs).enumerate() {
            assert_eq!(cpu, gpu, "IFFT coefficient mismatch at index {i}");
        }
    }

    #[test]
    fn gpu_stride_matches_cpu_break_in_parts() {
        let state = crate::metal::state::StarkMetalState::new().expect("Metal init failed");

        let coeffs: Vec<FpE> = (0..16).map(|i| FpE::from(i as u64 + 100)).collect();
        let poly = Polynomial::new(&coeffs);
        let cpu_parts = poly.break_in_parts(4);

        let coeffs_u64 = to_raw_u64(&coeffs);
        let gpu_buffer = state.inner().alloc_buffer_data(&coeffs_u64);

        let gpu_part_buffers =
            gpu_break_in_parts_buffer_to_buffers(&gpu_buffer, 16, 4, state.inner())
                .expect("GPU stride failed");

        assert_eq!(gpu_part_buffers.len(), 4, "expected 4 parts");

        for (part_idx, (cpu_part, gpu_buf)) in cpu_parts.iter().zip(&gpu_part_buffers).enumerate() {
            let gpu_raw: Vec<u64> = MetalState::retrieve_contents(gpu_buf);
            let gpu_elements: Vec<FpE> = gpu_raw.into_iter().map(FieldElement::from).collect();
            let cpu_coeffs = cpu_part.coefficients();
            assert_eq!(
                cpu_coeffs.len(),
                gpu_elements.len(),
                "part {part_idx}: length mismatch"
            );
            for (i, (cpu_val, gpu_val)) in cpu_coeffs.iter().zip(&gpu_elements).enumerate() {
                assert_eq!(
                    cpu_val, gpu_val,
                    "part {part_idx}, index {i}: CPU={cpu_val:?} GPU={gpu_val:?}"
                );
            }
        }
    }

    #[test]
    fn gpu_stride_fft_matches_cpu_break_in_parts_fft() {
        let state = crate::metal::state::StarkMetalState::new().expect("Metal init failed");
        let coset_state = CosetShiftState::new().expect("CosetShiftState init failed");

        let coeffs: Vec<FpE> = (0..32).map(|i| FpE::from(i as u64 + 1)).collect();
        let poly = Polynomial::new(&coeffs);
        let offset = FpE::from(7u64);
        let blowup_factor = 4;
        let num_parts = 2;

        let cpu_parts = poly.break_in_parts(num_parts);
        let cpu_evals: Vec<Vec<FpE>> = cpu_parts
            .iter()
            .map(|p| {
                Polynomial::evaluate_offset_fft::<Goldilocks64Field>(
                    p,
                    blowup_factor,
                    None,
                    &offset,
                )
                .expect("CPU FFT failed")
            })
            .collect();

        let coeffs_u64 = to_raw_u64(&coeffs);
        let gpu_buffer = state.inner().alloc_buffer_data(&coeffs_u64);

        let part_buffers =
            gpu_break_in_parts_buffer_to_buffers(&gpu_buffer, 32, num_parts, state.inner())
                .expect("GPU stride failed");

        let part_len = 32 / num_parts;
        let buffer_results = gpu_evaluate_offset_fft_buffer_to_buffers_batch(
            &part_buffers,
            part_len,
            blowup_factor,
            &offset,
            &coset_state,
            state.inner(),
        )
        .expect("GPU buffer-to-buffer FFT failed");

        assert_eq!(buffer_results.len(), num_parts, "expected 2 FFT results");

        for (part_idx, ((gpu_buf, _domain_size), cpu_eval)) in
            buffer_results.iter().zip(&cpu_evals).enumerate()
        {
            let gpu_raw: Vec<u64> = MetalState::retrieve_contents(gpu_buf);
            let gpu_elements: Vec<FpE> = gpu_raw.into_iter().map(FieldElement::from_raw).collect();
            assert_eq!(
                cpu_eval.len(),
                gpu_elements.len(),
                "part {part_idx}: length mismatch"
            );
            for (i, (cpu_val, gpu_val)) in cpu_eval.iter().zip(&gpu_elements).enumerate() {
                assert_eq!(
                    cpu_val, gpu_val,
                    "part {part_idx}, index {i}: CPU={cpu_val:?} GPU={gpu_val:?}"
                );
            }
        }
    }

    #[test]
    fn gpu_interpolate_offset_fft_matches_cpu() {
        let metal_state = crate::metal::state::StarkMetalState::new().unwrap();
        let coset_state = CosetShiftState::new().unwrap();

        let original_coeffs: Vec<FpE> = (0..16).map(|i| FpE::from(i as u64 + 1)).collect();
        let offset = FpE::from(7u64);
        let blowup_factor = 1;

        let evals = gpu_evaluate_offset_fft::<Goldilocks64Field>(
            &original_coeffs,
            blowup_factor,
            &offset,
            metal_state.inner(),
        )
        .unwrap();

        let gpu_buffer = gpu_interpolate_offset_fft_to_buffer(
            &evals,
            &offset,
            &coset_state,
            metal_state.inner(),
        )
        .unwrap();

        let gpu_u64: Vec<u64> = unsafe {
            coset_state
                .state
                .read_buffer(&gpu_buffer, original_coeffs.len())
        };
        let recovered_coeffs: Vec<FpE> = gpu_u64.iter().map(|&v| FpE::from(v)).collect();

        assert_eq!(original_coeffs.len(), recovered_coeffs.len());
        for (i, (orig, recov)) in original_coeffs.iter().zip(&recovered_coeffs).enumerate() {
            assert_eq!(
                orig, recov,
                "Coset IFFT roundtrip mismatch at index {i}: original={:?} recovered={:?}",
                orig, recov
            );
        }
    }
}
