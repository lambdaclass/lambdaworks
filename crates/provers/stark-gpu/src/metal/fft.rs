//! Metal GPU FFT wrapper functions for the STARK prover.
//!
//! Provides high-level `gpu_interpolate_fft` and `gpu_evaluate_offset_fft` functions
//! that use the existing Metal FFT infrastructure from `lambdaworks-math` for
//! Goldilocks-compatible fields.

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
        traits::{IsFFTField, IsPrimeField, IsSubFieldOf, RootsConfig},
    },
};

/// Interpolates polynomial coefficients from evaluations using Metal GPU FFT.
///
/// Given `evaluations` at roots of unity, this computes the polynomial coefficients
/// via inverse FFT on the GPU. Equivalent to `Polynomial::interpolate_fft` but
/// executed on Metal.
///
/// # Algorithm
///
/// 1. Generate inverse twiddle factors with `RootsConfig::BitReverseInversed`
/// 2. Run forward FFT with inverse twiddles (this performs the inverse transform)
/// 3. Normalize each coefficient by multiplying by `1/n`
///
/// # Type Parameters
///
/// - `F`: An FFT-compatible field (must provide primitive roots of unity)
///
/// # Errors
///
/// Returns `MetalError` if GPU twiddle generation or FFT execution fails.
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

    // Normalize by 1/n
    let n_inv = FieldElement::<F>::from(evaluations.len() as u64)
        .inv()
        .expect("Power-of-two length is always invertible in an FFT field");

    Ok(coeffs.iter().map(|c| c * &n_inv).collect())
}

/// Evaluates a polynomial on an offset coset domain using Metal GPU FFT.
///
/// Given polynomial `coefficients`, this computes evaluations on the domain
/// `{offset * w^i}` where `w` is a primitive root of unity of order
/// `coefficients.len() * blowup_factor`. This is the Low-Degree Extension (LDE)
/// operation used in STARK provers.
///
/// # Algorithm
///
/// 1. Multiply coefficient `k` by `offset^k` (coset shift)
/// 2. Zero-pad to `domain_size = len * blowup_factor`
/// 3. Generate forward twiddle factors with `RootsConfig::BitReverse`
/// 4. Run forward FFT on the GPU
///
/// # Type Parameters
///
/// - `F`: An FFT-compatible field (must provide primitive roots of unity)
///
/// # Arguments
///
/// - `coefficients`: The polynomial coefficients
/// - `blowup_factor`: The LDE blowup factor (must be a power of two)
/// - `offset`: The coset offset element
/// - `state`: Metal GPU state
///
/// # Errors
///
/// Returns `MetalError` if GPU twiddle generation or FFT execution fails.
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
    let domain_size = coefficients.len() * blowup_factor;

    // Step 1: Multiply coefficient k by offset^k (coset shift)
    let mut shifted = Vec::with_capacity(domain_size);
    let mut offset_power = FieldElement::<F>::one();
    for coeff in coefficients {
        shifted.push(coeff * &offset_power);
        offset_power = &offset_power * offset;
    }

    // Step 2: Zero-pad to domain_size
    shifted.resize(domain_size, FieldElement::zero());

    // Step 3 & 4: Generate forward twiddles and run FFT
    let order = domain_size.trailing_zeros() as u64;
    let twiddles = gen_twiddles::<F>(order, RootsConfig::BitReverse, state)?;
    fft(&shifted, &twiddles, state)
}

/// Evaluates a polynomial on an offset coset domain, returning a GPU Metal Buffer.
///
/// Like [`gpu_evaluate_offset_fft`] but keeps the result on GPU for direct use by
/// downstream GPU operations (e.g., Merkle tree hashing) without CPU readback.
///
/// The returned buffer contains `F::BaseType` elements in bit-reversed order
/// (natural FFT output order after bit-reverse permutation).
///
/// # Returns
///
/// A tuple of (Metal Buffer, element count) where the buffer contains the
/// bit-reversed FFT evaluations.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn gpu_evaluate_offset_fft_to_buffer<F>(
    coefficients: &[FieldElement<F>],
    blowup_factor: usize,
    offset: &FieldElement<F>,
    state: &MetalState,
) -> Result<(metal::Buffer, usize), MetalError>
where
    F: IsFFTField + IsSubFieldOf<F>,
    F::BaseType: Copy,
{
    let domain_size = coefficients.len() * blowup_factor;

    // Step 1: Multiply coefficient k by offset^k (coset shift)
    let mut shifted = Vec::with_capacity(domain_size);
    let mut offset_power = FieldElement::<F>::one();
    for coeff in coefficients {
        shifted.push(coeff * &offset_power);
        offset_power = &offset_power * offset;
    }

    // Step 2: Zero-pad to domain_size
    shifted.resize(domain_size, FieldElement::zero());

    // Step 3: Generate twiddles directly as GPU buffer (no CPU download)
    let order = domain_size.trailing_zeros() as u64;
    let twiddles_buffer = gen_twiddles_to_buffer::<F>(order, RootsConfig::BitReverse, state)?;

    // Step 4: FFT with result staying on GPU
    let result_buffer = fft_to_buffer::<F>(&shifted, &twiddles_buffer, state)?;
    Ok((result_buffer, domain_size))
}

/// Batch-evaluates multiple polynomials on the same offset coset domain, returning GPU Buffers.
///
/// Like calling [`gpu_evaluate_offset_fft_to_buffer`] for each polynomial but generates
/// twiddle factors only once and reuses them for all polynomials in the batch.
///
/// Returns a vector of (Metal Buffer, element count) pairs, one per polynomial.
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

    // Generate twiddles ONCE for the shared domain
    let twiddles_buffer = gen_twiddles_to_buffer::<F>(order, RootsConfig::BitReverse, state)?;

    let mut results = Vec::with_capacity(polynomials.len());

    for coefficients in polynomials {
        let poly_domain_size = coefficients.len() * blowup_factor;

        // Coset shift: multiply coefficient k by offset^k
        let mut shifted = Vec::with_capacity(poly_domain_size);
        let mut offset_power = FieldElement::<F>::one();
        for coeff in *coefficients {
            shifted.push(coeff * &offset_power);
            offset_power = &offset_power * offset;
        }
        shifted.resize(poly_domain_size, FieldElement::zero());

        // FFT with shared twiddles, result stays on GPU
        let result_buffer = fft_to_buffer::<F>(&shifted, &twiddles_buffer, state)?;
        results.push((result_buffer, poly_domain_size));
    }

    Ok(results)
}

/// Batch-evaluates multiple polynomials from GPU coefficient buffers on a coset domain.
///
/// Like [`gpu_evaluate_offset_fft_to_buffers_batch`] but reads coefficients from existing
/// Metal Buffers (e.g., produced by [`gpu_break_in_parts_buffer_to_buffers`]) instead of
/// CPU slices. The entire pipeline stays on GPU: coset shift → zero-pad → FFT.
///
/// # Arguments
///
/// - `coeff_buffers`: GPU buffers, each containing `part_len` Goldilocks u64 coefficients
/// - `part_len`: Number of coefficients per part
/// - `blowup_factor`: LDE blowup factor (domain_size = part_len * blowup_factor)
/// - `offset`: Coset offset element
/// - `coset_state`: Pre-compiled coset shift Metal state
/// - `metal_state`: Metal state for FFT
///
/// # Returns
///
/// A vector of (Metal Buffer, element count) pairs, one per polynomial.
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

    // Generate twiddles ONCE for the shared domain
    let twiddles_buffer =
        gen_twiddles_to_buffer::<Goldilocks64Field>(order, RootsConfig::BitReverse, metal_state)?;

    // Pre-allocate a single working buffer for all FFT calls (reused across iterations)
    let working_buffer = metal_state.alloc_buffer::<u64>(domain_size);

    let mut results = Vec::with_capacity(coeff_buffers.len());

    for coeff_buf in coeff_buffers {
        // Step 1: GPU coset shift + zero-pad to domain_size
        let shifted_buffer = gpu_coset_shift_buffer_to_buffer(
            coeff_buf,
            part_len,
            offset,
            domain_size,
            coset_state,
        )?;

        // Step 2: FFT with shared twiddles and reused working buffer
        let result_buffer = fft_buffer_to_buffer_reuse::<Goldilocks64Field>(
            &shifted_buffer,
            domain_size,
            &twiddles_buffer,
            &working_buffer,
            metal_state,
        )?;
        drop(shifted_buffer); // Free coset-shifted buffer before next iteration

        results.push((result_buffer, domain_size));
    }

    Ok(results)
}

// =============================================================================
// GPU Coset Shift + Scale kernels (Goldilocks-specific)
// =============================================================================

/// Source code for the Goldilocks field header (fp_u64.h.metal).
/// Prepended to the coset shift shader at runtime to resolve the Fp64Goldilocks class.
#[cfg(all(target_os = "macos", feature = "metal"))]
const GOLDILOCKS_FIELD_HEADER: &str =
    include_str!("../../../../math/src/gpu/metal/shaders/field/fp_u64.h.metal");

/// Source code for the coset shift and scale Metal kernels.
#[cfg(all(target_os = "macos", feature = "metal"))]
const COSET_SHIFT_SHADER: &str = include_str!("shaders/coset_shift.metal");

/// Pre-compiled Metal state for coset shift and scale operations.
///
/// Caches compiled pipelines for `goldilocks_coset_shift` and `goldilocks_scale`
/// kernels. Create once and reuse across the entire prove call.
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
        // Concatenate the Goldilocks field header with the coset shift shader
        // since runtime compilation via new_library_with_source does not support
        // file-system #include directives.
        let combined_source = format!("{}\n{}", GOLDILOCKS_FIELD_HEADER, COSET_SHIFT_SHADER);

        let mut state = DynamicMetalState::new()?;
        state.load_library(&combined_source)?;
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

    /// Compile the coset shift shader sharing a device and queue with an existing Metal state.
    ///
    /// This avoids creating new Metal device/queue pairs, reducing GPU resource usage
    /// when many shader states coexist.
    pub fn from_device_and_queue(
        device: &metal::Device,
        queue: &metal::CommandQueue,
    ) -> Result<Self, MetalError> {
        let combined_source = format!("{}\n{}", GOLDILOCKS_FIELD_HEADER, COSET_SHIFT_SHADER);

        let mut state = DynamicMetalState::from_device_and_queue(device, queue);
        state.load_library(&combined_source)?;
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

/// Perform coset shift on GPU: output[k] = coeffs[k] * offset^k, zero-padded to output_len.
///
/// Uploads coefficients to the GPU, dispatches the `goldilocks_coset_shift` kernel,
/// and returns the result as a Metal Buffer containing `output_len` Goldilocks u64 values.
///
/// This replaces the CPU loop that multiplies each coefficient by successive powers
/// of the offset, moving the work to the GPU for parallelism.
///
/// # Arguments
///
/// - `coeffs`: Polynomial coefficients in canonical Goldilocks form
/// - `offset`: The coset offset element
/// - `output_len`: Total output length (includes zero-padding for blowup factor)
/// - `state`: Pre-compiled coset shift Metal state
///
/// # Returns
///
/// A Metal Buffer containing `output_len` u64 values (coset-shifted + zero-padded).
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn gpu_coset_shift_to_buffer(
    coeffs: &[FieldElement<Goldilocks64Field>],
    offset: &FieldElement<Goldilocks64Field>,
    output_len: usize,
    coset_state: &CosetShiftState,
) -> Result<metal::Buffer, MetalError> {
    use metal::MTLSize;

    let input_len = coeffs.len();

    // Convert coefficients to canonical u64 representation for GPU upload
    let coeffs_u64: Vec<u64> = coeffs
        .iter()
        .map(|fe| Goldilocks64Field::canonical(fe.value()))
        .collect();
    let offset_u64 = Goldilocks64Field::canonical(offset.value());

    // Allocate GPU buffers
    let buf_input = coset_state.state.alloc_buffer_with_data(&coeffs_u64)?;
    let buf_output = coset_state
        .state
        .alloc_buffer(output_len * std::mem::size_of::<u64>())?;
    let buf_offset = coset_state
        .state
        .alloc_buffer_with_data(std::slice::from_ref(&offset_u64))?;
    let input_len_u32 = input_len as u32;
    let output_len_u32 = output_len as u32;
    let buf_input_len = coset_state
        .state
        .alloc_buffer_with_data(std::slice::from_ref(&input_len_u32))?;
    let buf_output_len = coset_state
        .state
        .alloc_buffer_with_data(std::slice::from_ref(&output_len_u32))?;

    // Dispatch the coset shift kernel
    let pipeline = coset_state
        .state
        .get_pipeline_ref("goldilocks_coset_shift")
        .ok_or_else(|| MetalError::FunctionError("goldilocks_coset_shift".to_string()))?;

    let threads_per_group = coset_state.coset_shift_max_threads.min(256);
    let thread_groups = (output_len as u64).div_ceil(threads_per_group);

    let command_buffer = coset_state.state.command_queue().new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(&buf_input), 0);
    encoder.set_buffer(1, Some(&buf_output), 0);
    encoder.set_buffer(2, Some(&buf_offset), 0);
    encoder.set_buffer(3, Some(&buf_input_len), 0);
    encoder.set_buffer(4, Some(&buf_output_len), 0);

    encoder.dispatch_thread_groups(
        MTLSize::new(thread_groups, 1, 1),
        MTLSize::new(threads_per_group, 1, 1),
    );
    encoder.end_encoding();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    if command_buffer.status() == metal::MTLCommandBufferStatus::Error {
        return Err(MetalError::ExecutionError(
            "GPU coset shift command buffer error".to_string(),
        ));
    }

    Ok(buf_output)
}

/// Scale all elements of a GPU buffer: output[k] = input[k] * scalar.
///
/// Dispatches the `goldilocks_scale` kernel on an existing GPU buffer and
/// returns a new buffer with the scaled values.
///
/// # Arguments
///
/// - `buffer`: Input Metal Buffer containing `len` Goldilocks u64 values
/// - `len`: Number of elements in the buffer
/// - `scalar`: The scalar to multiply by
/// - `state`: Pre-compiled coset shift Metal state
///
/// # Returns
///
/// A new Metal Buffer containing `len` u64 values (each scaled by `scalar`).
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn gpu_scale_buffer(
    buffer: &metal::Buffer,
    len: usize,
    scalar: &FieldElement<Goldilocks64Field>,
    coset_state: &CosetShiftState,
) -> Result<metal::Buffer, MetalError> {
    use metal::MTLSize;

    let scalar_u64 = Goldilocks64Field::canonical(scalar.value());

    // Allocate output buffer and parameter buffers
    let buf_output = coset_state
        .state
        .alloc_buffer(len * std::mem::size_of::<u64>())?;
    let buf_scalar = coset_state
        .state
        .alloc_buffer_with_data(std::slice::from_ref(&scalar_u64))?;
    let len_u32 = len as u32;
    let buf_len = coset_state
        .state
        .alloc_buffer_with_data(std::slice::from_ref(&len_u32))?;

    // Dispatch the scale kernel
    let pipeline = coset_state
        .state
        .get_pipeline_ref("goldilocks_scale")
        .ok_or_else(|| MetalError::FunctionError("goldilocks_scale".to_string()))?;

    let threads_per_group = coset_state.scale_max_threads.min(256);
    let thread_groups = (len as u64).div_ceil(threads_per_group);

    let command_buffer = coset_state.state.command_queue().new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(buffer), 0);
    encoder.set_buffer(1, Some(&buf_output), 0);
    encoder.set_buffer(2, Some(&buf_scalar), 0);
    encoder.set_buffer(3, Some(&buf_len), 0);

    encoder.dispatch_thread_groups(
        MTLSize::new(thread_groups, 1, 1),
        MTLSize::new(threads_per_group, 1, 1),
    );
    encoder.end_encoding();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    if command_buffer.status() == metal::MTLCommandBufferStatus::Error {
        return Err(MetalError::ExecutionError(
            "GPU scale command buffer error".to_string(),
        ));
    }

    Ok(buf_output)
}

/// Scale all elements of a GPU buffer in-place: buffer[k] = buffer[k] * scalar.
///
/// Like [`gpu_scale_buffer`] but overwrites the input buffer instead of allocating a new one.
/// This is safe because the `goldilocks_scale` kernel reads `input[gid]` into a local variable
/// before writing `output[gid]`, and each thread accesses a unique element.
///
/// Saves one buffer allocation (N×8B) per call.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn gpu_scale_buffer_inplace(
    buffer: &metal::Buffer,
    len: usize,
    scalar: &FieldElement<Goldilocks64Field>,
    coset_state: &CosetShiftState,
) -> Result<(), MetalError> {
    use metal::MTLSize;

    let scalar_u64 = Goldilocks64Field::canonical(scalar.value());

    let buf_scalar = coset_state
        .state
        .alloc_buffer_with_data(std::slice::from_ref(&scalar_u64))?;
    let len_u32 = len as u32;
    let buf_len = coset_state
        .state
        .alloc_buffer_with_data(std::slice::from_ref(&len_u32))?;

    let pipeline = coset_state
        .state
        .get_pipeline_ref("goldilocks_scale")
        .ok_or_else(|| MetalError::FunctionError("goldilocks_scale".to_string()))?;

    let threads_per_group = coset_state.scale_max_threads.min(256);
    let thread_groups = (len as u64).div_ceil(threads_per_group);

    let command_buffer = coset_state.state.command_queue().new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(buffer), 0); // input = output (same buffer)
    encoder.set_buffer(1, Some(buffer), 0);
    encoder.set_buffer(2, Some(&buf_scalar), 0);
    encoder.set_buffer(3, Some(&buf_len), 0);

    encoder.dispatch_thread_groups(
        MTLSize::new(thread_groups, 1, 1),
        MTLSize::new(threads_per_group, 1, 1),
    );
    encoder.end_encoding();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    if command_buffer.status() == metal::MTLCommandBufferStatus::Error {
        return Err(MetalError::ExecutionError(
            "GPU scale (in-place) command buffer error".to_string(),
        ));
    }

    Ok(())
}

/// Perform coset shift on data already in a GPU buffer: output[k] = input[k] * offset^k,
/// zero-padded to `output_len`.
///
/// Like [`gpu_coset_shift_to_buffer`] but reads from a Metal Buffer instead of a CPU slice.
/// This avoids a CPU-to-GPU transfer when the input was produced by a prior GPU operation.
///
/// # Arguments
///
/// - `input_buffer`: Metal buffer containing `input_len` Goldilocks u64 values
/// - `input_len`: Number of field elements in the input buffer
/// - `offset`: The coset offset element
/// - `output_len`: Total output length (includes zero-padding if larger than `input_len`)
/// - `state`: Pre-compiled coset shift Metal state
///
/// # Returns
///
/// A Metal Buffer containing `output_len` u64 values (coset-shifted + zero-padded).
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn gpu_coset_shift_buffer_to_buffer(
    input_buffer: &metal::Buffer,
    input_len: usize,
    offset: &FieldElement<Goldilocks64Field>,
    output_len: usize,
    state: &CosetShiftState,
) -> Result<metal::Buffer, MetalError> {
    use metal::MTLSize;

    let offset_u64 = Goldilocks64Field::canonical(offset.value());

    // Allocate GPU buffers
    let buf_output = state
        .state
        .alloc_buffer(output_len * std::mem::size_of::<u64>())?;
    let buf_offset = state
        .state
        .alloc_buffer_with_data(std::slice::from_ref(&offset_u64))?;
    let input_len_u32 = input_len as u32;
    let output_len_u32 = output_len as u32;
    let buf_input_len = state
        .state
        .alloc_buffer_with_data(std::slice::from_ref(&input_len_u32))?;
    let buf_output_len = state
        .state
        .alloc_buffer_with_data(std::slice::from_ref(&output_len_u32))?;

    // Dispatch the coset shift kernel
    let pipeline = state
        .state
        .get_pipeline_ref("goldilocks_coset_shift")
        .ok_or_else(|| MetalError::FunctionError("goldilocks_coset_shift".to_string()))?;

    let threads_per_group = state.coset_shift_max_threads.min(256);
    let thread_groups = (output_len as u64).div_ceil(threads_per_group);

    let command_buffer = state.state.command_queue().new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(input_buffer), 0);
    encoder.set_buffer(1, Some(&buf_output), 0);
    encoder.set_buffer(2, Some(&buf_offset), 0);
    encoder.set_buffer(3, Some(&buf_input_len), 0);
    encoder.set_buffer(4, Some(&buf_output_len), 0);

    encoder.dispatch_thread_groups(
        MTLSize::new(thread_groups, 1, 1),
        MTLSize::new(threads_per_group, 1, 1),
    );
    encoder.end_encoding();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    if command_buffer.status() == metal::MTLCommandBufferStatus::Error {
        return Err(MetalError::ExecutionError(
            "GPU coset shift (buffer-to-buffer) command buffer error".to_string(),
        ));
    }

    Ok(buf_output)
}

/// Element-wise cyclic multiply: output[i] = input[i] * pattern[i % pattern_len].
///
/// Used for combining base zerofier (small cyclic pattern) with end-exemptions
/// evaluations (large buffer), keeping the entire zerofier computation on GPU.
///
/// # Arguments
///
/// - `input`: Metal buffer containing `len` Goldilocks u64 values
/// - `len`: Number of elements in the input buffer
/// - `pattern`: Small cyclic pattern (e.g., base zerofier of length `blowup_factor * period`)
/// - `state`: Pre-compiled coset shift Metal state (contains the cyclic_mul pipeline)
///
/// # Returns
///
/// A new Metal Buffer containing `len` u64 values (each multiplied by the cyclic pattern).
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn gpu_cyclic_mul_buffer(
    input: &metal::Buffer,
    len: usize,
    pattern: &[u64],
    state: &CosetShiftState,
) -> Result<metal::Buffer, MetalError> {
    use metal::MTLSize;

    let pattern_len = pattern.len();

    // Allocate GPU buffers
    let buf_output = state.state.alloc_buffer(len * std::mem::size_of::<u64>())?;
    let buf_pattern = state.state.alloc_buffer_with_data(pattern)?;
    let len_u32 = len as u32;
    let pattern_len_u32 = pattern_len as u32;
    let buf_len = state
        .state
        .alloc_buffer_with_data(std::slice::from_ref(&len_u32))?;
    let buf_pattern_len = state
        .state
        .alloc_buffer_with_data(std::slice::from_ref(&pattern_len_u32))?;

    // Dispatch the cyclic mul kernel
    let pipeline = state
        .state
        .get_pipeline_ref("goldilocks_cyclic_mul")
        .ok_or_else(|| MetalError::FunctionError("goldilocks_cyclic_mul".to_string()))?;

    let threads_per_group = state.cyclic_mul_max_threads.min(256);
    let thread_groups = (len as u64).div_ceil(threads_per_group);

    let command_buffer = state.state.command_queue().new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(input), 0);
    encoder.set_buffer(1, Some(&buf_output), 0);
    encoder.set_buffer(2, Some(&buf_pattern), 0);
    encoder.set_buffer(3, Some(&buf_len), 0);
    encoder.set_buffer(4, Some(&buf_pattern_len), 0);

    encoder.dispatch_thread_groups(
        MTLSize::new(thread_groups, 1, 1),
        MTLSize::new(threads_per_group, 1, 1),
    );
    encoder.end_encoding();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    if command_buffer.status() == metal::MTLCommandBufferStatus::Error {
        return Err(MetalError::ExecutionError(
            "GPU cyclic mul command buffer error".to_string(),
        ));
    }

    Ok(buf_output)
}

/// GPU inverse FFT that returns a Metal Buffer.
///
/// Given evaluations at roots of unity, this computes polynomial coefficients via inverse
/// FFT entirely on the GPU and returns the result as a Metal Buffer (no CPU readback).
///
/// # Algorithm
///
/// 1. Generate inverse twiddle factors with `RootsConfig::BitReverseInversed`
/// 2. Upload evaluations as u64 to a GPU buffer
/// 3. Run `fft_buffer_to_buffer` with inverse twiddles
/// 4. Normalize by multiplying each element by `1/n` via `gpu_scale_buffer`
///
/// # Arguments
///
/// - `evaluations`: Field element evaluations at roots of unity (power-of-two length)
/// - `coset_state`: Pre-compiled coset shift Metal state (for the scale kernel)
/// - `metal_state`: Metal state containing the FFT shader library
///
/// # Returns
///
/// A Metal Buffer containing the polynomial coefficients as Goldilocks u64 values.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn gpu_ifft_to_buffer(
    evaluations: &[FieldElement<Goldilocks64Field>],
    coset_state: &CosetShiftState,
    metal_state: &MetalState,
) -> Result<metal::Buffer, MetalError> {
    type FpE = FieldElement<Goldilocks64Field>;

    let len = evaluations.len();
    let order = len.trailing_zeros() as u64;

    // Step 1: Generate inverse twiddle factors on GPU
    let inv_twiddles = gen_twiddles_to_buffer::<Goldilocks64Field>(
        order,
        RootsConfig::BitReverseInversed,
        metal_state,
    )?;

    // Step 2: Upload evaluations as u64 to a GPU buffer
    let evals_u64: Vec<u64> = evaluations
        .iter()
        .map(|fe| Goldilocks64Field::canonical(fe.value()))
        .collect();
    let eval_buffer = metal_state.alloc_buffer_data(&evals_u64);

    // Step 3: Run FFT with inverse twiddles (buffer-to-buffer)
    let result_buffer =
        fft_buffer_to_buffer::<Goldilocks64Field>(&eval_buffer, len, &inv_twiddles, metal_state)?;

    // Step 4: Normalize by 1/n
    let n_inv = FpE::from(len as u64)
        .inv()
        .expect("Power-of-two length is always invertible in an FFT field");
    let normalized = gpu_scale_buffer(&result_buffer, len, &n_inv, coset_state)?;

    Ok(normalized)
}

/// Coset IFFT on GPU: recovers polynomial coefficients from evaluations on a coset domain.
///
/// Given evaluations on `{offset * w^i}`, this computes the polynomial coefficients
/// by first performing an inverse FFT and then applying the inverse coset shift:
/// `coeff[k] = ifft_result[k] * offset_inv^k`.
///
/// The entire computation stays on GPU and the result is returned as a Metal Buffer.
///
/// # Algorithm
///
/// 1. Inverse FFT via [`gpu_ifft_to_buffer`]
/// 2. Compute `offset_inv = 1 / offset`
/// 3. Apply coset shift with `offset_inv` via [`gpu_coset_shift_buffer_to_buffer`]
///
/// # Arguments
///
/// - `evaluations`: Evaluations on the coset domain `{offset * w^i}`
/// - `coset_offset`: The coset offset element
/// - `coset_state`: Pre-compiled coset shift Metal state
/// - `metal_state`: Metal state containing the FFT shader library
///
/// # Returns
///
/// A Metal Buffer containing the polynomial coefficients as Goldilocks u64 values.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn gpu_interpolate_offset_fft_to_buffer(
    evaluations: &[FieldElement<Goldilocks64Field>],
    coset_offset: &FieldElement<Goldilocks64Field>,
    coset_state: &CosetShiftState,
    metal_state: &MetalState,
) -> Result<metal::Buffer, MetalError> {
    let len = evaluations.len();

    // Step 1: Inverse FFT to get coefficients buffer
    let coeffs_buffer = gpu_ifft_to_buffer(evaluations, coset_state, metal_state)?;

    // Step 2: Compute inverse of the coset offset
    let offset_inv = coset_offset.inv().expect("Coset offset must be invertible");

    // Step 3: Apply coset shift with offset_inv: coeff[k] *= offset_inv^k
    let result =
        gpu_coset_shift_buffer_to_buffer(&coeffs_buffer, len, &offset_inv, len, coset_state)?;

    Ok(result)
}

/// Batch-interpolates multiple evaluation columns into coefficient GPU buffers.
///
/// Given columns of evaluations at roots of unity (all same length), this computes
/// the polynomial coefficients for each column entirely on GPU:
/// 1. Generate inverse twiddles ONCE (shared across all columns)
/// 2. For each column: upload → IFFT → ÷N normalization
/// 3. Return `Vec<metal::Buffer>` (coefficient buffers on GPU)
///
/// This replaces the per-column `gpu_interpolate_fft` + CPU readback path,
/// eliminating per-column twiddle generation and CPU ÷N normalization.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn gpu_interpolate_columns_to_buffers(
    columns: &[Vec<FieldElement<Goldilocks64Field>>],
    coset_state: &CosetShiftState,
    metal_state: &MetalState,
) -> Result<Vec<metal::Buffer>, MetalError> {
    type FpE = FieldElement<Goldilocks64Field>;

    if columns.is_empty() {
        return Ok(Vec::new());
    }

    let len = columns[0].len();
    let order = len.trailing_zeros() as u64;

    // Generate inverse twiddles ONCE for all columns
    let inv_twiddles = gen_twiddles_to_buffer::<Goldilocks64Field>(
        order,
        RootsConfig::BitReverseInversed,
        metal_state,
    )?;

    // Pre-compute 1/n for normalization
    let n_inv = FpE::from(len as u64)
        .inv()
        .expect("Power-of-two length is always invertible in an FFT field");

    // Pre-allocate a single working buffer for all IFFT calls (reused across iterations)
    let working_buffer = metal_state.alloc_buffer::<u64>(len);

    let mut coeff_buffers = Vec::with_capacity(columns.len());

    for col in columns {
        // Upload evaluations as u64 to GPU
        let evals_u64: Vec<u64> = col
            .iter()
            .map(|fe| Goldilocks64Field::canonical(fe.value()))
            .collect();
        let eval_buffer = metal_state.alloc_buffer_data(&evals_u64);

        // IFFT with shared inverse twiddles and reused working buffer
        let result_buffer = fft_buffer_to_buffer_reuse::<Goldilocks64Field>(
            &eval_buffer,
            len,
            &inv_twiddles,
            &working_buffer,
            metal_state,
        )?;
        drop(eval_buffer); // Free input GPU buffer before scaling

        // Normalize by 1/n in-place (avoids allocating a separate output buffer)
        gpu_scale_buffer_inplace(&result_buffer, len, &n_inv, coset_state)?;

        coeff_buffers.push(result_buffer);
    }

    Ok(coeff_buffers)
}

/// Evaluates multiple coefficient GPU buffers on a coset domain, returning GPU Buffers.
///
/// Like [`gpu_evaluate_offset_fft_buffer_to_buffers_batch`] but all coefficients have
/// the same length. Generates twiddles once and shares them.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn gpu_evaluate_coeff_buffers_on_lde(
    coeff_buffers: &[metal::Buffer],
    coeff_len: usize,
    blowup_factor: usize,
    offset: &FieldElement<Goldilocks64Field>,
    coset_state: &CosetShiftState,
    metal_state: &MetalState,
) -> Result<(Vec<metal::Buffer>, usize), MetalError> {
    if coeff_buffers.is_empty() {
        return Ok((Vec::new(), 0));
    }

    let domain_size = coeff_len * blowup_factor;
    let order = domain_size.trailing_zeros() as u64;

    // Generate twiddles ONCE for the shared domain
    let twiddles_buffer =
        gen_twiddles_to_buffer::<Goldilocks64Field>(order, RootsConfig::BitReverse, metal_state)?;

    // Pre-allocate a single working buffer for all FFT calls (reused across iterations)
    let working_buffer = metal_state.alloc_buffer::<u64>(domain_size);

    let mut lde_buffers = Vec::with_capacity(coeff_buffers.len());

    for coeff_buf in coeff_buffers {
        // GPU coset shift + zero-pad to domain_size
        let shifted_buffer = gpu_coset_shift_buffer_to_buffer(
            coeff_buf,
            coeff_len,
            offset,
            domain_size,
            coset_state,
        )?;

        // FFT with shared twiddles and reused working buffer
        let result_buffer = fft_buffer_to_buffer_reuse::<Goldilocks64Field>(
            &shifted_buffer,
            domain_size,
            &twiddles_buffer,
            &working_buffer,
            metal_state,
        )?;
        drop(shifted_buffer); // Free coset-shifted buffer before next iteration

        lde_buffers.push(result_buffer);
    }

    Ok((lde_buffers, domain_size))
}

/// Fused IFFT → scale → coset shift → FFT in a single Metal command buffer.
///
/// Performs the full LDE pipeline for one column without intermediate commit/wait calls:
/// 1. IFFT (eval → working → coeff_buffer)
/// 2. Scale by 1/N (coeff_buffer in-place)
/// 3. Coset shift + zero-pad (coeff_buffer → shifted_buffer)
/// 4. FFT (shifted → working → lde_buffer)
///
/// This eliminates 3 command buffer submissions per column (4→1).
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

    // Allocate intermediate and output buffers
    let coeff_buffer = metal_state.alloc_buffer::<u64>(trace_len);
    let shifted_buffer = coset_state
        .state
        .alloc_buffer(domain_size * std::mem::size_of::<u64>())?;
    let lde_buffer = metal_state.alloc_buffer::<u64>(domain_size);

    // Pre-compute scalar and parameter buffers for scale and coset shift
    let scalar_u64 = Goldilocks64Field::canonical(n_inv.value());
    let buf_scalar = coset_state
        .state
        .alloc_buffer_with_data(std::slice::from_ref(&scalar_u64))?;
    let scale_len_u32 = trace_len as u32;
    let buf_scale_len = coset_state
        .state
        .alloc_buffer_with_data(std::slice::from_ref(&scale_len_u32))?;

    let offset_u64 = Goldilocks64Field::canonical(coset_offset.value());
    let buf_offset = coset_state
        .state
        .alloc_buffer_with_data(std::slice::from_ref(&offset_u64))?;
    let input_len_u32 = trace_len as u32;
    let output_len_u32 = domain_size as u32;
    let buf_input_len = coset_state
        .state
        .alloc_buffer_with_data(std::slice::from_ref(&input_len_u32))?;
    let buf_output_len = coset_state
        .state
        .alloc_buffer_with_data(std::slice::from_ref(&output_len_u32))?;

    objc::rc::autoreleasepool(|| {
        let command_buffer = metal_state.queue.new_command_buffer();

        // Stage 1: IFFT (eval_buffer → working_buffer → coeff_buffer)
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

        // Stage 3: Coset shift + zero-pad (coeff_buffer → shifted_buffer)
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

        // Stage 4: FFT (shifted_buffer → working_buffer → lde_buffer)
        fft_encode_to_command_buffer::<Goldilocks64Field>(
            command_buffer,
            &shifted_buffer,
            domain_size,
            fwd_twiddles,
            working_buffer,
            &lde_buffer,
            metal_state,
        )?;

        // Single commit + wait for all 4 stages
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

/// Computes LDE directly from trace evaluations, fusing IFFT + coset LDE per column.
///
/// Given columns of evaluations at roots of unity (all same length N), this computes
/// the low-degree extension on the coset `{offset * ω_{4N}^i}` entirely on GPU:
///
/// For each column:
/// 1. Upload evaluations → GPU
/// 2. IFFT_N (inverse twiddles, shared) → coefficients
/// 3. ÷N normalization (in-place)
/// 4. Coset shift + zero-pad: `d[k] = c[k] * offset^k` for k < N, 0 for k ≥ N
/// 5. FFT_{4N} (forward twiddles, shared) → LDE evaluations
/// 6. Drop intermediate coefficient and shifted buffers
///
/// # Advantages over separate `gpu_interpolate_columns_to_buffers` + `gpu_evaluate_coeff_buffers_on_lde`
///
/// - **Lower peak memory**: Only one N-word coefficient buffer exists at a time
///   (vs all columns' coefficient buffers simultaneously).
///   At 2^20 with 3 columns, this saves 2 × N × 8B = 16 MiB.
/// - **Single working buffer**: One 4N-word buffer reused for both IFFT and FFT.
/// - **No intermediate `Vec<Buffer>`** for coefficient storage.
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

    // Generate twiddles ONCE for all columns
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

    // Pre-compute 1/N for normalization
    let n_inv = FpE::from(trace_len as u64)
        .inv()
        .expect("Power-of-two length is always invertible in an FFT field");

    // Single working buffer sized for the larger FFT (4N), reused for both IFFT and FFT
    let working_buffer = metal_state.alloc_buffer::<u64>(domain_size);

    let mut lde_buffers = Vec::with_capacity(columns.len());

    for col in columns {
        // Upload evaluations as u64 to GPU
        let evals_u64: Vec<u64> = col
            .iter()
            .map(|fe| Goldilocks64Field::canonical(fe.value()))
            .collect();
        let eval_buffer = metal_state.alloc_buffer_data(&evals_u64);

        // Fused IFFT → scale → coset shift → FFT in single command buffer
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

/// Coset IFFT on a GPU buffer: recovers polynomial coefficients from evaluations already on GPU.
///
/// Like [`gpu_interpolate_offset_fft_to_buffer`] but reads from an existing Metal Buffer
/// instead of a CPU slice, avoiding a CPU-to-GPU transfer when the evaluations were
/// produced by a prior GPU operation (e.g., the DEEP composition kernel).
///
/// # Algorithm
///
/// 1. Generate inverse twiddles on GPU
/// 2. `fft_buffer_to_buffer` with inverse twiddles
/// 3. Normalize by `1/n` via `gpu_scale_buffer`
/// 4. Apply inverse coset shift via `gpu_coset_shift_buffer_to_buffer`
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

    // Step 1: Generate inverse twiddle factors on GPU
    let inv_twiddles = gen_twiddles_to_buffer::<Goldilocks64Field>(
        order,
        RootsConfig::BitReverseInversed,
        metal_state,
    )?;

    // Step 2: FFT with inverse twiddles (buffer-to-buffer)
    let result_buffer =
        fft_buffer_to_buffer::<Goldilocks64Field>(eval_buffer, len, &inv_twiddles, metal_state)?;

    // Step 3: Normalize by 1/n
    let n_inv = FpE::from(len as u64)
        .inv()
        .expect("Power-of-two length is always invertible in an FFT field");
    let normalized = gpu_scale_buffer(&result_buffer, len, &n_inv, coset_state)?;

    // Step 4: Inverse coset shift: coeff[k] *= offset_inv^k
    let offset_inv = coset_offset.inv().expect("Coset offset must be invertible");
    let result = gpu_coset_shift_buffer_to_buffer(&normalized, len, &offset_inv, len, coset_state)?;

    Ok(result)
}

/// Embedded Metal shader source for the coefficient striding kernel.
#[cfg(all(target_os = "macos", feature = "metal"))]
const STRIDE_SHADER: &str = include_str!("shaders/stride.metal");

/// Parameters matching the Metal `StrideParams` struct.
#[cfg(all(target_os = "macos", feature = "metal"))]
#[repr(C)]
#[derive(Copy, Clone)]
struct StrideParams {
    num_coeffs: u32,
    num_parts: u32,
}

/// GPU coefficient striding: break_in_parts on GPU.
///
/// Given a GPU buffer containing N coefficients (output of IFFT), produces
/// `num_parts` GPU buffers where part `i` contains coefficients at indices
/// `[i, i+k, i+2k, ...]` (same as `Polynomial::break_in_parts`).
///
/// This avoids CPU readback → break_in_parts → re-upload for the LDE FFT path.
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

    // Create a single output buffer for all parts concatenated
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

    // Split concatenated GPU buffer into per-part buffers using Metal blit encoder.
    // This avoids a GPU→CPU→GPU round-trip: data stays on GPU throughout.
    let mut part_buffers = Vec::with_capacity(num_parts);
    for _ in 0..num_parts {
        let part_buf = state.alloc_buffer_data(&vec![0u64; part_len]);
        part_buffers.push(part_buf);
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

        // Evaluate on roots of unity (blowup=1, no offset => offset=1)
        let evals = Polynomial::evaluate_fft::<Goldilocks64Field>(&poly, 1, None).unwrap();

        // Interpolate back using GPU
        let recovered_coeffs =
            gpu_interpolate_fft::<Goldilocks64Field>(&evals, state.inner()).unwrap();

        assert_eq!(original_coeffs.len(), recovered_coeffs.len());
        for (orig, recov) in original_coeffs.iter().zip(&recovered_coeffs) {
            assert_eq!(orig, recov, "Roundtrip coefficient mismatch");
        }
    }

    // =========================================================================
    // GPU Coset Shift + Scale differential tests
    // =========================================================================

    #[test]
    fn gpu_coset_shift_matches_cpu() {
        let coset_state = CosetShiftState::new().unwrap();

        let coeffs: Vec<FpE> = (0..16).map(|i| FpE::from(i as u64 + 1)).collect();
        let offset = FpE::from(7u64);
        let blowup_factor = 4;
        let output_len = coeffs.len() * blowup_factor;

        // CPU coset shift: shifted[k] = coeffs[k] * offset^k, then zero-pad
        let mut cpu_shifted = Vec::with_capacity(output_len);
        let mut offset_power = FpE::one();
        for coeff in &coeffs {
            cpu_shifted.push(coeff * offset_power);
            offset_power *= offset;
        }
        cpu_shifted.resize(output_len, FpE::zero());

        // GPU coset shift
        let gpu_buffer =
            gpu_coset_shift_to_buffer(&coeffs, &offset, output_len, &coset_state).unwrap();

        // Read back GPU result
        let gpu_u64: Vec<u64> = unsafe { coset_state.state.read_buffer(&gpu_buffer, output_len) };
        let gpu_shifted: Vec<FpE> = gpu_u64.iter().map(|&v| FpE::from(v)).collect();

        assert_eq!(cpu_shifted.len(), gpu_shifted.len());
        for (i, (cpu, gpu)) in cpu_shifted.iter().zip(&gpu_shifted).enumerate() {
            assert_eq!(
                cpu, gpu,
                "Coset shift mismatch at index {i}: CPU={:?} GPU={:?}",
                cpu, gpu
            );
        }
    }

    #[test]
    fn gpu_coset_shift_matches_cpu_large() {
        let coset_state = CosetShiftState::new().unwrap();

        // Larger test: 256 coefficients with blowup factor 4
        let coeffs: Vec<FpE> = (0..256).map(|i| FpE::from(i as u64 * 31 + 17)).collect();
        let offset = FpE::from(42u64);
        let output_len = coeffs.len() * 4;

        // CPU coset shift
        let mut cpu_shifted = Vec::with_capacity(output_len);
        let mut offset_power = FpE::one();
        for coeff in &coeffs {
            cpu_shifted.push(coeff * offset_power);
            offset_power *= offset;
        }
        cpu_shifted.resize(output_len, FpE::zero());

        // GPU coset shift
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

        // CPU scale
        let cpu_scaled: Vec<FpE> = values.iter().map(|v| v * scalar).collect();

        // Upload values to GPU, then scale
        let values_u64: Vec<u64> = values
            .iter()
            .map(|fe| Goldilocks64Field::canonical(fe.value()))
            .collect();
        let input_buffer = coset_state
            .state
            .alloc_buffer_with_data(&values_u64)
            .unwrap();
        let gpu_buffer =
            gpu_scale_buffer(&input_buffer, values.len(), &scalar, &coset_state).unwrap();

        // Read back GPU result
        let gpu_u64: Vec<u64> = unsafe { coset_state.state.read_buffer(&gpu_buffer, values.len()) };
        let gpu_scaled: Vec<FpE> = gpu_u64.iter().map(|&v| FpE::from(v)).collect();

        assert_eq!(cpu_scaled.len(), gpu_scaled.len());
        for (i, (cpu, gpu)) in cpu_scaled.iter().zip(&gpu_scaled).enumerate() {
            assert_eq!(
                cpu, gpu,
                "Scale mismatch at index {i}: CPU={:?} GPU={:?}",
                cpu, gpu
            );
        }
    }

    #[test]
    fn gpu_scale_matches_cpu_large() {
        let coset_state = CosetShiftState::new().unwrap();

        // Larger test: 1024 elements
        let values: Vec<FpE> = (0..1024).map(|i| FpE::from(i as u64 * 7 + 3)).collect();
        let scalar = FpE::from(999u64);

        // CPU scale
        let cpu_scaled: Vec<FpE> = values.iter().map(|v| v * scalar).collect();

        // GPU scale
        let values_u64: Vec<u64> = values
            .iter()
            .map(|fe| Goldilocks64Field::canonical(fe.value()))
            .collect();
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

    // =========================================================================
    // GPU IFFT + Coset IFFT differential tests
    // =========================================================================

    #[test]
    fn gpu_ifft_matches_cpu() {
        let metal_state = crate::metal::state::StarkMetalState::new().unwrap();
        let coset_state = CosetShiftState::new().unwrap();

        let values: Vec<FpE> = (0..16).map(|i| FpE::from(i as u64 + 1)).collect();

        // CPU inverse FFT (returns Vec)
        let cpu_coeffs =
            gpu_interpolate_fft::<Goldilocks64Field>(&values, metal_state.inner()).unwrap();

        // GPU inverse FFT (returns buffer)
        let gpu_buffer = gpu_ifft_to_buffer(&values, &coset_state, metal_state.inner()).unwrap();

        // Read back GPU result
        let gpu_u64: Vec<u64> = unsafe { coset_state.state.read_buffer(&gpu_buffer, values.len()) };
        let gpu_coeffs: Vec<FpE> = gpu_u64.iter().map(|&v| FpE::from(v)).collect();

        assert_eq!(cpu_coeffs.len(), gpu_coeffs.len());
        for (i, (cpu, gpu)) in cpu_coeffs.iter().zip(&gpu_coeffs).enumerate() {
            assert_eq!(
                cpu, gpu,
                "IFFT coefficient mismatch at index {i}: CPU={:?} GPU={:?}",
                cpu, gpu
            );
        }
    }

    /// Differential test: GPU stride (break_in_parts) matches CPU Polynomial::break_in_parts.
    #[test]
    fn gpu_stride_matches_cpu_break_in_parts() {
        let state = crate::metal::state::StarkMetalState::new().expect("Metal init failed");

        // 16 coefficients, break into 4 parts of 4 each
        let coeffs: Vec<FpE> = (0..16).map(|i| FpE::from(i as u64 + 100)).collect();
        let poly = Polynomial::new(&coeffs);
        let cpu_parts = poly.break_in_parts(4);

        // Upload coefficients to GPU
        let coeffs_u64: Vec<u64> = coeffs
            .iter()
            .map(|fe| Goldilocks64Field::canonical(fe.value()))
            .collect();
        let gpu_buffer = state.inner().alloc_buffer_data(&coeffs_u64);

        // GPU stride
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

    /// Differential test: GPU stride + GPU buffer-to-buffer FFT matches CPU break_in_parts + CPU FFT.
    #[test]
    fn gpu_stride_fft_matches_cpu_break_in_parts_fft() {
        let state = crate::metal::state::StarkMetalState::new().expect("Metal init failed");
        let coset_state = CosetShiftState::new().expect("CosetShiftState init failed");

        // 32 coefficients, break into 2 parts, blowup_factor = 4
        let coeffs: Vec<FpE> = (0..32).map(|i| FpE::from(i as u64 + 1)).collect();
        let poly = Polynomial::new(&coeffs);
        let offset = FpE::from(7u64);
        let blowup_factor = 4;
        let num_parts = 2;

        // CPU path: break_in_parts + evaluate_offset_fft
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

        // GPU path: stride + buffer-to-buffer batch FFT
        let coeffs_u64: Vec<u64> = coeffs
            .iter()
            .map(|fe| Goldilocks64Field::canonical(fe.value()))
            .collect();
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

        // Create a polynomial with known coefficients
        let original_coeffs: Vec<FpE> = (0..16).map(|i| FpE::from(i as u64 + 1)).collect();
        let offset = FpE::from(7u64);
        let blowup_factor = 1;

        // Evaluate the polynomial on the coset domain using GPU FFT
        let evals = gpu_evaluate_offset_fft::<Goldilocks64Field>(
            &original_coeffs,
            blowup_factor,
            &offset,
            metal_state.inner(),
        )
        .unwrap();

        // Recover coefficients via GPU coset IFFT
        let gpu_buffer = gpu_interpolate_offset_fft_to_buffer(
            &evals,
            &offset,
            &coset_state,
            metal_state.inner(),
        )
        .unwrap();

        // Read back GPU result
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
