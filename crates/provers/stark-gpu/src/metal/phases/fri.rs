//! GPU Phase 4: DEEP Composition Polynomial + FRI + Queries.
//!
//! This module mirrors `round_4_compute_and_run_fri_on_the_deep_composition_polynomial`
//! from the CPU STARK prover. It computes the DEEP composition polynomial, runs FRI
//! commit and query phases, performs grinding, and extracts Merkle opening proofs.
//!
//! GPU acceleration is used for:
//! - DEEP composition polynomial (Metal shader)
//! - FRI layer FFT evaluation (Metal FFT)
//! - FRI layer Merkle commit (Metal Keccak256)

use lambdaworks_math::fft::cpu::bit_reversing::reverse_index;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::{IsFFTField, IsField, IsSubFieldOf};
use lambdaworks_math::polynomial::Polynomial;
use lambdaworks_math::traits::AsBytes;

use lambdaworks_crypto::fiat_shamir::is_transcript::IsStarkTranscript;

use stark_platinum_prover::config::{BatchedMerkleTree, BatchedMerkleTreeBackend, Commitment};
use stark_platinum_prover::domain::Domain;
use stark_platinum_prover::fri;
use stark_platinum_prover::fri::fri_commitment::FriLayer;
use stark_platinum_prover::fri::fri_decommit::FriDecommitment;
use stark_platinum_prover::grinding;
use stark_platinum_prover::proof::stark::{DeepPolynomialOpening, PolynomialOpenings};
use stark_platinum_prover::prover::ProvingError;
use stark_platinum_prover::table::Table;
use stark_platinum_prover::traits::AIR;

use crate::metal::phases::composition::GpuRound2Result;
use crate::metal::phases::ood::GpuRound3Result;
use crate::metal::phases::rap::GpuRound1Result;

#[cfg(all(target_os = "macos", feature = "metal"))]
use crate::metal::deep_composition::{DeepCompositionState, DomainInversionState};
#[cfg(all(target_os = "macos", feature = "metal"))]
use crate::metal::fft::gpu_evaluate_offset_fft;
#[cfg(all(target_os = "macos", feature = "metal"))]
use crate::metal::fft::{gpu_coset_shift_buffer_to_buffer, CosetShiftState};
#[cfg(all(target_os = "macos", feature = "metal"))]
use crate::metal::merkle::{
    gpu_fri_layer_commit, gpu_fri_layer_commit_from_buffer, gpu_generate_nonce, GpuMerkleState,
};
#[cfg(all(target_os = "macos", feature = "metal"))]
use crate::metal::state::StarkMetalState;
#[cfg(all(target_os = "macos", feature = "metal"))]
use lambdaworks_math::fft::cpu::bit_reversing::in_place_bit_reverse_permute;
#[cfg(all(target_os = "macos", feature = "metal"))]
use lambdaworks_math::fft::gpu::metal::ops::{fft_buffer_to_buffer, gen_twiddles_to_buffer};
#[cfg(all(target_os = "macos", feature = "metal"))]
use lambdaworks_math::field::fields::u64_goldilocks_field::{
    Degree3GoldilocksExtensionField, Goldilocks64Field,
};
#[cfg(all(target_os = "macos", feature = "metal"))]
use lambdaworks_math::field::traits::{IsPrimeField, RootsConfig};

/// Fp3 extension field type alias.
#[cfg(all(target_os = "macos", feature = "metal"))]
type Fp3 = Degree3GoldilocksExtensionField;
/// Fp3 field element type alias.
#[cfg(all(target_os = "macos", feature = "metal"))]
type Fp3E = FieldElement<Fp3>;

#[cfg(all(target_os = "macos", feature = "metal"))]
use lambdaworks_gpu::metal::abstractions::{
    errors::MetalError,
    state::{DynamicMetalState, MetalState},
};

// =============================================================================
// GPU FRI Fold kernel (Goldilocks-specific)
// =============================================================================

/// Source code for the Goldilocks field header (fp_u64.h.metal).
/// Prepended to the FRI fold shader at runtime to resolve the Fp64Goldilocks class.
#[cfg(all(target_os = "macos", feature = "metal"))]
const FRI_GOLDILOCKS_FIELD_HEADER: &str =
    include_str!("../../../../../math/src/gpu/metal/shaders/field/fp_u64.h.metal");

/// Source code for the FRI fold Metal kernel.
#[cfg(all(target_os = "macos", feature = "metal"))]
const FRI_FOLD_SHADER: &str = include_str!("../shaders/fri_fold.metal");

/// Pre-compiled Metal state for the FRI fold kernel.
///
/// Caches the compiled pipeline for `goldilocks_fri_fold`.
/// Create once and reuse across all FRI folding rounds.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub struct FriFoldState {
    state: DynamicMetalState,
    fri_fold_max_threads: u64,
}

#[cfg(all(target_os = "macos", feature = "metal"))]
impl FriFoldState {
    /// Compile the FRI fold shader and prepare the pipeline.
    pub fn new() -> Result<Self, MetalError> {
        // Concatenate the Goldilocks field header with the FRI fold shader
        // since runtime compilation via new_library_with_source does not support
        // file-system #include directives.
        let combined_source = format!("{}\n{}", FRI_GOLDILOCKS_FIELD_HEADER, FRI_FOLD_SHADER);

        let mut state = DynamicMetalState::new()?;
        state.load_library(&combined_source)?;
        let fri_fold_max_threads = state.prepare_pipeline("goldilocks_fri_fold")?;
        Ok(Self {
            state,
            fri_fold_max_threads,
        })
    }

    /// Compile the FRI fold shader sharing a device and queue with an existing Metal state.
    ///
    /// This avoids creating new Metal device/queue pairs, reducing GPU resource usage
    /// when many shader states coexist.
    pub fn from_device_and_queue(
        device: &metal::Device,
        queue: &metal::CommandQueue,
    ) -> Result<Self, MetalError> {
        let combined_source = format!("{}\n{}", FRI_GOLDILOCKS_FIELD_HEADER, FRI_FOLD_SHADER);

        let mut state = DynamicMetalState::from_device_and_queue(device, queue);
        state.load_library(&combined_source)?;
        let fri_fold_max_threads = state.prepare_pipeline("goldilocks_fri_fold")?;
        Ok(Self {
            state,
            fri_fold_max_threads,
        })
    }
}

/// Perform FRI fold on GPU from an existing Metal buffer.
///
/// Dispatches the `goldilocks_fri_fold` kernel: `result[k] = 2 * (coeffs[2k] + beta * coeffs[2k+1])`.
///
/// # Arguments
///
/// - `coeffs_buffer`: Input Metal buffer containing `num_coeffs` Goldilocks u64 values
/// - `num_coeffs`: Number of coefficients (must be even)
/// - `beta`: The FRI folding challenge
/// - `state`: Pre-compiled FRI fold Metal state
///
/// # Returns
///
/// A tuple of (Metal Buffer, half_len) where the buffer contains the folded polynomial
/// coefficients and half_len = num_coeffs / 2.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn gpu_fold_polynomial(
    coeffs_buffer: &metal::Buffer,
    num_coeffs: usize,
    beta: &FieldElement<Goldilocks64Field>,
    state: &FriFoldState,
) -> Result<(metal::Buffer, usize), MetalError> {
    use metal::MTLSize;

    // Handle edge case: 0 or 1 coefficients.
    // The GPU kernel requires at least 2 coefficients (one even/odd pair).
    if num_coeffs <= 1 {
        let val: u64 = if num_coeffs == 1 {
            // Fold of a single coefficient: result = 2 * coeffs[0] (even part, no odd part).
            let vals: Vec<u64> = MetalState::retrieve_contents(coeffs_buffer);
            let c0 = vals.first().copied().unwrap_or(0);
            let fe = FieldElement::<Goldilocks64Field>::from(c0);
            let result = FieldElement::<Goldilocks64Field>::from(2u64) * fe;
            Goldilocks64Field::canonical(result.value())
        } else {
            0u64
        };
        let buf = state
            .state
            .alloc_buffer_with_data(std::slice::from_ref(&val))?;
        return Ok((buf, 1));
    }

    // If odd number of coefficients, pad to even by copying into a new buffer with a trailing zero.
    // The GPU kernel processes pairs: result[k] = 2 * (coeffs[2k] + beta * coeffs[2k+1]).
    let (input_buf_owned, input_ref, padded_len) = if !num_coeffs.is_multiple_of(2) {
        let padded_len = num_coeffs + 1;
        let buf_padded = state
            .state
            .alloc_buffer(padded_len * std::mem::size_of::<u64>())?;
        // Blit copy original data; trailing element is zero from alloc_buffer.
        let cmd = state.state.command_queue().new_command_buffer();
        let blit = cmd.new_blit_command_encoder();
        blit.copy_from_buffer(
            coeffs_buffer,
            0,
            &buf_padded,
            0,
            (num_coeffs * std::mem::size_of::<u64>()) as u64,
        );
        blit.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        // We keep ownership and pass reference below
        (Some(buf_padded), None, padded_len)
    } else {
        (None, Some(coeffs_buffer), num_coeffs)
    };

    let actual_input: &metal::Buffer = match (&input_buf_owned, input_ref) {
        (Some(buf), _) => buf,
        (_, Some(buf)) => buf,
        _ => unreachable!(),
    };

    let half_len = padded_len / 2;

    let beta_u64 = Goldilocks64Field::canonical(beta.value());

    // Allocate output buffer and parameter buffers
    let buf_output = state
        .state
        .alloc_buffer(half_len * std::mem::size_of::<u64>())?;
    let buf_beta = state
        .state
        .alloc_buffer_with_data(std::slice::from_ref(&beta_u64))?;
    let half_len_u32 = half_len as u32;
    let buf_half_len = state
        .state
        .alloc_buffer_with_data(std::slice::from_ref(&half_len_u32))?;

    // Dispatch the FRI fold kernel
    let pipeline = state
        .state
        .get_pipeline_ref("goldilocks_fri_fold")
        .ok_or_else(|| MetalError::FunctionError("goldilocks_fri_fold".to_string()))?;

    let threads_per_group = state.fri_fold_max_threads.min(256);
    let thread_groups = (half_len as u64).div_ceil(threads_per_group);

    let command_buffer = state.state.command_queue().new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(actual_input), 0);
    encoder.set_buffer(1, Some(&buf_output), 0);
    encoder.set_buffer(2, Some(&buf_beta), 0);
    encoder.set_buffer(3, Some(&buf_half_len), 0);

    encoder.dispatch_thread_groups(
        MTLSize::new(thread_groups, 1, 1),
        MTLSize::new(threads_per_group, 1, 1),
    );
    encoder.end_encoding();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    if command_buffer.status() == metal::MTLCommandBufferStatus::Error {
        return Err(MetalError::ExecutionError(
            "GPU FRI fold command buffer error".to_string(),
        ));
    }

    Ok((buf_output, half_len))
}

/// Perform FRI fold on GPU from CPU polynomial data.
///
/// Converts polynomial coefficients to u64, uploads them to the GPU, and dispatches
/// the `goldilocks_fri_fold` kernel.
///
/// # Arguments
///
/// - `poly`: The polynomial to fold
/// - `beta`: The FRI folding challenge
/// - `state`: Pre-compiled FRI fold Metal state
///
/// # Returns
///
/// A tuple of (Metal Buffer, half_len) where the buffer contains the folded polynomial
/// coefficients and half_len = num_coeffs / 2.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn gpu_fold_polynomial_from_cpu(
    poly: &Polynomial<FieldElement<Goldilocks64Field>>,
    beta: &FieldElement<Goldilocks64Field>,
    state: &FriFoldState,
) -> Result<(metal::Buffer, usize), MetalError> {
    let coeffs = poly.coefficients();
    let num_coeffs = coeffs.len();

    // Convert coefficients to canonical u64 representation for GPU upload
    let coeffs_u64: Vec<u64> = coeffs
        .iter()
        .map(|fe| Goldilocks64Field::canonical(fe.value()))
        .collect();

    let buf_input = state.state.alloc_buffer_with_data(&coeffs_u64)?;

    gpu_fold_polynomial(&buf_input, num_coeffs, beta, state)
}

// =============================================================================
// GPU Evaluation-Domain FRI Fold
// =============================================================================

/// Source code for the eval-domain FRI fold Metal kernel.
#[cfg(all(target_os = "macos", feature = "metal"))]
const FRI_FOLD_EVAL_SHADER: &str = include_str!("../shaders/fri_fold_eval.metal");

/// Source code for the FRI domain inverse precomputation Metal kernel.
#[cfg(all(target_os = "macos", feature = "metal"))]
const FRI_DOMAIN_INV_SHADER: &str = include_str!("../shaders/fri_domain_inv.metal");

/// Source code for the FRI inverse squaring Metal kernel.
#[cfg(all(target_os = "macos", feature = "metal"))]
const FRI_SQUARE_INV_SHADER: &str = include_str!("../shaders/fri_square_inv.metal");

/// Parameters for the eval-domain FRI fold kernel.
#[cfg(all(target_os = "macos", feature = "metal"))]
#[repr(C)]
#[derive(Copy, Clone)]
struct FriFoldEvalParams {
    half_len: u32,
}

/// Parameters for the FRI domain inverse kernel.
#[cfg(all(target_os = "macos", feature = "metal"))]
#[repr(C)]
#[derive(Copy, Clone)]
struct FriDomainInvParams {
    half_len: u32,
    log_half_len: u32,
}

/// Parameters for the FRI inverse squaring kernel.
#[cfg(all(target_os = "macos", feature = "metal"))]
#[repr(C)]
#[derive(Copy, Clone)]
struct FriSquareInvParams {
    len: u32,
}

/// Pre-compiled Metal state for the eval-domain FRI fold kernel.
///
/// Caches the compiled pipeline for `goldilocks_fri_fold_eval`.
/// Create once and reuse across all FRI folding rounds.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub struct FriFoldEvalState {
    state: DynamicMetalState,
    max_threads: u64,
}

#[cfg(all(target_os = "macos", feature = "metal"))]
impl FriFoldEvalState {
    pub fn new() -> Result<Self, MetalError> {
        let combined_source = format!("{}\n{}", FRI_GOLDILOCKS_FIELD_HEADER, FRI_FOLD_EVAL_SHADER);
        let mut state = DynamicMetalState::new()?;
        state.load_library(&combined_source)?;
        let max_threads = state.prepare_pipeline("goldilocks_fri_fold_eval")?;
        Ok(Self { state, max_threads })
    }

    pub fn from_device_and_queue(
        device: &metal::Device,
        queue: &metal::CommandQueue,
    ) -> Result<Self, MetalError> {
        let combined_source = format!("{}\n{}", FRI_GOLDILOCKS_FIELD_HEADER, FRI_FOLD_EVAL_SHADER);
        let mut state = DynamicMetalState::from_device_and_queue(device, queue);
        state.load_library(&combined_source)?;
        let max_threads = state.prepare_pipeline("goldilocks_fri_fold_eval")?;
        Ok(Self { state, max_threads })
    }
}

/// Eval-domain FRI fold on GPU (bit-reversed order).
///
/// Given evaluations in bit-reversed order on a coset domain, the paired
/// elements `(x, -x)` are adjacent at positions `(2i, 2i+1)`. Computes:
///
///   result[i] = (evals[2i] + evals[2i+1]) + beta * (evals[2i] - evals[2i+1]) * inv_x[i]
///
/// This matches the verifier's convention (includes the factor of 2).
/// The `inv_x_buffer` must contain precomputed `1/x_i` for i=0..N/2-1,
/// where `x_i` is the domain point at bit-reversed position `2i`.
///
/// Output is in bit-reversed order for the next FRI layer.
/// Returns `(output_buffer, half_len)`.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn gpu_fold_evaluations(
    evals_buffer: &metal::Buffer,
    num_evals: usize,
    beta: &FieldElement<Goldilocks64Field>,
    inv_x_buffer: &metal::Buffer,
    state: &FriFoldEvalState,
) -> Result<(metal::Buffer, usize), MetalError> {
    use metal::MTLSize;

    assert!(
        num_evals >= 2 && num_evals.is_multiple_of(2),
        "eval-domain fold requires even number of evaluations"
    );

    let half_len = num_evals / 2;
    let beta_u64 = Goldilocks64Field::canonical(beta.value());

    let buf_output = state
        .state
        .alloc_buffer(half_len * std::mem::size_of::<u64>())?;
    let buf_beta = state
        .state
        .alloc_buffer_with_data(std::slice::from_ref(&beta_u64))?;
    let params = FriFoldEvalParams {
        half_len: half_len as u32,
    };
    let buf_params = state
        .state
        .alloc_buffer_with_data(std::slice::from_ref(&params))?;

    let pipeline = state
        .state
        .get_pipeline_ref("goldilocks_fri_fold_eval")
        .ok_or_else(|| MetalError::FunctionError("goldilocks_fri_fold_eval".to_string()))?;

    let threads_per_group = state.max_threads.min(256);
    let thread_groups = (half_len as u64).div_ceil(threads_per_group);

    let command_buffer = state.state.command_queue().new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(evals_buffer), 0);
    encoder.set_buffer(1, Some(&buf_output), 0);
    encoder.set_buffer(2, Some(&buf_beta), 0);
    encoder.set_buffer(3, Some(inv_x_buffer), 0);
    encoder.set_buffer(4, Some(&buf_params), 0);
    encoder.dispatch_thread_groups(
        MTLSize::new(thread_groups, 1, 1),
        MTLSize::new(threads_per_group, 1, 1),
    );
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    if command_buffer.status() == metal::MTLCommandBufferStatus::Error {
        return Err(MetalError::ExecutionError(
            "GPU eval-domain FRI fold error".to_string(),
        ));
    }

    Ok((buf_output, half_len))
}

/// Fused domain inverse + eval-domain fold in a single GPU command buffer.
///
/// Combines [`gpu_compute_fri_domain_inverses`] and [`gpu_fold_evaluations`] into one
/// command buffer submission, eliminating one `wait_until_completed()` per FRI layer.
/// The domain inverse output feeds directly into the fold kernel without CPU sync.
///
/// Returns `(folded_buffer, half_len, inv_x_buffer)` — the `inv_x_buffer` can be
/// stride-2 squared for subsequent FRI layers via [`gpu_fold_with_squared_inv`].
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn gpu_fold_evaluations_with_domain_inv(
    evals_buffer: &metal::Buffer,
    num_evals: usize,
    beta: &FieldElement<Goldilocks64Field>,
    h_inv: &FieldElement<Goldilocks64Field>,
    omega_inv: &FieldElement<Goldilocks64Field>,
    fold_eval_state: &FriFoldEvalState,
    domain_inv_state: &FriDomainInvState,
) -> Result<(metal::Buffer, usize, metal::Buffer), MetalError> {
    use metal::MTLSize;

    assert!(
        num_evals >= 2 && num_evals.is_multiple_of(2),
        "eval-domain fold requires even number of evaluations"
    );

    let half_len = num_evals / 2;

    // --- Domain inverse parameters ---
    let h_inv_u64 = Goldilocks64Field::canonical(h_inv.value());
    let omega_inv_u64 = Goldilocks64Field::canonical(omega_inv.value());

    let buf_h_inv = domain_inv_state
        .state
        .alloc_buffer_with_data(std::slice::from_ref(&h_inv_u64))?;
    let buf_omega_inv = domain_inv_state
        .state
        .alloc_buffer_with_data(std::slice::from_ref(&omega_inv_u64))?;
    let buf_inv_x = domain_inv_state
        .state
        .alloc_buffer(half_len * std::mem::size_of::<u64>())?;
    let inv_params = FriDomainInvParams {
        half_len: half_len as u32,
        log_half_len: half_len.trailing_zeros(),
    };
    let buf_inv_params = domain_inv_state
        .state
        .alloc_buffer_with_data(std::slice::from_ref(&inv_params))?;

    // --- Fold parameters ---
    let beta_u64 = Goldilocks64Field::canonical(beta.value());
    let buf_output = fold_eval_state
        .state
        .alloc_buffer(half_len * std::mem::size_of::<u64>())?;
    let buf_beta = fold_eval_state
        .state
        .alloc_buffer_with_data(std::slice::from_ref(&beta_u64))?;
    let fold_params = FriFoldEvalParams {
        half_len: half_len as u32,
    };
    let buf_fold_params = fold_eval_state
        .state
        .alloc_buffer_with_data(std::slice::from_ref(&fold_params))?;

    // --- Get pipelines ---
    let inv_pipeline = domain_inv_state
        .state
        .get_pipeline_ref("compute_fri_domain_inverses")
        .ok_or_else(|| MetalError::FunctionError("compute_fri_domain_inverses".to_string()))?;

    let fold_pipeline = fold_eval_state
        .state
        .get_pipeline_ref("goldilocks_fri_fold_eval")
        .ok_or_else(|| MetalError::FunctionError("goldilocks_fri_fold_eval".to_string()))?;

    // --- Single command buffer for both dispatches ---
    let command_buffer = domain_inv_state.state.command_queue().new_command_buffer();

    // Encoder 1: Domain inverses
    {
        let inv_threads_per_group = domain_inv_state.max_threads.min(256);
        let inv_thread_groups = (half_len as u64).div_ceil(inv_threads_per_group);

        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(inv_pipeline);
        encoder.set_buffer(0, Some(&buf_h_inv), 0);
        encoder.set_buffer(1, Some(&buf_omega_inv), 0);
        encoder.set_buffer(2, Some(&buf_inv_x), 0);
        encoder.set_buffer(3, Some(&buf_inv_params), 0);
        encoder.dispatch_thread_groups(
            MTLSize::new(inv_thread_groups, 1, 1),
            MTLSize::new(inv_threads_per_group, 1, 1),
        );
        encoder.end_encoding();
    }

    // Encoder 2: Fold (reads buf_inv_x written by encoder 1; Metal guarantees ordering)
    {
        let fold_threads_per_group = fold_eval_state.max_threads.min(256);
        let fold_thread_groups = (half_len as u64).div_ceil(fold_threads_per_group);

        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(fold_pipeline);
        encoder.set_buffer(0, Some(evals_buffer), 0);
        encoder.set_buffer(1, Some(&buf_output), 0);
        encoder.set_buffer(2, Some(&buf_beta), 0);
        encoder.set_buffer(3, Some(&buf_inv_x), 0);
        encoder.set_buffer(4, Some(&buf_fold_params), 0);
        encoder.dispatch_thread_groups(
            MTLSize::new(fold_thread_groups, 1, 1),
            MTLSize::new(fold_threads_per_group, 1, 1),
        );
        encoder.end_encoding();
    }

    command_buffer.commit();
    command_buffer.wait_until_completed();

    if command_buffer.status() == metal::MTLCommandBufferStatus::Error {
        return Err(MetalError::ExecutionError(
            "GPU fused domain-inv+fold error".to_string(),
        ));
    }

    Ok((buf_output, half_len, buf_inv_x))
}

/// Fused stride-2-square-inverse + eval-domain fold in a single GPU command buffer.
///
/// For FRI layers 2+: derives domain inverses from the previous layer's `inv_x` buffer
/// via stride-2 squaring (`inv_x_next[j] = inv_x_prev[2*j]^2`, 1 mul/element) instead
/// of recomputing from scratch via exponentiation-by-squaring (log₂(N) muls/element).
///
/// Returns `(folded_buffer, half_len, new_inv_x_buffer)`.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn gpu_fold_with_squared_inv(
    evals_buffer: &metal::Buffer,
    num_evals: usize,
    beta: &FieldElement<Goldilocks64Field>,
    prev_inv_x: &metal::Buffer,
    fold_eval_state: &FriFoldEvalState,
    square_inv_state: &FriSquareInvState,
) -> Result<(metal::Buffer, usize, metal::Buffer), MetalError> {
    use metal::MTLSize;

    assert!(
        num_evals >= 2 && num_evals.is_multiple_of(2),
        "eval-domain fold requires even number of evaluations"
    );

    let half_len = num_evals / 2;

    // --- Stride-2 square inverse output buffer ---
    let buf_inv_x = square_inv_state
        .state
        .alloc_buffer(half_len * std::mem::size_of::<u64>())?;
    let sq_params = FriSquareInvParams {
        len: half_len as u32,
    };
    let buf_sq_params = square_inv_state
        .state
        .alloc_buffer_with_data(std::slice::from_ref(&sq_params))?;

    // --- Fold parameters ---
    let beta_u64 = Goldilocks64Field::canonical(beta.value());
    let buf_output = fold_eval_state
        .state
        .alloc_buffer(half_len * std::mem::size_of::<u64>())?;
    let buf_beta = fold_eval_state
        .state
        .alloc_buffer_with_data(std::slice::from_ref(&beta_u64))?;
    let fold_params = FriFoldEvalParams {
        half_len: half_len as u32,
    };
    let buf_fold_params = fold_eval_state
        .state
        .alloc_buffer_with_data(std::slice::from_ref(&fold_params))?;

    // --- Get pipelines ---
    let sq_pipeline = square_inv_state
        .state
        .get_pipeline_ref("fri_square_inverses")
        .ok_or_else(|| MetalError::FunctionError("fri_square_inverses".to_string()))?;

    let fold_pipeline = fold_eval_state
        .state
        .get_pipeline_ref("goldilocks_fri_fold_eval")
        .ok_or_else(|| MetalError::FunctionError("goldilocks_fri_fold_eval".to_string()))?;

    // --- Single command buffer: stride-2 square inverses then fold ---
    let command_buffer = square_inv_state.state.command_queue().new_command_buffer();

    // Encoder 1: Stride-2 square previous layer's inverses
    {
        let threads_per_group = square_inv_state.max_threads.min(256);
        let thread_groups = (half_len as u64).div_ceil(threads_per_group);

        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(sq_pipeline);
        encoder.set_buffer(0, Some(prev_inv_x), 0);
        encoder.set_buffer(1, Some(&buf_inv_x), 0);
        encoder.set_buffer(2, Some(&buf_sq_params), 0);
        encoder.dispatch_thread_groups(
            MTLSize::new(thread_groups, 1, 1),
            MTLSize::new(threads_per_group, 1, 1),
        );
        encoder.end_encoding();
    }

    // Encoder 2: Fold (reads stride-2 squared inv_x from encoder 1)
    {
        let fold_threads_per_group = fold_eval_state.max_threads.min(256);
        let fold_thread_groups = (half_len as u64).div_ceil(fold_threads_per_group);

        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(fold_pipeline);
        encoder.set_buffer(0, Some(evals_buffer), 0);
        encoder.set_buffer(1, Some(&buf_output), 0);
        encoder.set_buffer(2, Some(&buf_beta), 0);
        encoder.set_buffer(3, Some(&buf_inv_x), 0);
        encoder.set_buffer(4, Some(&buf_fold_params), 0);
        encoder.dispatch_thread_groups(
            MTLSize::new(fold_thread_groups, 1, 1),
            MTLSize::new(fold_threads_per_group, 1, 1),
        );
        encoder.end_encoding();
    }

    command_buffer.commit();
    command_buffer.wait_until_completed();

    if command_buffer.status() == metal::MTLCommandBufferStatus::Error {
        return Err(MetalError::ExecutionError(
            "GPU fused stride-2-square-inv+fold error".to_string(),
        ));
    }

    Ok((buf_output, half_len, buf_inv_x))
}

/// Fused stride-2-square-inverse + eval-domain fold + Merkle commit in a single
/// GPU command buffer.
///
/// For FRI layers 1+: combines the fold from [`gpu_fold_with_squared_inv`] with the
/// Merkle tree construction from [`gpu_fri_layer_commit_from_buffer`] into one
/// command buffer submission, eliminating one `wait_until_completed()` per FRI layer.
///
/// Returns `(folded_buffer, half_len, new_inv_x_buffer, merkle_tree, root)`.
#[cfg(all(target_os = "macos", feature = "metal"))]
#[allow(clippy::too_many_arguments)]
pub fn gpu_fold_and_commit_fused(
    evals_buffer: &metal::Buffer,
    num_evals: usize,
    beta: &FieldElement<Goldilocks64Field>,
    prev_inv_x: &metal::Buffer,
    fold_eval_state: &FriFoldEvalState,
    square_inv_state: &FriSquareInvState,
    keccak_state: &GpuMerkleState,
) -> Result<
    (
        metal::Buffer,
        usize,
        metal::Buffer,
        BatchedMerkleTree<Goldilocks64Field>,
        Commitment,
    ),
    MetalError,
> {
    use metal::{MTLCommandBufferStatus, MTLResourceOptions, MTLSize};

    assert!(
        num_evals >= 2 && num_evals.is_multiple_of(2),
        "eval-domain fold requires even number of evaluations"
    );

    let half_len = num_evals / 2;

    // --- Stride-2 square inverse output buffer ---
    let buf_inv_x = square_inv_state
        .state
        .alloc_buffer(half_len * std::mem::size_of::<u64>())?;
    let sq_params = FriSquareInvParams {
        len: half_len as u32,
    };
    let buf_sq_params = square_inv_state
        .state
        .alloc_buffer_with_data(std::slice::from_ref(&sq_params))?;

    // --- Fold parameters ---
    let beta_u64 = Goldilocks64Field::canonical(beta.value());
    let buf_output = fold_eval_state
        .state
        .alloc_buffer(half_len * std::mem::size_of::<u64>())?;
    let buf_beta = fold_eval_state
        .state
        .alloc_buffer_with_data(std::slice::from_ref(&beta_u64))?;
    let fold_params = FriFoldEvalParams {
        half_len: half_len as u32,
    };
    let buf_fold_params = fold_eval_state
        .state
        .alloc_buffer_with_data(std::slice::from_ref(&fold_params))?;

    // --- Merkle tree allocation ---
    let num_leaves = half_len / 2; // FRI paired leaves
    let num_cols = 2usize;
    let leaves_len = num_leaves.next_power_of_two();
    let total_nodes = 2 * leaves_len - 1;
    let tree_buf = keccak_state.state.device().new_buffer(
        (total_nodes * 32) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // --- Get pipelines ---
    let sq_pipeline = square_inv_state
        .state
        .get_pipeline_ref("fri_square_inverses")
        .ok_or_else(|| MetalError::FunctionError("fri_square_inverses".to_string()))?;

    let fold_pipeline = fold_eval_state
        .state
        .get_pipeline_ref("goldilocks_fri_fold_eval")
        .ok_or_else(|| MetalError::FunctionError("goldilocks_fri_fold_eval".to_string()))?;

    // --- Single command buffer: square-inv → fold → hash leaves → build tree ---
    let command_buffer = square_inv_state.state.command_queue().new_command_buffer();

    // Encoder 1: Stride-2 square previous layer's inverses
    {
        let threads_per_group = square_inv_state.max_threads.min(256);
        let thread_groups = (half_len as u64).div_ceil(threads_per_group);

        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(sq_pipeline);
        encoder.set_buffer(0, Some(prev_inv_x), 0);
        encoder.set_buffer(1, Some(&buf_inv_x), 0);
        encoder.set_buffer(2, Some(&buf_sq_params), 0);
        encoder.dispatch_thread_groups(
            MTLSize::new(thread_groups, 1, 1),
            MTLSize::new(threads_per_group, 1, 1),
        );
        encoder.end_encoding();
    }

    // Encoder 2: Fold (reads stride-2 squared inv_x from encoder 1)
    {
        let fold_threads_per_group = fold_eval_state.max_threads.min(256);
        let fold_thread_groups = (half_len as u64).div_ceil(fold_threads_per_group);

        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(fold_pipeline);
        encoder.set_buffer(0, Some(evals_buffer), 0);
        encoder.set_buffer(1, Some(&buf_output), 0);
        encoder.set_buffer(2, Some(&buf_beta), 0);
        encoder.set_buffer(3, Some(&buf_inv_x), 0);
        encoder.set_buffer(4, Some(&buf_fold_params), 0);
        encoder.dispatch_thread_groups(
            MTLSize::new(fold_thread_groups, 1, 1),
            MTLSize::new(fold_threads_per_group, 1, 1),
        );
        encoder.end_encoding();
    }

    // Encoders 3+: Hash leaves + build tree levels (reads fold output from encoder 2)
    crate::metal::merkle::encode_hash_and_build_tree(
        command_buffer,
        &buf_output,
        num_leaves,
        num_cols,
        &tree_buf,
        leaves_len,
        keccak_state,
    )?;

    // --- ONE sync point ---
    command_buffer.commit();
    command_buffer.wait_until_completed();

    if command_buffer.status() == MTLCommandBufferStatus::Error {
        return Err(MetalError::ExecutionError(
            "GPU fused fold+commit error".to_string(),
        ));
    }

    // Read back tree nodes
    let mut nodes = vec![[0u8; 32]; total_nodes];
    unsafe {
        let ptr = tree_buf.contents() as *const u8;
        std::ptr::copy_nonoverlapping(ptr, nodes.as_mut_ptr() as *mut u8, total_nodes * 32);
    }

    // Padding (FRI folded evals are always power-of-two, so this is typically a no-op)
    if num_leaves < leaves_len {
        let last_real = leaves_len - 1 + num_leaves - 1;
        let pad_hash = nodes[last_real];
        for node in nodes
            .iter_mut()
            .take(leaves_len - 1 + leaves_len)
            .skip(last_real + 1)
        {
            *node = pad_hash;
        }
    }

    let root = nodes[0];
    let tree = BatchedMerkleTree::<Goldilocks64Field>::from_nodes(nodes)
        .ok_or_else(|| MetalError::ExecutionError("Failed to build FRI Merkle tree".into()))?;

    Ok((buf_output, half_len, buf_inv_x, tree, root))
}

/// Pre-compiled Metal state for FRI domain inverse precomputation.
///
/// Caches the compiled pipeline for `compute_fri_domain_inverses`.
/// This kernel computes `inv_x[i] = h^{-1} * ω^{-bitrev(i)}` using known constants,
/// avoiding expensive per-element Fermat inversions.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub struct FriDomainInvState {
    state: DynamicMetalState,
    max_threads: u64,
}

#[cfg(all(target_os = "macos", feature = "metal"))]
impl FriDomainInvState {
    pub fn new() -> Result<Self, MetalError> {
        let combined_source = format!("{}\n{}", FRI_GOLDILOCKS_FIELD_HEADER, FRI_DOMAIN_INV_SHADER);
        let mut state = DynamicMetalState::new()?;
        state.load_library(&combined_source)?;
        let max_threads = state.prepare_pipeline("compute_fri_domain_inverses")?;
        Ok(Self { state, max_threads })
    }

    pub fn from_device_and_queue(
        device: &metal::Device,
        queue: &metal::CommandQueue,
    ) -> Result<Self, MetalError> {
        let combined_source = format!("{}\n{}", FRI_GOLDILOCKS_FIELD_HEADER, FRI_DOMAIN_INV_SHADER);
        let mut state = DynamicMetalState::from_device_and_queue(device, queue);
        state.load_library(&combined_source)?;
        let max_threads = state.prepare_pipeline("compute_fri_domain_inverses")?;
        Ok(Self { state, max_threads })
    }
}

/// Compute FRI domain inverses on GPU from known constants.
///
/// For a coset domain `{h * ω^i}` in bit-reversed order, computes:
///   `inv_x[i] = h^{-1} * ω^{-bitrev(i)}`  for i=0..domain_size/2-1
///
/// This uses only multiplications (no Fermat inversions), since `h^{-1}`
/// and `ω^{-1}` are known constants computed on the CPU.
///
/// For subsequent FRI layers, use [`gpu_square_fri_inverses`] instead:
///   `inv_x_next[i] = inv_x[i]^2`  (since `1/x^2 = (1/x)^2`).
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn gpu_compute_fri_domain_inverses(
    h_inv: &FieldElement<Goldilocks64Field>,
    omega_inv: &FieldElement<Goldilocks64Field>,
    domain_size: usize,
    state: &FriDomainInvState,
) -> Result<metal::Buffer, MetalError> {
    use metal::MTLSize;

    let half_len = domain_size / 2;
    let h_inv_u64 = Goldilocks64Field::canonical(h_inv.value());
    let omega_inv_u64 = Goldilocks64Field::canonical(omega_inv.value());

    let buf_h_inv = state
        .state
        .alloc_buffer_with_data(std::slice::from_ref(&h_inv_u64))?;
    let buf_omega_inv = state
        .state
        .alloc_buffer_with_data(std::slice::from_ref(&omega_inv_u64))?;
    let buf_output = state
        .state
        .alloc_buffer(half_len * std::mem::size_of::<u64>())?;
    let params = FriDomainInvParams {
        half_len: half_len as u32,
        log_half_len: half_len.trailing_zeros(),
    };
    let buf_params = state
        .state
        .alloc_buffer_with_data(std::slice::from_ref(&params))?;

    let pipeline = state
        .state
        .get_pipeline_ref("compute_fri_domain_inverses")
        .ok_or_else(|| MetalError::FunctionError("compute_fri_domain_inverses".to_string()))?;

    let threads_per_group = state.max_threads.min(256);
    let thread_groups = (half_len as u64).div_ceil(threads_per_group);

    let command_buffer = state.state.command_queue().new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(&buf_h_inv), 0);
    encoder.set_buffer(1, Some(&buf_omega_inv), 0);
    encoder.set_buffer(2, Some(&buf_output), 0);
    encoder.set_buffer(3, Some(&buf_params), 0);
    encoder.dispatch_thread_groups(
        MTLSize::new(thread_groups, 1, 1),
        MTLSize::new(threads_per_group, 1, 1),
    );
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    if command_buffer.status() == metal::MTLCommandBufferStatus::Error {
        return Err(MetalError::ExecutionError(
            "GPU FRI domain inverse error".to_string(),
        ));
    }

    Ok(buf_output)
}

/// Pre-compiled Metal state for FRI inverse squaring.
///
/// Caches the compiled pipeline for `fri_square_inverses`.
/// Used for subsequent FRI layers: `inv_x_next[i] = inv_x[i]^2`.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub struct FriSquareInvState {
    state: DynamicMetalState,
    max_threads: u64,
}

#[cfg(all(target_os = "macos", feature = "metal"))]
impl FriSquareInvState {
    pub fn new() -> Result<Self, MetalError> {
        let combined_source = format!("{}\n{}", FRI_GOLDILOCKS_FIELD_HEADER, FRI_SQUARE_INV_SHADER);
        let mut state = DynamicMetalState::new()?;
        state.load_library(&combined_source)?;
        let max_threads = state.prepare_pipeline("fri_square_inverses")?;
        Ok(Self { state, max_threads })
    }

    pub fn from_device_and_queue(
        device: &metal::Device,
        queue: &metal::CommandQueue,
    ) -> Result<Self, MetalError> {
        let combined_source = format!("{}\n{}", FRI_GOLDILOCKS_FIELD_HEADER, FRI_SQUARE_INV_SHADER);
        let mut state = DynamicMetalState::from_device_and_queue(device, queue);
        state.load_library(&combined_source)?;
        let max_threads = state.prepare_pipeline("fri_square_inverses")?;
        Ok(Self { state, max_threads })
    }
}

/// Stride-2 square FRI domain inverses on GPU for the next FRI layer.
///
/// Given `inv_x_prev[i]` from the previous layer (length `2 * output_len`),
/// computes `inv_x_next[j] = inv_x_prev[2*j]^2` for `j = 0..output_len-1`.
///
/// Why stride-2: The fold kernel pairs `evals[2*i]` and `evals[2*i+1]`, using
/// `inv_x[i] = 1/x[2*i]`. After folding, `result[i]` sits at `x[2*i]^2`.
/// The next fold pairs `result[2*j]` and `result[2*j+1]`, needing
/// `1/(x[4*j]^2) = inv_x_prev[2*j]^2`.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn gpu_square_fri_inverses(
    inv_x_buffer: &metal::Buffer,
    output_len: usize,
    state: &FriSquareInvState,
) -> Result<metal::Buffer, MetalError> {
    use metal::MTLSize;

    let buf_output = state
        .state
        .alloc_buffer(output_len * std::mem::size_of::<u64>())?;
    let params = FriSquareInvParams {
        len: output_len as u32,
    };
    let buf_params = state
        .state
        .alloc_buffer_with_data(std::slice::from_ref(&params))?;

    let pipeline = state
        .state
        .get_pipeline_ref("fri_square_inverses")
        .ok_or_else(|| MetalError::FunctionError("fri_square_inverses".to_string()))?;

    let threads_per_group = state.max_threads.min(256);
    let thread_groups = (output_len as u64).div_ceil(threads_per_group);

    let command_buffer = state.state.command_queue().new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(inv_x_buffer), 0);
    encoder.set_buffer(1, Some(&buf_output), 0);
    encoder.set_buffer(2, Some(&buf_params), 0);
    encoder.dispatch_thread_groups(
        MTLSize::new(thread_groups, 1, 1),
        MTLSize::new(threads_per_group, 1, 1),
    );
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    if command_buffer.status() == metal::MTLCommandBufferStatus::Error {
        return Err(MetalError::ExecutionError(
            "GPU FRI square inverses error".to_string(),
        ));
    }

    Ok(buf_output)
}

// =============================================================================
// GPU Bit-Reverse Permutation (buffer-to-buffer)
// =============================================================================

/// Bit-reverse permutation on a Metal buffer using the existing Goldilocks kernel.
///
/// Takes a buffer of u64 values in natural order, returns a new buffer in bit-reversed order.
/// Uses the `bitrev_permutation_Goldilocks` kernel from the math crate's shader library.
#[cfg(all(target_os = "macos", feature = "metal"))]
fn gpu_bitrev_buffer_to_buffer(
    input_buffer: &metal::Buffer,
    len: usize,
    state: &MetalState,
) -> Result<metal::Buffer, MetalError> {
    use metal::MTLSize;

    let pipeline = state.setup_pipeline("bitrev_permutation_Goldilocks")?;
    let output_buffer = state.alloc_buffer::<u64>(len);

    let command_buffer = state.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(input_buffer), 0);
    encoder.set_buffer(1, Some(&output_buffer), 0);

    let grid_size = MTLSize::new(len as u64, 1, 1);
    let threadgroup_size =
        MTLSize::new(pipeline.max_total_threads_per_threadgroup().min(256), 1, 1);
    encoder.dispatch_threads(grid_size, threadgroup_size);
    encoder.end_encoding();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    if command_buffer.status() == metal::MTLCommandBufferStatus::Error {
        return Err(MetalError::ExecutionError(
            "GPU bitrev permutation error".to_string(),
        ));
    }

    Ok(output_buffer)
}

// =============================================================================
// GPU FRI Fold kernel for Fp3 (Goldilocks degree-3 extension)
// =============================================================================

/// Source code for the Fp3 FRI fold Metal kernel.
#[cfg(all(target_os = "macos", feature = "metal"))]
const FRI_FOLD_FP3_SHADER: &str = include_str!("../shaders/fri_fold_fp3.metal");

/// Pre-compiled Metal state for the Fp3 FRI fold kernel.
///
/// Caches the compiled pipeline for `goldilocks_fp3_fri_fold`.
/// Create once and reuse across all Fp3 FRI folding rounds.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub struct FriFoldFp3State {
    state: DynamicMetalState,
    fri_fold_max_threads: u64,
}

#[cfg(all(target_os = "macos", feature = "metal"))]
impl FriFoldFp3State {
    /// Compile the Fp3 FRI fold shader and prepare the pipeline.
    pub fn new() -> Result<Self, MetalError> {
        let combined_source = crate::metal::fp3::combined_fp3_source(FRI_FOLD_FP3_SHADER);

        let mut state = DynamicMetalState::new()?;
        state.load_library(&combined_source)?;
        let fri_fold_max_threads = state.prepare_pipeline("goldilocks_fp3_fri_fold")?;
        Ok(Self {
            state,
            fri_fold_max_threads,
        })
    }

    /// Compile the Fp3 FRI fold shader sharing a device and queue with an existing Metal state.
    pub fn from_device_and_queue(
        device: &metal::Device,
        queue: &metal::CommandQueue,
    ) -> Result<Self, MetalError> {
        let combined_source = crate::metal::fp3::combined_fp3_source(FRI_FOLD_FP3_SHADER);

        let mut state = DynamicMetalState::from_device_and_queue(device, queue);
        state.load_library(&combined_source)?;
        let fri_fold_max_threads = state.prepare_pipeline("goldilocks_fp3_fri_fold")?;
        Ok(Self {
            state,
            fri_fold_max_threads,
        })
    }
}

/// Perform Fp3 FRI fold on GPU from an existing Metal buffer.
///
/// Dispatches the `goldilocks_fp3_fri_fold` kernel:
/// `result[k] = 2 * (coeffs[2k] + beta * coeffs[2k+1])` in Fp3 arithmetic.
///
/// # Arguments
///
/// - `coeffs_buffer`: Input Metal buffer containing `num_coeffs` Fp3 elements (3 u64s each)
/// - `num_coeffs`: Number of Fp3 coefficients
/// - `beta`: The FRI folding challenge (Fp3)
/// - `state`: Pre-compiled Fp3 FRI fold Metal state
///
/// # Returns
///
/// A tuple of (Metal Buffer, half_len) where the buffer contains the folded Fp3
/// coefficients (3 u64s each) and half_len = num_coeffs / 2.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn gpu_fold_polynomial_fp3(
    coeffs_buffer: &metal::Buffer,
    num_coeffs: usize,
    beta: &Fp3E,
    state: &FriFoldFp3State,
) -> Result<(metal::Buffer, usize), MetalError> {
    use metal::MTLSize;

    if num_coeffs <= 1 {
        // Edge case: 0 or 1 Fp3 coefficients.
        let val: [u64; 3] = if num_coeffs == 1 {
            let vals: Vec<u64> = MetalState::retrieve_contents(coeffs_buffer);
            let c0 = FieldElement::<Goldilocks64Field>::from(vals.first().copied().unwrap_or(0));
            let c1 = FieldElement::<Goldilocks64Field>::from(vals.get(1).copied().unwrap_or(0));
            let c2 = FieldElement::<Goldilocks64Field>::from(vals.get(2).copied().unwrap_or(0));
            let fe = Fp3E::new([c0, c1, c2]);
            let result = fe + fe; // 2 * fe
            let comps = result.value();
            [
                Goldilocks64Field::canonical(comps[0].value()),
                Goldilocks64Field::canonical(comps[1].value()),
                Goldilocks64Field::canonical(comps[2].value()),
            ]
        } else {
            [0u64; 3]
        };
        let buf = state.state.alloc_buffer_with_data(&val)?;
        return Ok((buf, 1));
    }

    // Pad to even if needed
    let (input_buf_owned, input_ref, padded_len) = if !num_coeffs.is_multiple_of(2) {
        let padded_len = num_coeffs + 1;
        // Each Fp3 element is 3 u64s
        let buf_padded = state
            .state
            .alloc_buffer(padded_len * 3 * std::mem::size_of::<u64>())?;
        let cmd = state.state.command_queue().new_command_buffer();
        let blit = cmd.new_blit_command_encoder();
        blit.copy_from_buffer(
            coeffs_buffer,
            0,
            &buf_padded,
            0,
            (num_coeffs * 3 * std::mem::size_of::<u64>()) as u64,
        );
        blit.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        (Some(buf_padded), None, padded_len)
    } else {
        (None, Some(coeffs_buffer), num_coeffs)
    };

    let actual_input: &metal::Buffer = match (&input_buf_owned, input_ref) {
        (Some(buf), _) => buf,
        (_, Some(buf)) => buf,
        _ => unreachable!(),
    };

    let half_len = padded_len / 2;

    // Pack beta as 3 u64s
    let beta_comps = beta.value();
    let beta_u64: [u64; 3] = [
        Goldilocks64Field::canonical(beta_comps[0].value()),
        Goldilocks64Field::canonical(beta_comps[1].value()),
        Goldilocks64Field::canonical(beta_comps[2].value()),
    ];

    let buf_output = state
        .state
        .alloc_buffer(half_len * 3 * std::mem::size_of::<u64>())?;
    let buf_beta = state.state.alloc_buffer_with_data(&beta_u64)?;
    let half_len_u32 = half_len as u32;
    let buf_half_len = state
        .state
        .alloc_buffer_with_data(std::slice::from_ref(&half_len_u32))?;

    let pipeline = state
        .state
        .get_pipeline_ref("goldilocks_fp3_fri_fold")
        .ok_or_else(|| MetalError::FunctionError("goldilocks_fp3_fri_fold".to_string()))?;

    let threads_per_group = state.fri_fold_max_threads.min(256);
    let thread_groups = (half_len as u64).div_ceil(threads_per_group);

    let command_buffer = state.state.command_queue().new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(actual_input), 0);
    encoder.set_buffer(1, Some(&buf_output), 0);
    encoder.set_buffer(2, Some(&buf_beta), 0);
    encoder.set_buffer(3, Some(&buf_half_len), 0);

    encoder.dispatch_thread_groups(
        MTLSize::new(thread_groups, 1, 1),
        MTLSize::new(threads_per_group, 1, 1),
    );
    encoder.end_encoding();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    if command_buffer.status() == metal::MTLCommandBufferStatus::Error {
        return Err(MetalError::ExecutionError(
            "GPU Fp3 FRI fold command buffer error".to_string(),
        ));
    }

    Ok((buf_output, half_len))
}

/// Perform Fp3 FRI fold on GPU from CPU polynomial data.
///
/// Converts Fp3 polynomial coefficients to flat u64 triples, uploads to GPU,
/// and dispatches the `goldilocks_fp3_fri_fold` kernel.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn gpu_fold_polynomial_fp3_from_cpu(
    poly: &Polynomial<Fp3E>,
    beta: &Fp3E,
    state: &FriFoldFp3State,
) -> Result<(metal::Buffer, usize), MetalError> {
    let coeffs = poly.coefficients();
    let num_coeffs = coeffs.len();

    // Convert Fp3 coefficients to flat u64 triples
    let coeffs_u64: Vec<u64> = coeffs
        .iter()
        .flat_map(|fe| {
            let comps = fe.value();
            [
                Goldilocks64Field::canonical(comps[0].value()),
                Goldilocks64Field::canonical(comps[1].value()),
                Goldilocks64Field::canonical(comps[2].value()),
            ]
        })
        .collect();

    let buf_input = state.state.alloc_buffer_with_data(&coeffs_u64)?;
    gpu_fold_polynomial_fp3(&buf_input, num_coeffs, beta, state)
}

/// GPU-accelerated FRI commit phase for Fp3 (degree-3 Goldilocks extension).
///
/// Replaces `fri::commit_phase` with GPU fold for Fp3 polynomials. Uses CPU
/// FFT via `fft_extension` for layer evaluation and CPU Merkle commit for now.
///
/// Polynomial Fp3 coefficients stay on GPU Metal buffers between fold rounds.
#[cfg(all(target_os = "macos", feature = "metal"))]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
pub fn gpu_fri_commit_phase_fp3(
    number_layers: usize,
    p_0: Polynomial<Fp3E>,
    transcript: &mut impl IsStarkTranscript<Fp3, Goldilocks64Field>,
    coset_offset: &FieldElement<Goldilocks64Field>,
    domain_size: usize,
    fri_fold_state: &FriFoldFp3State,
) -> Result<(Fp3E, Vec<FriLayer<Fp3, BatchedMerkleTreeBackend<Fp3>>>), ProvingError> {
    let mut domain_size = domain_size;
    let mut fri_layer_list = Vec::with_capacity(number_layers);
    let mut coset_offset = *coset_offset;

    let mut current_buffer: Option<(metal::Buffer, usize)> = None;
    let mut current_poly_cpu: Option<Polynomial<Fp3E>> = Some(p_0);

    for _ in 1..number_layers {
        let zeta: Fp3E = transcript.sample_field_element();
        coset_offset = coset_offset.square();
        domain_size /= 2;

        // Fold polynomial on GPU
        let (folded_buf, folded_len) = if let Some(poly) = current_poly_cpu.take() {
            gpu_fold_polynomial_fp3_from_cpu(&poly, &zeta, fri_fold_state)
                .map_err(|e| ProvingError::FieldOperationError(format!("GPU Fp3 FRI fold: {e}")))?
        } else {
            let (buf, len) = current_buffer.take().unwrap();
            gpu_fold_polynomial_fp3(&buf, len, &zeta, fri_fold_state)
                .map_err(|e| ProvingError::FieldOperationError(format!("GPU Fp3 FRI fold: {e}")))?
        };

        // Read back Fp3 coefficients from GPU for FFT + Merkle (CPU path for now)
        let coeffs_u64: Vec<u64> = MetalState::retrieve_contents(&folded_buf);
        let coeffs: Vec<Fp3E> = coeffs_u64
            .chunks(3)
            .map(|chunk| {
                Fp3E::new([
                    FieldElement::<Goldilocks64Field>::from(chunk[0]),
                    FieldElement::<Goldilocks64Field>::from(chunk[1]),
                    FieldElement::<Goldilocks64Field>::from(chunk[2]),
                ])
            })
            .collect();
        let poly = Polynomial::new(&coeffs);
        let current_layer = fri::new_fri_layer(&poly, &coset_offset, domain_size)?;

        current_buffer = Some((folded_buf, folded_len));
        let commitment = current_layer.merkle_tree.root;
        fri_layer_list.push(current_layer);
        transcript.append_bytes(&commitment);
    }

    // Final fold to get the last value
    let zeta: Fp3E = transcript.sample_field_element();
    let (last_buf, _last_len) = if let Some(poly) = current_poly_cpu.take() {
        gpu_fold_polynomial_fp3_from_cpu(&poly, &zeta, fri_fold_state)
            .map_err(|e| ProvingError::FieldOperationError(format!("GPU Fp3 FRI fold: {e}")))?
    } else {
        let (buf, len) = current_buffer.take().unwrap();
        gpu_fold_polynomial_fp3(&buf, len, &zeta, fri_fold_state)
            .map_err(|e| ProvingError::FieldOperationError(format!("GPU Fp3 FRI fold: {e}")))?
    };

    let last_u64: Vec<u64> = MetalState::retrieve_contents(&last_buf);
    let last_value = if last_u64.len() >= 3 {
        Fp3E::new([
            FieldElement::<Goldilocks64Field>::from(last_u64[0]),
            FieldElement::<Goldilocks64Field>::from(last_u64[1]),
            FieldElement::<Goldilocks64Field>::from(last_u64[2]),
        ])
    } else {
        Fp3E::zero()
    };
    transcript.append_field_element(&last_value);

    Ok((last_value, fri_layer_list))
}

/// Type alias matching the CPU prover's deep polynomial openings type.
pub type DeepPolynomialOpenings<F, E> = Vec<DeepPolynomialOpening<F, E>>;

/// Result of GPU Phase 4 (FRI round).
///
/// This is the GPU equivalent of `Round4<Field, FieldExtension>` from the CPU prover.
/// Contains the FRI last value, layer commitments, query decommitments, deep polynomial
/// openings (Merkle proofs for trace and composition poly evaluations), and the grinding
/// nonce.
pub struct GpuRound4Result<F: IsField>
where
    FieldElement<F>: AsBytes,
{
    /// The final constant value from FRI folding.
    pub fri_last_value: FieldElement<F>,
    /// Merkle roots of each FRI inner layer.
    pub fri_layers_merkle_roots: Vec<Commitment>,
    /// Merkle opening proofs for trace and composition polynomial evaluations at query points.
    pub deep_poly_openings: DeepPolynomialOpenings<F, F>,
    /// FRI query decommitments (symmetric evaluations + auth paths for each layer).
    pub query_list: Vec<FriDecommitment<F>>,
    /// Grinding nonce (None if grinding_factor == 0).
    pub nonce: Option<u64>,
}

/// Executes GPU Phase 4 of the STARK prover: DEEP composition + FRI + queries.
///
/// This mirrors `round_4_compute_and_run_fri_on_the_deep_composition_polynomial` from
/// the CPU prover:
///
/// 1. Sample gamma from transcript and compute deep composition coefficients
/// 2. Compute DEEP composition polynomial (CPU)
/// 3. Run FRI commit phase (iterative folding + Merkle commits)
/// 4. Grinding: find nonce if security_bits > 0
/// 5. Sample query indexes
/// 6. Run FRI query phase
/// 7. Extract FRI Merkle roots from layers
/// 8. Open deep composition poly (get Merkle proofs for trace and composition poly)
/// 9. Assemble and return GpuRound4Result
///
/// # Type Parameters
///
/// - `F`: The base field (must equal the extension field for our GPU prover)
/// - `A`: An AIR whose `Field` and `FieldExtension` are both `F`
///
/// # Errors
///
/// Returns `ProvingError` if FRI, grinding, or Merkle operations fail.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn gpu_round_4<F, A>(
    air: &A,
    domain: &Domain<F>,
    round_1_result: &GpuRound1Result<F>,
    round_2_result: &GpuRound2Result<F>,
    round_3_result: &GpuRound3Result<F>,
    transcript: &mut impl IsStarkTranscript<F, F>,
) -> Result<GpuRound4Result<F>, ProvingError>
where
    F: IsFFTField + IsSubFieldOf<F> + Send + Sync,
    F::BaseType: Copy,
    FieldElement<F>: AsBytes + Sync + Send,
    A: AIR<Field = F, FieldExtension = F>,
{
    // Step 1: Sample gamma and compute deep composition coefficients.
    // These are powers of gamma used to linearly combine the trace and composition
    // polynomial terms into the DEEP composition polynomial.
    let gamma = transcript.sample_field_element();

    let n_terms_composition_poly = round_2_result.lde_composition_poly_evaluations.len();
    let num_terms_trace =
        air.context().transition_offsets.len() * air.step_size() * air.context().trace_columns;

    let mut deep_composition_coefficients: Vec<_> =
        core::iter::successors(Some(FieldElement::one()), |x| Some(x * &gamma))
            .take(n_terms_composition_poly + num_terms_trace)
            .collect();

    // Split: first `num_terms_trace` coefficients are for trace terms,
    // remainder are for composition poly terms.
    let trace_term_coeffs: Vec<_> = deep_composition_coefficients
        .drain(..num_terms_trace)
        .collect::<Vec<_>>()
        .chunks(air.context().transition_offsets.len() * air.step_size())
        .map(|chunk| chunk.to_vec())
        .collect();

    let composition_gammas = deep_composition_coefficients;

    // Step 2: Compute DEEP composition polynomial (CPU).
    // Combine main + aux trace polys (since F == FieldExtension, no conversion needed).
    let mut all_trace_polys = round_1_result.main_trace_polys.clone();
    all_trace_polys.extend(round_1_result.aux_trace_polys.iter().cloned());

    let deep_composition_poly = compute_deep_composition_poly(
        &all_trace_polys,
        &round_2_result.composition_poly_parts,
        &round_3_result.trace_ood_evaluations,
        &round_3_result.composition_poly_parts_ood_evaluation,
        &round_3_result.z,
        &domain.trace_primitive_root,
        &composition_gammas,
        &trace_term_coeffs,
    );

    // Step 3: Run FRI commit phase.
    let domain_size = domain.lde_roots_of_unity_coset.len();
    let (fri_last_value, fri_layers) = fri::commit_phase::<F, F>(
        domain.root_order as usize,
        deep_composition_poly,
        transcript,
        &domain.coset_offset,
        domain_size,
    )?;

    // Step 4: Grinding.
    let security_bits = air.context().proof_options.grinding_factor;
    let mut nonce = None;
    if security_bits > 0 {
        let nonce_value = grinding::generate_nonce(&transcript.state(), security_bits)
            .ok_or(ProvingError::NonceNotFound(security_bits))?;
        transcript.append_bytes(&nonce_value.to_be_bytes());
        nonce = Some(nonce_value);
    }

    // Step 5: Sample query indexes.
    let number_of_queries = air.options().fri_number_of_queries;
    let domain_size_u64 = domain_size as u64;
    let iotas: Vec<usize> = (0..number_of_queries)
        .map(|_| transcript.sample_u64(domain_size_u64 >> 1) as usize)
        .collect();

    // Step 6: Run FRI query phase.
    let query_list = fri::query_phase(&fri_layers, &iotas)?;

    // Step 7: Extract FRI Merkle roots from layers.
    let fri_layers_merkle_roots: Vec<_> = fri_layers
        .iter()
        .map(|layer| layer.merkle_tree.root)
        .collect();

    // Step 8: Open deep composition poly (Merkle proofs for trace + composition poly).
    let deep_poly_openings =
        open_deep_composition_poly(domain, round_1_result, round_2_result, &iotas)?;

    Ok(GpuRound4Result {
        fri_last_value,
        fri_layers_merkle_roots,
        deep_poly_openings,
        query_list,
        nonce,
    })
}

/// Computes the DEEP composition polynomial.
///
/// The DEEP composition polynomial is a linear combination of:
/// - H terms: `sum_i gamma_i * (H_i(X) - H_i(z^N)) / (X - z^N)`
/// - Trace terms: `sum_jk gamma_jk * (t_j(X) - t_j(z*g^k)) / (X - z*g^k)`
///
/// where each division is performed via Ruffini division (synthetic division by a
/// linear factor).
#[allow(clippy::too_many_arguments)]
fn compute_deep_composition_poly<F>(
    trace_polys: &[Polynomial<FieldElement<F>>],
    composition_poly_parts: &[Polynomial<FieldElement<F>>],
    trace_ood_evaluations: &Table<F>,
    composition_poly_ood_evaluations: &[FieldElement<F>],
    z: &FieldElement<F>,
    primitive_root: &FieldElement<F>,
    composition_gammas: &[FieldElement<F>],
    trace_term_coeffs: &[Vec<FieldElement<F>>],
) -> Polynomial<FieldElement<F>>
where
    F: IsFFTField + IsSubFieldOf<F>,
{
    let z_power = z.pow(composition_poly_parts.len());

    // H terms: sum_i gamma_i * (H_i(X) - H_i(z^N)) / (X - z^N)
    let mut h_terms = Polynomial::zero();
    for (i, part) in composition_poly_parts.iter().enumerate() {
        let h_i_eval = &composition_poly_ood_evaluations[i];
        let h_i_term = &composition_gammas[i] * (part - h_i_eval);
        h_terms += h_i_term;
    }
    debug_assert_eq!(h_terms.evaluate(&z_power), FieldElement::zero());
    h_terms.ruffini_division_inplace(&z_power);

    // Trace terms: sum_jk gamma_jk * (t_j(X) - t_j(z*g^k)) / (X - z*g^k)
    let trace_evaluations_columns = trace_ood_evaluations.columns();

    // Pre-compute z_shifted values: z * g^k for each frame offset
    let num_offsets = trace_ood_evaluations.height;
    let z_shifted_values: Vec<FieldElement<F>> = (0..num_offsets)
        .map(|offset| primitive_root.pow(offset) * z)
        .collect();

    let trace_terms =
        trace_polys
            .iter()
            .enumerate()
            .fold(Polynomial::zero(), |accumulator, (i, t_j)| {
                let gammas_i = &trace_term_coeffs[i];
                let trace_evaluations_i = &trace_evaluations_columns[i];

                let trace_int = trace_evaluations_i
                    .iter()
                    .zip(&z_shifted_values)
                    .zip(gammas_i)
                    .fold(
                        Polynomial::zero(),
                        |trace_agg, ((trace_term_poly_evaluation, z_shifted), trace_gamma)| {
                            let mut poly = t_j - trace_term_poly_evaluation;
                            poly.ruffini_division_inplace(z_shifted);
                            trace_agg + poly * trace_gamma
                        },
                    );
                accumulator + trace_int
            });

    h_terms + trace_terms
}

/// Opens trace polynomials at a given query index.
///
/// For a given challenge index, computes `index = challenge * 2` and `index_sym = challenge * 2 + 1`
/// (the symmetric pair), retrieves Merkle proofs from the tree, and extracts the
/// evaluations from the column-major LDE data using bit-reversed indexing.
fn open_trace_polys<F>(
    domain_size: usize,
    tree: &BatchedMerkleTree<F>,
    lde_evaluations: &[Vec<FieldElement<F>>],
    challenge: usize,
) -> Result<PolynomialOpenings<F>, ProvingError>
where
    F: IsField,
    FieldElement<F>: AsBytes + Sync + Send,
{
    let index = challenge * 2;
    let index_sym = challenge * 2 + 1;

    let proof = tree.get_proof_by_pos(index).ok_or_else(|| {
        ProvingError::MerkleTreeError(format!("Failed to get proof at position {}", index))
    })?;
    let proof_sym = tree.get_proof_by_pos(index_sym).ok_or_else(|| {
        ProvingError::MerkleTreeError(format!("Failed to get proof at position {}", index_sym))
    })?;

    let actual_index = reverse_index(index, domain_size as u64);
    let actual_index_sym = reverse_index(index_sym, domain_size as u64);

    let evaluations: Vec<_> = lde_evaluations
        .iter()
        .map(|col| col[actual_index].clone())
        .collect();
    let evaluations_sym: Vec<_> = lde_evaluations
        .iter()
        .map(|col| col[actual_index_sym].clone())
        .collect();

    Ok(PolynomialOpenings {
        proof,
        proof_sym,
        evaluations,
        evaluations_sym,
    })
}

/// Opens composition polynomial at a given query index.
///
/// The composition polynomial Merkle tree uses a special paired-row layout
/// where consecutive bit-reversed evaluations are merged. This function retrieves
/// the Merkle proof and extracts evaluations accordingly.
fn open_composition_poly<F>(
    composition_poly_merkle_tree: &BatchedMerkleTree<F>,
    lde_composition_poly_evaluations: &[Vec<FieldElement<F>>],
    index: usize,
) -> Result<PolynomialOpenings<F>, ProvingError>
where
    F: IsField,
    FieldElement<F>: AsBytes + Sync + Send,
{
    let proof = composition_poly_merkle_tree
        .get_proof_by_pos(index)
        .ok_or_else(|| {
            ProvingError::MerkleTreeError(format!("Failed to get proof at position {}", index))
        })?;

    let lde_composition_poly_parts_evaluation: Vec<_> = lde_composition_poly_evaluations
        .iter()
        .flat_map(|part| {
            vec![
                part[reverse_index(index * 2, part.len() as u64)].clone(),
                part[reverse_index(index * 2 + 1, part.len() as u64)].clone(),
            ]
        })
        .collect();

    Ok(PolynomialOpenings {
        proof: proof.clone(),
        proof_sym: proof,
        evaluations: lde_composition_poly_parts_evaluation
            .clone()
            .into_iter()
            .step_by(2)
            .collect(),
        evaluations_sym: lde_composition_poly_parts_evaluation
            .into_iter()
            .skip(1)
            .step_by(2)
            .collect(),
    })
}

/// Opens trace polynomials at a given query index using GPU Metal buffers via UMA.
///
/// Like [`open_trace_polys`] but reads evaluations directly from GPU buffer shared
/// memory instead of CPU Vecs. On Apple Silicon UMA, this is a direct pointer
/// dereference with no data copy.
#[cfg(all(target_os = "macos", feature = "metal"))]
fn open_trace_polys_from_buffers(
    domain_size: usize,
    tree: &BatchedMerkleTree<Goldilocks64Field>,
    lde_buffers: &[metal::Buffer],
    challenge: usize,
) -> Result<PolynomialOpenings<Goldilocks64Field>, ProvingError> {
    use crate::metal::phases::rap::read_element_from_buffer;

    let index = challenge * 2;
    let index_sym = challenge * 2 + 1;

    let proof = tree.get_proof_by_pos(index).ok_or_else(|| {
        ProvingError::MerkleTreeError(format!("Failed to get proof at position {}", index))
    })?;
    let proof_sym = tree.get_proof_by_pos(index_sym).ok_or_else(|| {
        ProvingError::MerkleTreeError(format!("Failed to get proof at position {}", index_sym))
    })?;

    let actual_index = reverse_index(index, domain_size as u64);
    let actual_index_sym = reverse_index(index_sym, domain_size as u64);

    let evaluations: Vec<_> = lde_buffers
        .iter()
        .map(|buf| read_element_from_buffer(buf, actual_index))
        .collect();
    let evaluations_sym: Vec<_> = lde_buffers
        .iter()
        .map(|buf| read_element_from_buffer(buf, actual_index_sym))
        .collect();

    Ok(PolynomialOpenings {
        proof,
        proof_sym,
        evaluations,
        evaluations_sym,
    })
}

/// Opens composition polynomial at a given query index using GPU Metal buffers via UMA.
///
/// Like [`open_composition_poly`] but reads evaluations directly from GPU buffer
/// shared memory.
#[cfg(all(target_os = "macos", feature = "metal"))]
fn open_composition_poly_from_buffers(
    composition_poly_merkle_tree: &BatchedMerkleTree<Goldilocks64Field>,
    lde_composition_buffers: &[metal::Buffer],
    index: usize,
) -> Result<PolynomialOpenings<Goldilocks64Field>, ProvingError> {
    use crate::metal::phases::rap::read_element_from_buffer;

    let proof = composition_poly_merkle_tree
        .get_proof_by_pos(index)
        .ok_or_else(|| {
            ProvingError::MerkleTreeError(format!("Failed to get proof at position {}", index))
        })?;

    let lde_composition_poly_parts_evaluation: Vec<_> = lde_composition_buffers
        .iter()
        .flat_map(|buf| {
            let part_len = buf.length() as usize / std::mem::size_of::<u64>();
            vec![
                read_element_from_buffer(buf, reverse_index(index * 2, part_len as u64)),
                read_element_from_buffer(buf, reverse_index(index * 2 + 1, part_len as u64)),
            ]
        })
        .collect();

    Ok(PolynomialOpenings {
        proof: proof.clone(),
        proof_sym: proof,
        evaluations: lde_composition_poly_parts_evaluation
            .clone()
            .into_iter()
            .step_by(2)
            .collect(),
        evaluations_sym: lde_composition_poly_parts_evaluation
            .into_iter()
            .skip(1)
            .step_by(2)
            .collect(),
    })
}

/// Opens the deep composition polynomial at a set of query indexes.
///
/// For each query index, this produces Merkle opening proofs for:
/// - Main trace polynomial evaluations
/// - Auxiliary trace polynomial evaluations (if present)
/// - Composition polynomial part evaluations
fn open_deep_composition_poly<F>(
    domain: &Domain<F>,
    round_1_result: &GpuRound1Result<F>,
    round_2_result: &GpuRound2Result<F>,
    indexes_to_open: &[usize],
) -> Result<DeepPolynomialOpenings<F, F>, ProvingError>
where
    F: IsFFTField,
    FieldElement<F>: AsBytes + Sync + Send,
{
    let domain_size = domain.lde_roots_of_unity_coset.len();
    let mut openings = Vec::new();

    for index in indexes_to_open.iter() {
        // Open main trace
        let main_trace_opening = open_trace_polys(
            domain_size,
            &round_1_result.main_merkle_tree,
            &round_1_result.main_lde_evaluations,
            *index,
        )?;

        // Open composition polynomial
        let composition_openings = open_composition_poly(
            &round_2_result.composition_poly_merkle_tree,
            &round_2_result.lde_composition_poly_evaluations,
            *index,
        )?;

        // Open auxiliary trace (if present)
        let aux_trace_polys = match (
            round_1_result.aux_merkle_tree.as_ref(),
            round_1_result.aux_lde_evaluations.is_empty(),
        ) {
            (Some(aux_tree), false) => Some(open_trace_polys(
                domain_size,
                aux_tree,
                &round_1_result.aux_lde_evaluations,
                *index,
            )?),
            _ => None,
        };

        openings.push(DeepPolynomialOpening {
            composition_poly: composition_openings,
            main_trace_polys: main_trace_opening,
            aux_trace_polys,
        });
    }

    Ok(openings)
}

/// Opens the deep composition polynomial using GPU Metal buffers via UMA zero-copy.
///
/// Goldilocks-specific variant of [`open_deep_composition_poly`] that reads evaluations
/// directly from GPU buffer shared memory instead of CPU Vecs. On Apple Silicon UMA,
/// this is a direct pointer dereference — no allocation or memcpy.
///
/// Requires GPU buffers to be populated in both round 1 and round 2 results.
#[cfg(all(target_os = "macos", feature = "metal"))]
fn open_deep_composition_poly_from_buffers(
    domain: &Domain<Goldilocks64Field>,
    round_1_result: &GpuRound1Result<Goldilocks64Field>,
    round_2_result: &GpuRound2Result<Goldilocks64Field>,
    indexes_to_open: &[usize],
) -> Result<DeepPolynomialOpenings<Goldilocks64Field, Goldilocks64Field>, ProvingError> {
    let domain_size = domain.lde_roots_of_unity_coset.len();

    let main_bufs = round_1_result
        .main_lde_gpu_buffers
        .as_ref()
        .expect("open_deep_composition_poly_from_buffers requires main_lde_gpu_buffers");
    let comp_bufs = round_2_result
        .lde_composition_gpu_buffers
        .as_ref()
        .expect("open_deep_composition_poly_from_buffers requires lde_composition_gpu_buffers");

    let mut openings = Vec::new();

    for index in indexes_to_open.iter() {
        // Open main trace from GPU buffers
        let main_trace_opening = open_trace_polys_from_buffers(
            domain_size,
            &round_1_result.main_merkle_tree,
            main_bufs,
            *index,
        )?;

        // Open composition polynomial from GPU buffers
        let composition_openings = open_composition_poly_from_buffers(
            &round_2_result.composition_poly_merkle_tree,
            comp_bufs,
            *index,
        )?;

        // Open auxiliary trace from GPU buffers (if present)
        let aux_trace_polys = match (
            round_1_result.aux_merkle_tree.as_ref(),
            round_1_result.aux_lde_gpu_buffers.as_ref(),
        ) {
            (Some(aux_tree), Some(aux_bufs)) if !aux_bufs.is_empty() => Some(
                open_trace_polys_from_buffers(domain_size, aux_tree, aux_bufs, *index)?,
            ),
            _ => None,
        };

        openings.push(DeepPolynomialOpening {
            composition_poly: composition_openings,
            main_trace_polys: main_trace_opening,
            aux_trace_polys,
        });
    }

    Ok(openings)
}

/// Minimum evaluation size to use GPU for FRI layers.
/// Below this threshold, CPU is faster due to GPU dispatch overhead.
#[cfg(all(target_os = "macos", feature = "metal"))]
const GPU_FRI_THRESHOLD: usize = 4096;

/// Create a single FRI layer using GPU FFT + GPU Keccak256 Merkle.
///
/// Uses `gpu_evaluate_offset_fft` for the FFT evaluation and
/// `gpu_fri_layer_commit` for the Merkle tree construction.
///
/// Retained for fallback/testing; the fused path uses `gpu_new_fri_layer_fused`.
#[cfg(all(target_os = "macos", feature = "metal"))]
#[allow(dead_code)]
fn gpu_new_fri_layer(
    poly: &Polynomial<FieldElement<Goldilocks64Field>>,
    coset_offset: &FieldElement<Goldilocks64Field>,
    domain_size: usize,
    gpu_state: &StarkMetalState,
    keccak_state: &GpuMerkleState,
) -> Result<FriLayer<Goldilocks64Field, BatchedMerkleTreeBackend<Goldilocks64Field>>, ProvingError>
{
    // GPU FFT: pad coefficients to domain_size, evaluate with blowup=1
    let coefficients = poly.coefficients();
    let mut padded_coeffs = coefficients.to_vec();
    padded_coeffs.resize(domain_size, FieldElement::zero());

    let mut evaluation =
        gpu_evaluate_offset_fft(&padded_coeffs, 1, coset_offset, gpu_state.inner())
            .map_err(|e| ProvingError::FieldOperationError(format!("GPU FFT in FRI: {e}")))?;

    in_place_bit_reverse_permute(&mut evaluation);

    // GPU Merkle: build paired tree from bit-reversed evaluations
    let (merkle_tree, _root) = gpu_fri_layer_commit(&evaluation, keccak_state)
        .map_err(|e| ProvingError::MerkleTreeError(format!("GPU Merkle in FRI: {e}")))?;

    Ok(FriLayer::new(
        &evaluation,
        merkle_tree,
        *coset_offset,
        domain_size,
    ))
}

/// Create a FRI layer entirely on GPU from coefficients already in a Metal buffer.
///
/// Pipeline: coset shift -> FFT -> Merkle hash, all on GPU.
/// Only reads back evaluations at the end (needed for FriLayer query data).
#[cfg(all(target_os = "macos", feature = "metal"))]
fn gpu_new_fri_layer_fused(
    coeffs_buffer: &metal::Buffer,
    num_coeffs: usize,
    coset_offset: &FieldElement<Goldilocks64Field>,
    domain_size: usize,
    gpu_state: &StarkMetalState,
    keccak_state: &GpuMerkleState,
    coset_state: &CosetShiftState,
) -> Result<FriLayer<Goldilocks64Field, BatchedMerkleTreeBackend<Goldilocks64Field>>, ProvingError>
{
    // Step 1: Coset shift + zero-pad on GPU: output[k] = coeffs[k] * offset^k
    let shifted_buffer = gpu_coset_shift_buffer_to_buffer(
        coeffs_buffer,
        num_coeffs,
        coset_offset,
        domain_size,
        coset_state,
    )
    .map_err(|e| ProvingError::FieldOperationError(format!("GPU coset shift in FRI: {e}")))?;

    // Step 2: Generate twiddles on GPU
    let order = domain_size.trailing_zeros() as u64;
    let twiddles_buffer = gen_twiddles_to_buffer::<Goldilocks64Field>(
        order,
        RootsConfig::BitReverse,
        gpu_state.inner(),
    )
    .map_err(|e| ProvingError::FieldOperationError(format!("GPU twiddle gen in FRI: {e}")))?;

    // Step 3: FFT on GPU (buffer-to-buffer, no CPU transfer)
    let eval_buffer = fft_buffer_to_buffer::<Goldilocks64Field>(
        &shifted_buffer,
        domain_size,
        &twiddles_buffer,
        gpu_state.inner(),
    )
    .map_err(|e| ProvingError::FieldOperationError(format!("GPU FFT in FRI: {e}")))?;

    // Step 4: Merkle commit directly from GPU buffer
    let (merkle_tree, _root) =
        gpu_fri_layer_commit_from_buffer(&eval_buffer, domain_size, keccak_state)
            .map_err(|e| ProvingError::MerkleTreeError(format!("GPU Merkle in FRI: {e}")))?;

    // Step 5: Read back evaluations for FriLayer (needed for query phase)
    let eval_u64: Vec<u64> = MetalState::retrieve_contents(&eval_buffer);
    let evaluation: Vec<FieldElement<Goldilocks64Field>> =
        eval_u64.into_iter().map(FieldElement::from).collect();

    Ok(FriLayer::new(
        &evaluation,
        merkle_tree,
        *coset_offset,
        domain_size,
    ))
}

/// GPU-accelerated FRI commit phase for Goldilocks field.
///
/// Replaces `fri::commit_phase` with GPU fold + GPU FFT + GPU Keccak256 Merkle for
/// large layers (>= `GPU_FRI_THRESHOLD` evaluations), falling back to CPU
/// for small layers where GPU dispatch overhead dominates.
///
/// Polynomial coefficients stay on GPU Metal buffers throughout, avoiding
/// CPU-GPU bouncing between FRI layers.
#[cfg(all(target_os = "macos", feature = "metal"))]
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub fn gpu_fri_commit_phase_goldilocks(
    number_layers: usize,
    p_0: Polynomial<FieldElement<Goldilocks64Field>>,
    transcript: &mut impl IsStarkTranscript<Goldilocks64Field, Goldilocks64Field>,
    coset_offset: &FieldElement<Goldilocks64Field>,
    domain_size: usize,
    gpu_state: &StarkMetalState,
    keccak_state: &GpuMerkleState,
    coset_state: &CosetShiftState,
    fri_fold_state: &FriFoldState,
) -> Result<
    (
        FieldElement<Goldilocks64Field>,
        Vec<FriLayer<Goldilocks64Field, BatchedMerkleTreeBackend<Goldilocks64Field>>>,
    ),
    ProvingError,
> {
    type F = Goldilocks64Field;
    type FpE = FieldElement<F>;

    let mut domain_size = domain_size;
    let mut fri_layer_list = Vec::with_capacity(number_layers);
    let mut coset_offset = *coset_offset;

    // Track current polynomial as GPU buffer. On first fold, upload from CPU.
    let mut current_buffer: Option<(metal::Buffer, usize)> = None;
    let mut current_poly_cpu: Option<Polynomial<FpE>> = Some(p_0);

    for _ in 1..number_layers {
        let zeta = transcript.sample_field_element();
        coset_offset = coset_offset.square();
        domain_size /= 2;

        // Fold polynomial on GPU
        let (folded_buf, folded_len) = if let Some(poly) = current_poly_cpu.take() {
            // First iteration: upload CPU polynomial and fold
            gpu_fold_polynomial_from_cpu(&poly, &zeta, fri_fold_state)
                .map_err(|e| ProvingError::FieldOperationError(format!("GPU FRI fold: {e}")))?
        } else {
            // Subsequent iterations: fold from existing GPU buffer
            let (buf, len) = current_buffer.take().unwrap();
            gpu_fold_polynomial(&buf, len, &zeta, fri_fold_state)
                .map_err(|e| ProvingError::FieldOperationError(format!("GPU FRI fold: {e}")))?
        };

        // Build FRI layer
        let current_layer = if domain_size >= GPU_FRI_THRESHOLD {
            gpu_new_fri_layer_fused(
                &folded_buf,
                folded_len,
                &coset_offset,
                domain_size,
                gpu_state,
                keccak_state,
                coset_state,
            )?
        } else {
            // Small layers: read back from GPU and use CPU path
            let coeffs_u64: Vec<u64> = MetalState::retrieve_contents(&folded_buf);
            let coeffs: Vec<FpE> = coeffs_u64.into_iter().map(FpE::from).collect();
            let poly = Polynomial::new(&coeffs);
            fri::new_fri_layer(&poly, &coset_offset, domain_size)?
        };

        current_buffer = Some((folded_buf, folded_len));
        let commitment = current_layer.merkle_tree.root;
        fri_layer_list.push(current_layer);
        transcript.append_bytes(&commitment);
    }

    // Final fold to get the last value
    let zeta = transcript.sample_field_element();
    let (last_buf, _last_len) = if let Some(poly) = current_poly_cpu.take() {
        gpu_fold_polynomial_from_cpu(&poly, &zeta, fri_fold_state)
            .map_err(|e| ProvingError::FieldOperationError(format!("GPU FRI fold: {e}")))?
    } else {
        let (buf, len) = current_buffer.take().unwrap();
        gpu_fold_polynomial(&buf, len, &zeta, fri_fold_state)
            .map_err(|e| ProvingError::FieldOperationError(format!("GPU FRI fold: {e}")))?
    };

    let last_u64: Vec<u64> = MetalState::retrieve_contents(&last_buf);
    let last_value = if last_u64.is_empty() {
        FpE::zero()
    } else {
        FpE::from(last_u64[0])
    };
    transcript.append_field_element(&last_value);

    Ok((last_value, fri_layer_list))
}

/// GPU-accelerated FRI commit phase starting from a GPU buffer.
///
/// Like [`gpu_fri_commit_phase_goldilocks`] but the initial polynomial coefficients
/// are already on a Metal buffer (e.g., from `gpu_compute_deep_composition_poly_to_buffer`),
/// so no CPU-to-GPU upload is needed for the first fold.
#[cfg(all(target_os = "macos", feature = "metal"))]
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub fn gpu_fri_commit_phase_goldilocks_from_buffer(
    number_layers: usize,
    p_0_buffer: metal::Buffer,
    p_0_len: usize,
    transcript: &mut impl IsStarkTranscript<Goldilocks64Field, Goldilocks64Field>,
    coset_offset: &FieldElement<Goldilocks64Field>,
    domain_size: usize,
    gpu_state: &StarkMetalState,
    keccak_state: &GpuMerkleState,
    coset_state: &CosetShiftState,
    fri_fold_state: &FriFoldState,
) -> Result<
    (
        FieldElement<Goldilocks64Field>,
        Vec<FriLayer<Goldilocks64Field, BatchedMerkleTreeBackend<Goldilocks64Field>>>,
    ),
    ProvingError,
> {
    type F = Goldilocks64Field;
    type FpE = FieldElement<F>;

    let mut domain_size = domain_size;
    let mut fri_layer_list = Vec::with_capacity(number_layers);
    let mut coset_offset = *coset_offset;
    let mut current_buffer: (metal::Buffer, usize) = (p_0_buffer, p_0_len);

    for _ in 1..number_layers {
        let zeta = transcript.sample_field_element();
        coset_offset = coset_offset.square();
        domain_size /= 2;

        // Fold polynomial on GPU from existing buffer
        let (buf, len) = current_buffer;
        let (folded_buf, folded_len) = gpu_fold_polynomial(&buf, len, &zeta, fri_fold_state)
            .map_err(|e| ProvingError::FieldOperationError(format!("GPU FRI fold: {e}")))?;

        // Build FRI layer
        let current_layer = if domain_size >= GPU_FRI_THRESHOLD {
            gpu_new_fri_layer_fused(
                &folded_buf,
                folded_len,
                &coset_offset,
                domain_size,
                gpu_state,
                keccak_state,
                coset_state,
            )?
        } else {
            // Small layers: read back from GPU and use CPU path
            let coeffs_u64: Vec<u64> = MetalState::retrieve_contents(&folded_buf);
            let coeffs: Vec<FpE> = coeffs_u64.into_iter().map(FpE::from).collect();
            let poly = Polynomial::new(&coeffs);
            fri::new_fri_layer(&poly, &coset_offset, domain_size)?
        };

        current_buffer = (folded_buf, folded_len);
        let commitment = current_layer.merkle_tree.root;
        fri_layer_list.push(current_layer);
        transcript.append_bytes(&commitment);
    }

    // Final fold to get the last value
    let zeta = transcript.sample_field_element();
    let (buf, len) = current_buffer;
    let (last_buf, _last_len) = gpu_fold_polynomial(&buf, len, &zeta, fri_fold_state)
        .map_err(|e| ProvingError::FieldOperationError(format!("GPU FRI fold: {e}")))?;

    let last_u64: Vec<u64> = MetalState::retrieve_contents(&last_buf);
    let last_value = if last_u64.is_empty() {
        FpE::zero()
    } else {
        FpE::from(last_u64[0])
    };
    transcript.append_field_element(&last_value);

    Ok((last_value, fri_layer_list))
}

/// GPU-accelerated FRI commit phase using eval-domain fold (no IFFT, no per-layer FFT).
///
/// Takes DEEP composition evaluations in natural order and produces `FriLayer` list
/// using eval-domain fold. This eliminates the IFFT and per-layer coset_shift+FFT
/// that the coefficient-domain path requires.
///
/// The evaluations are first bit-reversed, then for each FRI layer:
/// 1. Domain inverses are computed from known constants (h_inv, omega_inv)
/// 2. Merkle tree is committed from current evaluations
/// 3. Eval-domain fold produces next layer's evaluations
///
/// The output `FriLayer` list is identical to what the coefficient-domain path
/// produces, so the CPU verifier works unchanged.
#[cfg(all(target_os = "macos", feature = "metal"))]
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub fn gpu_fri_commit_phase_eval_domain(
    number_layers: usize,
    evals_buffer: metal::Buffer,
    evals_len: usize,
    transcript: &mut impl IsStarkTranscript<Goldilocks64Field, Goldilocks64Field>,
    coset_offset: &FieldElement<Goldilocks64Field>,
    domain_size: usize,
    gpu_state: &StarkMetalState,
    keccak_state: &GpuMerkleState,
    fold_eval_state: &FriFoldEvalState,
    domain_inv_state: &FriDomainInvState,
    square_inv_state: &FriSquareInvState,
) -> Result<
    (
        FieldElement<Goldilocks64Field>,
        Vec<FriLayer<Goldilocks64Field, BatchedMerkleTreeBackend<Goldilocks64Field>>>,
    ),
    ProvingError,
> {
    type F = Goldilocks64Field;
    type FpE = FieldElement<F>;

    let mut domain_size = domain_size;
    let mut fri_layer_list = Vec::with_capacity(number_layers);
    let mut coset_offset = *coset_offset;

    // Step 1: Bit-reverse the evaluations from natural order to bit-reversed order.
    let mut current_evals =
        gpu_bitrev_buffer_to_buffer(&evals_buffer, evals_len, gpu_state.inner())
            .map_err(|e| ProvingError::FieldOperationError(format!("GPU bitrev: {e}")))?;
    let mut current_len = evals_len;

    // Compute initial h_inv and omega_inv from domain parameters.
    // The LDE domain is {coset_offset * omega^i} where omega is the LDE primitive root.
    let lde_log_order = domain_size.trailing_zeros() as u64;
    let omega = F::get_primitive_root_of_unity(lde_log_order).map_err(|_| {
        ProvingError::FieldOperationError("Failed to get LDE primitive root".to_string())
    })?;
    let omega_inv = omega.inv().unwrap();
    let h_inv = coset_offset.inv().unwrap();

    // Layer 0 uses from-scratch domain inverse computation + fold.
    // Layers 1+ use stride-2 squaring of the previous layer's inv_x.
    let mut prev_inv_x: Option<metal::Buffer> = None;

    for layer_idx in 1..number_layers {
        let zeta = transcript.sample_field_element();
        coset_offset = coset_offset.square();
        domain_size /= 2;

        let (folded_buf, folded_len, inv_x, merkle_tree) = if layer_idx == 1 {
            // Layer 0: compute domain inverses from scratch (exp-by-squaring).
            // Cannot fuse with commit since it uses a different fold function.
            let (fb, fl, ix) = gpu_fold_evaluations_with_domain_inv(
                &current_evals,
                current_len,
                &zeta,
                &h_inv,
                &omega_inv,
                fold_eval_state,
                domain_inv_state,
            )
            .map_err(|e| {
                ProvingError::FieldOperationError(format!("GPU fused domain-inv+fold: {e}"))
            })?;
            let (mt, _root) =
                gpu_fri_layer_commit_from_buffer(&fb, fl, keccak_state).map_err(|e| {
                    ProvingError::MerkleTreeError(format!("GPU Merkle in eval-domain FRI: {e}"))
                })?;
            (fb, fl, ix, mt)
        } else {
            // Layers 1+: fused stride-2 square + fold + Merkle commit in single command buffer.
            let (fb, fl, ix, mt, _root) = gpu_fold_and_commit_fused(
                &current_evals,
                current_len,
                &zeta,
                prev_inv_x.as_ref().unwrap(),
                fold_eval_state,
                square_inv_state,
                keccak_state,
            )
            .map_err(|e| {
                ProvingError::FieldOperationError(format!("GPU fused fold+commit: {e}"))
            })?;
            (fb, fl, ix, mt)
        };

        // Read back evaluations for FriLayer (needed for CPU query phase).
        let eval_u64: Vec<u64> = MetalState::retrieve_contents(&folded_buf);
        let evaluation: Vec<FpE> = eval_u64.into_iter().map(FpE::from).collect();

        let current_layer = FriLayer::new(&evaluation, merkle_tree, coset_offset, domain_size);

        let commitment = current_layer.merkle_tree.root;
        fri_layer_list.push(current_layer);
        transcript.append_bytes(&commitment);

        // Update for next layer.
        current_evals = folded_buf;
        current_len = folded_len;
        prev_inv_x = Some(inv_x);
    }

    // Final fold to get the last value.
    let zeta = transcript.sample_field_element();

    let (last_buf, _last_len, _last_inv_x) = if let Some(ref inv_x) = prev_inv_x {
        gpu_fold_with_squared_inv(
            &current_evals,
            current_len,
            &zeta,
            inv_x,
            fold_eval_state,
            square_inv_state,
        )
        .map_err(|e| {
            ProvingError::FieldOperationError(format!("GPU fused stride2-square+fold (final): {e}"))
        })?
    } else {
        // Edge case: only 1 layer total, no previous inv_x.
        gpu_fold_evaluations_with_domain_inv(
            &current_evals,
            current_len,
            &zeta,
            &h_inv,
            &omega_inv,
            fold_eval_state,
            domain_inv_state,
        )
        .map_err(|e| {
            ProvingError::FieldOperationError(format!("GPU fused domain-inv+fold (final): {e}"))
        })?
    };

    let last_u64: Vec<u64> = MetalState::retrieve_contents(&last_buf);
    let last_value = if last_u64.is_empty() {
        FpE::zero()
    } else {
        FpE::from(last_u64[0])
    };
    transcript.append_field_element(&last_value);

    Ok((last_value, fri_layer_list))
}

/// GPU-optimized Phase 4 for Goldilocks field with GPU DEEP composition.
///
/// This is a concrete version of [`gpu_round_4`] that uses the Metal GPU shader
/// for DEEP composition polynomial computation instead of CPU Ruffini division.
/// Also uses GPU FFT + GPU Keccak256 for FRI layer construction.
///
/// If `precompiled_deep` is `Some`, uses the pre-compiled shader state.
#[cfg(all(target_os = "macos", feature = "metal"))]
#[allow(clippy::too_many_arguments)]
pub fn gpu_round_4_goldilocks<A>(
    air: &A,
    domain: &Domain<Goldilocks64Field>,
    round_1_result: &GpuRound1Result<Goldilocks64Field>,
    round_2_result: &GpuRound2Result<Goldilocks64Field>,
    round_3_result: &GpuRound3Result<Goldilocks64Field>,
    transcript: &mut impl IsStarkTranscript<Goldilocks64Field, Goldilocks64Field>,
    gpu_state: &StarkMetalState,
    precompiled_deep: Option<&DeepCompositionState>,
    keccak_state: &GpuMerkleState,
    fold_eval_state: &FriFoldEvalState,
    fri_domain_inv_state: &FriDomainInvState,
    fri_square_inv_state: &FriSquareInvState,
    domain_inv_state: Option<&DomainInversionState>,
) -> Result<GpuRound4Result<Goldilocks64Field>, ProvingError>
where
    A: AIR<Field = Goldilocks64Field, FieldExtension = Goldilocks64Field>,
{
    // Step 1: Sample gamma and compute deep composition coefficients.
    let gamma = transcript.sample_field_element();

    let n_terms_composition_poly = round_2_result.composition_poly_parts.len();
    let num_terms_trace =
        air.context().transition_offsets.len() * air.step_size() * air.context().trace_columns;

    let mut deep_composition_coefficients: Vec<_> =
        core::iter::successors(Some(FieldElement::one()), |x| Some(x * gamma))
            .take(n_terms_composition_poly + num_terms_trace)
            .collect();

    let trace_term_coeffs: Vec<_> = deep_composition_coefficients
        .drain(..num_terms_trace)
        .collect::<Vec<_>>()
        .chunks(air.context().transition_offsets.len() * air.step_size())
        .map(|chunk| chunk.to_vec())
        .collect();

    let composition_gammas = deep_composition_coefficients;

    // Step 2: Compute DEEP composition EVALUATIONS on GPU (skip IFFT).
    let t4_deep = std::time::Instant::now();
    let (deep_evals_buffer, deep_evals_len) =
        crate::metal::deep_composition::gpu_compute_deep_composition_evals_to_buffer(
            round_1_result,
            round_2_result,
            round_3_result,
            domain,
            &composition_gammas,
            &trace_term_coeffs,
            precompiled_deep,
            domain_inv_state,
        )?;
    eprintln!("    4a DEEP comp:      {:>10.2?}", t4_deep.elapsed());

    // Step 3: Run eval-domain FRI commit (no IFFT, no per-layer FFT).
    let t4_fri = std::time::Instant::now();
    let domain_size = domain.lde_roots_of_unity_coset.len();
    let (fri_last_value, fri_layers) = gpu_fri_commit_phase_eval_domain(
        domain.root_order as usize,
        deep_evals_buffer,
        deep_evals_len,
        transcript,
        &domain.coset_offset,
        domain_size,
        gpu_state,
        keccak_state,
        fold_eval_state,
        fri_domain_inv_state,
        fri_square_inv_state,
    )?;
    eprintln!("    4b FRI commit:     {:>10.2?}", t4_fri.elapsed());

    // Step 4: Grinding (GPU if available, CPU fallback for Poseidon backend).
    let t4_grind = std::time::Instant::now();
    let security_bits = air.context().proof_options.grinding_factor;
    let mut nonce = None;
    if security_bits > 0 {
        let nonce_value = gpu_generate_nonce(&transcript.state(), security_bits, keccak_state)
            .or_else(|| grinding::generate_nonce(&transcript.state(), security_bits))
            .ok_or(ProvingError::NonceNotFound(security_bits))?;
        transcript.append_bytes(&nonce_value.to_be_bytes());
        nonce = Some(nonce_value);
    }
    eprintln!("    4c grind:          {:>10.2?}", t4_grind.elapsed());

    // Step 5: Sample query indexes.
    let number_of_queries = air.options().fri_number_of_queries;
    let domain_size_u64 = domain_size as u64;
    let iotas: Vec<usize> = (0..number_of_queries)
        .map(|_| transcript.sample_u64(domain_size_u64 >> 1) as usize)
        .collect();

    // Step 6: Run FRI query phase.
    let t4_query = std::time::Instant::now();
    let query_list = fri::query_phase(&fri_layers, &iotas)?;
    eprintln!("    4d query:          {:>10.2?}", t4_query.elapsed());

    // Step 7: Extract FRI Merkle roots from layers.
    let fri_layers_merkle_roots: Vec<_> = fri_layers
        .iter()
        .map(|layer| layer.merkle_tree.root)
        .collect();

    // Step 8: Open deep composition poly (Merkle proofs for trace + composition poly).
    // Use buffer-based opening when GPU buffers are available (UMA zero-copy).
    let has_gpu_bufs = round_1_result.main_lde_gpu_buffers.is_some()
        && round_2_result.lde_composition_gpu_buffers.is_some();
    let deep_poly_openings = if has_gpu_bufs {
        open_deep_composition_poly_from_buffers(domain, round_1_result, round_2_result, &iotas)?
    } else {
        open_deep_composition_poly(domain, round_1_result, round_2_result, &iotas)?
    };

    Ok(GpuRound4Result {
        fri_last_value,
        fri_layers_merkle_roots,
        deep_poly_openings,
        query_list,
        nonce,
    })
}

/// GPU-accelerated Phase 4 for Fp3 extension field proofs.
///
/// Uses CPU DEEP composition (the Fp3 DEEP shader expects base-field trace data,
/// but aux trace in F!=E is in Fp3) and GPU FRI fold.
#[cfg(all(target_os = "macos", feature = "metal"))]
#[allow(clippy::too_many_arguments)]
pub fn gpu_round_4_fp3<A>(
    air: &A,
    domain: &Domain<Goldilocks64Field>,
    round_1_result: &crate::metal::phases::fp3_types::GpuRound1ResultFp3,
    round_2_result: &crate::metal::phases::fp3_types::GpuRound2ResultFp3,
    round_3_result: &crate::metal::phases::fp3_types::GpuRound3ResultFp3,
    transcript: &mut impl IsStarkTranscript<Fp3, Goldilocks64Field>,
    fri_fold_state: &FriFoldFp3State,
) -> Result<crate::metal::phases::fp3_types::GpuRound4ResultFp3, ProvingError>
where
    A: AIR<Field = Goldilocks64Field, FieldExtension = Fp3>,
{
    type F = Goldilocks64Field;

    // Step 1: Sample gamma and compute deep composition coefficients (all Fp3).
    let gamma: Fp3E = transcript.sample_field_element();

    let n_terms_composition_poly = round_2_result.lde_composition_poly_evaluations.len();
    let num_terms_trace =
        air.context().transition_offsets.len() * air.step_size() * air.context().trace_columns;

    let mut deep_composition_coefficients: Vec<Fp3E> =
        core::iter::successors(Some(Fp3E::one()), |x| Some(x * gamma))
            .take(n_terms_composition_poly + num_terms_trace)
            .collect();

    let trace_term_coeffs: Vec<Vec<Fp3E>> = deep_composition_coefficients
        .drain(..num_terms_trace)
        .collect::<Vec<_>>()
        .chunks(air.context().transition_offsets.len() * air.step_size())
        .map(|chunk| chunk.to_vec())
        .collect();
    let composition_gammas = deep_composition_coefficients;

    // Step 2: CPU DEEP composition polynomial (Fp3).
    // Convert main trace polys (F) to Fp3 by embedding, combine with aux trace polys (Fp3).
    let mut all_trace_polys: Vec<Polynomial<Fp3E>> = round_1_result
        .main_trace_polys
        .iter()
        .map(|poly| {
            let fp3_coeffs: Vec<Fp3E> = poly
                .coefficients()
                .iter()
                .map(|c| c.to_extension::<Fp3>())
                .collect();
            Polynomial::new(&fp3_coeffs)
        })
        .collect();
    all_trace_polys.extend(round_1_result.aux_trace_polys.iter().cloned());

    let deep_composition_poly = compute_deep_composition_poly_fp3(
        &all_trace_polys,
        &round_2_result.composition_poly_parts,
        &round_3_result.trace_ood_evaluations,
        &round_3_result.composition_poly_parts_ood_evaluation,
        &round_3_result.z,
        &domain.trace_primitive_root,
        &composition_gammas,
        &trace_term_coeffs,
    );

    // Step 3: GPU FRI commit phase (Fp3 fold kernel).
    let domain_size = domain.lde_roots_of_unity_coset.len();
    let coset_offset_u64 = air.context().proof_options.coset_offset;
    let coset_offset = FieldElement::<F>::from(coset_offset_u64);

    let (fri_last_value, fri_layers) = gpu_fri_commit_phase_fp3(
        domain.root_order as usize,
        deep_composition_poly,
        transcript,
        &coset_offset,
        domain_size,
        fri_fold_state,
    )?;

    // Step 4: Grinding (CPU).
    let security_bits = air.context().proof_options.grinding_factor;
    let mut nonce = None;
    if security_bits > 0 {
        let nonce_value = grinding::generate_nonce(&transcript.state(), security_bits)
            .ok_or(ProvingError::NonceNotFound(security_bits))?;
        transcript.append_bytes(&nonce_value.to_be_bytes());
        nonce = Some(nonce_value);
    }

    // Step 5: Sample query indexes.
    let number_of_queries = air.options().fri_number_of_queries;
    let domain_size_u64 = domain_size as u64;
    let iotas: Vec<usize> = (0..number_of_queries)
        .map(|_| transcript.sample_u64(domain_size_u64 >> 1) as usize)
        .collect();

    // Step 6: FRI query phase (CPU).
    let query_list = fri::query_phase(&fri_layers, &iotas)?;

    // Step 7: Extract FRI Merkle roots.
    let fri_layers_merkle_roots: Vec<Commitment> = fri_layers
        .iter()
        .map(|layer| layer.merkle_tree.root)
        .collect();

    // Step 8: Open deep composition poly (Merkle proofs for trace + composition poly).
    let deep_poly_openings =
        open_deep_composition_poly_fp3(domain, round_1_result, round_2_result, &iotas)?;

    Ok(crate::metal::phases::fp3_types::GpuRound4ResultFp3 {
        fri_last_value,
        fri_layers_merkle_roots,
        deep_poly_openings,
        query_list,
        nonce,
    })
}

/// CPU DEEP composition polynomial for Fp3.
///
/// Mirrors `compute_deep_composition_poly` from the CPU prover but with
/// all operations in Fp3 extension field.
#[allow(clippy::too_many_arguments)]
fn compute_deep_composition_poly_fp3(
    trace_polys: &[Polynomial<Fp3E>],
    composition_poly_parts: &[Polynomial<Fp3E>],
    trace_ood_evaluations: &Table<Fp3>,
    composition_poly_ood_evaluations: &[Fp3E],
    z: &Fp3E,
    primitive_root: &FieldElement<Goldilocks64Field>,
    composition_gammas: &[Fp3E],
    trace_term_coeffs: &[Vec<Fp3E>],
) -> Polynomial<Fp3E> {
    let z_power = z.pow(composition_poly_parts.len());

    // H terms: sum_i gamma_i * (H_i(X) - H_i(z^N)) / (X - z^N)
    let mut h_terms = Polynomial::zero();
    for (i, part) in composition_poly_parts.iter().enumerate() {
        let h_i_eval = &composition_poly_ood_evaluations[i];
        let h_i_term = composition_gammas[i] * (part - h_i_eval);
        h_terms += h_i_term;
    }
    debug_assert_eq!(h_terms.evaluate(&z_power), Fp3E::zero());
    h_terms.ruffini_division_inplace(&z_power);

    // Trace terms
    let trace_evaluations_columns = trace_ood_evaluations.columns();
    let num_offsets = trace_ood_evaluations.height;
    let z_shifted_values: Vec<Fp3E> = (0..num_offsets)
        .map(|offset| primitive_root.pow(offset) * z)
        .collect();

    let trace_terms =
        trace_polys
            .iter()
            .enumerate()
            .fold(Polynomial::zero(), |accumulator, (i, t_j)| {
                let gammas_i = &trace_term_coeffs[i];
                let trace_evaluations_i = &trace_evaluations_columns[i];

                let trace_int = trace_evaluations_i
                    .iter()
                    .zip(&z_shifted_values)
                    .zip(gammas_i)
                    .fold(
                        Polynomial::zero(),
                        |trace_agg, ((trace_term_poly_evaluation, z_shifted), trace_gamma)| {
                            let mut poly = t_j - trace_term_poly_evaluation;
                            poly.ruffini_division_inplace(z_shifted);
                            trace_agg + poly * trace_gamma
                        },
                    );
                accumulator + trace_int
            });

    h_terms + trace_terms
}

/// Opens trace polynomials at a given query index for Fp3 proofs (base field openings).
fn open_trace_polys_base_field(
    domain_size: usize,
    tree: &BatchedMerkleTree<Goldilocks64Field>,
    lde_evaluations: &[Vec<FieldElement<Goldilocks64Field>>],
    challenge: usize,
) -> Result<PolynomialOpenings<Goldilocks64Field>, ProvingError> {
    open_trace_polys(domain_size, tree, lde_evaluations, challenge)
}

/// Opens trace polynomials at a given query index for Fp3 proofs (extension field openings).
fn open_trace_polys_extension(
    domain_size: usize,
    tree: &BatchedMerkleTree<Fp3>,
    lde_evaluations: &[Vec<Fp3E>],
    challenge: usize,
) -> Result<PolynomialOpenings<Fp3>, ProvingError> {
    let index = challenge * 2;
    let index_sym = challenge * 2 + 1;

    let proof = tree.get_proof_by_pos(index).ok_or_else(|| {
        ProvingError::MerkleTreeError(format!("Failed to get proof at position {}", index))
    })?;
    let proof_sym = tree.get_proof_by_pos(index_sym).ok_or_else(|| {
        ProvingError::MerkleTreeError(format!("Failed to get proof at position {}", index_sym))
    })?;

    let actual_index = reverse_index(index, domain_size as u64);
    let actual_index_sym = reverse_index(index_sym, domain_size as u64);

    let evaluations: Vec<_> = lde_evaluations
        .iter()
        .map(|col| col[actual_index])
        .collect();
    let evaluations_sym: Vec<_> = lde_evaluations
        .iter()
        .map(|col| col[actual_index_sym])
        .collect();

    Ok(PolynomialOpenings {
        proof,
        proof_sym,
        evaluations,
        evaluations_sym,
    })
}

/// Opens the deep composition polynomial at query indexes for Fp3 proofs.
///
/// Main trace openings are in F (base field), composition and aux trace openings are in Fp3.
fn open_deep_composition_poly_fp3(
    domain: &Domain<Goldilocks64Field>,
    round_1_result: &crate::metal::phases::fp3_types::GpuRound1ResultFp3,
    round_2_result: &crate::metal::phases::fp3_types::GpuRound2ResultFp3,
    indexes_to_open: &[usize],
) -> Result<Vec<DeepPolynomialOpening<Goldilocks64Field, Fp3>>, ProvingError> {
    let domain_size = domain.lde_roots_of_unity_coset.len();
    let mut openings = Vec::new();

    for index in indexes_to_open.iter() {
        // Open main trace (base field)
        let main_trace_opening = open_trace_polys_base_field(
            domain_size,
            &round_1_result.main_merkle_tree,
            &round_1_result.main_lde_evaluations,
            *index,
        )?;

        // Open composition polynomial (Fp3)
        let composition_openings = open_composition_poly(
            &round_2_result.composition_poly_merkle_tree,
            &round_2_result.lde_composition_poly_evaluations,
            *index,
        )?;

        // Open auxiliary trace (Fp3)
        let aux_trace_polys = match (
            round_1_result.aux_merkle_tree.as_ref(),
            round_1_result.aux_lde_evaluations.is_empty(),
        ) {
            (Some(aux_tree), false) => Some(open_trace_polys_extension(
                domain_size,
                aux_tree,
                &round_1_result.aux_lde_evaluations,
                *index,
            )?),
            _ => None,
        };

        openings.push(DeepPolynomialOpening {
            composition_poly: composition_openings,
            main_trace_polys: main_trace_opening,
            aux_trace_polys,
        });
    }

    Ok(openings)
}

#[cfg(all(test, target_os = "macos", feature = "metal"))]
mod tests {
    use super::*;
    use crate::metal::phases::composition::gpu_round_2;
    use crate::metal::phases::ood::gpu_round_3;
    use crate::metal::phases::rap::gpu_round_1;
    use crate::metal::state::StarkMetalState;
    use lambdaworks_crypto::fiat_shamir::default_transcript::DefaultTranscript;
    use lambdaworks_crypto::fiat_shamir::is_transcript::IsTranscript;
    use lambdaworks_math::field::fields::u64_goldilocks_field::Goldilocks64Field;
    use stark_platinum_prover::proof::options::ProofOptions;

    use stark_platinum_prover::examples::fibonacci_rap::{
        fibonacci_rap_trace, FibonacciRAP, FibonacciRAPPublicInputs,
    };

    type F = Goldilocks64Field;
    type FpE = FieldElement<F>;

    #[test]
    fn gpu_round_4_fibonacci_rap_goldilocks() {
        let trace_length = 32;
        let pub_inputs = FibonacciRAPPublicInputs {
            steps: trace_length - 1,
            a0: FpE::one(),
            a1: FpE::one(),
        };
        let proof_options = ProofOptions::default_test_options();
        let mut trace = fibonacci_rap_trace::<F>([FpE::one(), FpE::one()], trace_length);
        let air = FibonacciRAP::new(trace.num_rows(), &pub_inputs, &proof_options);
        let domain = Domain::new(&air);
        let state = StarkMetalState::new().unwrap();

        let mut transcript = DefaultTranscript::<F>::new(&[]);
        let round_1 = gpu_round_1(&air, &mut trace, &domain, &mut transcript, &state).unwrap();
        let round_2 = gpu_round_2(&air, &domain, &round_1, &mut transcript, &state).unwrap();
        let round_3 = gpu_round_3(&air, &domain, &round_1, &round_2, &mut transcript).unwrap();
        let round_4 =
            gpu_round_4(&air, &domain, &round_1, &round_2, &round_3, &mut transcript).unwrap();

        // FRI last value should be non-zero (extremely unlikely to be zero)
        assert_ne!(round_4.fri_last_value, FpE::zero());

        // Should have FRI layer merkle roots (number depends on domain size)
        assert!(
            !round_4.fri_layers_merkle_roots.is_empty(),
            "FRI layers merkle roots should not be empty"
        );

        // Query list should have `fri_number_of_queries` entries
        assert_eq!(
            round_4.query_list.len(),
            proof_options.fri_number_of_queries
        );

        // Deep poly openings should have `fri_number_of_queries` entries
        assert_eq!(
            round_4.deep_poly_openings.len(),
            proof_options.fri_number_of_queries
        );

        // Each deep poly opening should have main trace, composition, and aux trace openings
        for opening in &round_4.deep_poly_openings {
            assert!(
                !opening.main_trace_polys.evaluations.is_empty(),
                "main trace evaluations should not be empty"
            );
            assert!(
                !opening.composition_poly.evaluations.is_empty(),
                "composition poly evaluations should not be empty"
            );
            // FibonacciRAP has auxiliary trace
            assert!(
                opening.aux_trace_polys.is_some(),
                "aux trace openings should be present for FibonacciRAP"
            );
        }

        // Nonce should be present since grinding_factor > 0 in test options
        assert!(
            round_4.nonce.is_some(),
            "nonce should be present when grinding_factor > 0"
        );
    }

    /// Verify that the deep composition polynomial computation produces correct results
    /// by checking that h_terms evaluates to zero at z^N (a necessary mathematical property).
    #[test]
    fn deep_composition_poly_h_terms_vanish_at_z_power() {
        let trace_length = 16;
        let pub_inputs = FibonacciRAPPublicInputs {
            steps: trace_length - 1,
            a0: FpE::one(),
            a1: FpE::one(),
        };
        let proof_options = ProofOptions::default_test_options();
        let mut trace = fibonacci_rap_trace::<F>([FpE::one(), FpE::one()], trace_length);
        let air = FibonacciRAP::new(trace.num_rows(), &pub_inputs, &proof_options);
        let domain = Domain::new(&air);
        let state = StarkMetalState::new().unwrap();

        let mut transcript = DefaultTranscript::<F>::new(&[]);
        let round_1 = gpu_round_1(&air, &mut trace, &domain, &mut transcript, &state).unwrap();
        let round_2 = gpu_round_2(&air, &domain, &round_1, &mut transcript, &state).unwrap();
        let round_3 = gpu_round_3(&air, &domain, &round_1, &round_2, &mut transcript).unwrap();

        // Sample gamma the same way as gpu_round_4
        let gamma: FieldElement<F> = transcript.sample_field_element();
        let n_terms_composition_poly = round_2.lde_composition_poly_evaluations.len();

        let composition_gammas: Vec<_> =
            core::iter::successors(Some(FieldElement::one()), |x| Some(x * gamma))
                .take(n_terms_composition_poly)
                .collect();

        // Verify that sum_i gamma_i * (H_i(X) - H_i(z^N)) vanishes at z^N
        let z_power = round_3.z.pow(round_2.composition_poly_parts.len());
        let mut h_terms = Polynomial::zero();
        for (i, part) in round_2.composition_poly_parts.iter().enumerate() {
            let h_i_eval = &round_3.composition_poly_parts_ood_evaluation[i];
            let h_i_term = composition_gammas[i] * (part - h_i_eval);
            h_terms += h_i_term;
        }
        assert_eq!(
            h_terms.evaluate(&z_power),
            FieldElement::zero(),
            "H terms must vanish at z^N before Ruffini division"
        );
    }

    // =========================================================================
    // GPU FRI Fold differential tests
    // =========================================================================

    #[test]
    fn gpu_fold_matches_cpu() {
        use stark_platinum_prover::fri::fri_functions::fold_polynomial;

        let fri_state = FriFoldState::new().unwrap();

        // Create a polynomial with 64 coefficients
        let coeffs: Vec<FpE> = (0..64).map(|i| FpE::from(i as u64 * 31 + 17)).collect();
        let poly = Polynomial::new(&coeffs);
        let beta = FpE::from(42u64);

        // CPU: fold_polynomial returns (even + beta*odd), then multiply by 2
        let cpu_folded = FpE::from(2u64) * fold_polynomial(&poly, &beta);

        // GPU: gpu_fold_polynomial_from_cpu computes 2*(even + beta*odd) in one kernel
        let (gpu_buffer, gpu_len) = gpu_fold_polynomial_from_cpu(&poly, &beta, &fri_state).unwrap();

        assert_eq!(
            gpu_len, 32,
            "Folded polynomial should have half the coefficients"
        );

        // Read back GPU result
        let gpu_u64: Vec<u64> = unsafe { fri_state.state.read_buffer(&gpu_buffer, gpu_len) };
        let gpu_coeffs: Vec<FpE> = gpu_u64.iter().map(|&v| FpE::from(v)).collect();

        let cpu_coeffs = cpu_folded.coefficients();
        assert_eq!(
            cpu_coeffs.len(),
            gpu_coeffs.len(),
            "CPU and GPU folded polynomials should have the same length"
        );
        for (i, (cpu, gpu)) in cpu_coeffs.iter().zip(&gpu_coeffs).enumerate() {
            assert_eq!(
                cpu, gpu,
                "FRI fold mismatch at index {i}: CPU={:?} GPU={:?}",
                cpu, gpu
            );
        }
    }

    #[test]
    fn gpu_fold_matches_cpu_chained() {
        use stark_platinum_prover::fri::fri_functions::fold_polynomial;

        let fri_state = FriFoldState::new().unwrap();

        // Create a polynomial with 128 coefficients
        let coeffs: Vec<FpE> = (0..128).map(|i| FpE::from(i as u64 * 7 + 3)).collect();
        let poly = Polynomial::new(&coeffs);
        let beta1 = FpE::from(13u64);
        let beta2 = FpE::from(99u64);

        // CPU: two rounds of folding
        let cpu_fold1 = FpE::from(2u64) * fold_polynomial(&poly, &beta1);
        let cpu_fold2 = FpE::from(2u64) * fold_polynomial(&cpu_fold1, &beta2);

        // GPU: first fold from CPU data
        let (gpu_buffer1, gpu_len1) =
            gpu_fold_polynomial_from_cpu(&poly, &beta1, &fri_state).unwrap();
        assert_eq!(gpu_len1, 64);

        // GPU: second fold from existing GPU buffer
        let (gpu_buffer2, gpu_len2) =
            gpu_fold_polynomial(&gpu_buffer1, gpu_len1, &beta2, &fri_state).unwrap();
        assert_eq!(gpu_len2, 32);

        // Read back and compare
        let gpu_u64: Vec<u64> = unsafe { fri_state.state.read_buffer(&gpu_buffer2, gpu_len2) };
        let gpu_coeffs: Vec<FpE> = gpu_u64.iter().map(|&v| FpE::from(v)).collect();

        let cpu_coeffs = cpu_fold2.coefficients();
        assert_eq!(cpu_coeffs.len(), gpu_coeffs.len());
        for (i, (cpu, gpu)) in cpu_coeffs.iter().zip(&gpu_coeffs).enumerate() {
            assert_eq!(
                cpu, gpu,
                "Chained FRI fold mismatch at index {i}: CPU={:?} GPU={:?}",
                cpu, gpu
            );
        }
    }

    // =========================================================================
    // GPU Fp3 FRI Fold differential tests
    // =========================================================================

    type Fp3 =
        lambdaworks_math::field::fields::u64_goldilocks_field::Degree3GoldilocksExtensionField;
    type Fp3E = FieldElement<Fp3>;

    /// Helper: create an Fp3 element from 3 u64 values.
    fn fp3(a: u64, b: u64, c: u64) -> Fp3E {
        Fp3E::new([FpE::from(a), FpE::from(b), FpE::from(c)])
    }

    /// Helper: extract 3 u64 components from an Fp3 element.
    fn fp3_to_u64s(e: &Fp3E) -> [u64; 3] {
        let comps = e.value();
        [*comps[0].value(), *comps[1].value(), *comps[2].value()]
    }

    #[test]
    fn gpu_fp3_fold_matches_cpu() {
        use stark_platinum_prover::fri::fri_functions::fold_polynomial;

        let fri_state = FriFoldFp3State::new().unwrap();

        // Create a polynomial with 64 Fp3 coefficients
        let coeffs: Vec<Fp3E> = (0..64)
            .map(|i| fp3(i as u64 * 31 + 17, i as u64 * 7 + 3, i as u64 * 13 + 5))
            .collect();
        let poly = Polynomial::new(&coeffs);
        let beta = fp3(42, 7, 99);

        // CPU: fold_polynomial returns (even + beta*odd), then multiply by 2
        let two = fp3(2, 0, 0);
        let cpu_folded = two * fold_polynomial(&poly, &beta);

        // GPU: gpu_fold_polynomial_fp3_from_cpu
        let (gpu_buffer, gpu_len) =
            gpu_fold_polynomial_fp3_from_cpu(&poly, &beta, &fri_state).unwrap();

        assert_eq!(
            gpu_len, 32,
            "Folded polynomial should have half the coefficients"
        );

        // Read back GPU result (3 u64s per Fp3 element)
        let gpu_u64: Vec<u64> = MetalState::retrieve_contents(&gpu_buffer);
        let gpu_coeffs: Vec<Fp3E> = gpu_u64
            .chunks(3)
            .map(|chunk| {
                Fp3E::new([
                    FieldElement::from_raw(chunk[0]),
                    FieldElement::from_raw(chunk[1]),
                    FieldElement::from_raw(chunk[2]),
                ])
            })
            .collect();

        let cpu_coeffs = cpu_folded.coefficients();
        assert_eq!(
            cpu_coeffs.len(),
            gpu_coeffs.len(),
            "CPU and GPU folded Fp3 polynomials should have the same length"
        );
        for (i, (cpu, gpu)) in cpu_coeffs.iter().zip(&gpu_coeffs).enumerate() {
            assert_eq!(
                cpu,
                gpu,
                "Fp3 FRI fold mismatch at index {i}: CPU={:?} GPU={:?}",
                fp3_to_u64s(cpu),
                fp3_to_u64s(gpu)
            );
        }
    }

    #[test]
    fn gpu_fp3_fold_matches_cpu_chained() {
        use stark_platinum_prover::fri::fri_functions::fold_polynomial;

        let fri_state = FriFoldFp3State::new().unwrap();

        // Create a polynomial with 128 Fp3 coefficients
        let coeffs: Vec<Fp3E> = (0..128)
            .map(|i| fp3(i as u64 * 7 + 3, i as u64 * 11 + 2, i as u64 * 3 + 1))
            .collect();
        let poly = Polynomial::new(&coeffs);
        let beta1 = fp3(13, 0, 5);
        let beta2 = fp3(99, 77, 0);
        let two = fp3(2, 0, 0);

        // CPU: two rounds of folding
        let cpu_fold1 = two * fold_polynomial(&poly, &beta1);
        let cpu_fold2 = two * fold_polynomial(&cpu_fold1, &beta2);

        // GPU: first fold from CPU data
        let (gpu_buffer1, gpu_len1) =
            gpu_fold_polynomial_fp3_from_cpu(&poly, &beta1, &fri_state).unwrap();
        assert_eq!(gpu_len1, 64);

        // GPU: second fold from existing GPU buffer
        let (gpu_buffer2, gpu_len2) =
            gpu_fold_polynomial_fp3(&gpu_buffer1, gpu_len1, &beta2, &fri_state).unwrap();
        assert_eq!(gpu_len2, 32);

        // Read back and compare
        let gpu_u64: Vec<u64> = MetalState::retrieve_contents(&gpu_buffer2);
        let gpu_coeffs: Vec<Fp3E> = gpu_u64
            .chunks(3)
            .map(|chunk| {
                Fp3E::new([
                    FieldElement::from_raw(chunk[0]),
                    FieldElement::from_raw(chunk[1]),
                    FieldElement::from_raw(chunk[2]),
                ])
            })
            .collect();

        let cpu_coeffs = cpu_fold2.coefficients();
        assert_eq!(cpu_coeffs.len(), gpu_coeffs.len());
        for (i, (cpu, gpu)) in cpu_coeffs.iter().zip(&gpu_coeffs).enumerate() {
            assert_eq!(
                cpu,
                gpu,
                "Chained Fp3 FRI fold mismatch at index {i}: CPU={:?} GPU={:?}",
                fp3_to_u64s(cpu),
                fp3_to_u64s(gpu)
            );
        }
    }

    // =========================================================================
    // GPU Evaluation-Domain FRI Fold Tests
    // =========================================================================

    #[test]
    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn test_gpu_fri_domain_inverses_correctness() {
        use lambdaworks_gpu::metal::abstractions::state::MetalState;
        use lambdaworks_math::fft::cpu::bit_reversing::reverse_index;
        use lambdaworks_math::field::element::FieldElement;
        use lambdaworks_math::field::fields::u64_goldilocks_field::Goldilocks64Field;
        use lambdaworks_math::field::traits::IsFFTField;

        type F = Goldilocks64Field;
        type FE = FieldElement<F>;

        let domain_size: usize = 64;
        let half_len = domain_size / 2;
        let coset_offset = FE::from(7u64);

        // Compute h_inv and omega_inv
        let order = domain_size.trailing_zeros() as u64;
        let omega = F::get_primitive_root_of_unity(order).unwrap();
        let h_inv = coset_offset.inv().unwrap();
        let omega_inv = omega.inv().unwrap();

        let state = FriDomainInvState::new().unwrap();
        let inv_x_buffer =
            gpu_compute_fri_domain_inverses(&h_inv, &omega_inv, domain_size, &state).unwrap();

        // Read back results
        let inv_x_u64: Vec<u64> = MetalState::retrieve_contents(&inv_x_buffer);
        assert_eq!(inv_x_u64.len(), half_len);

        // Verify: inv_x[i] should equal 1/(h * omega^{bitrev(i)})
        // where bitrev is over log2(half_len) bits
        for (i, inv_x_val) in inv_x_u64.iter().enumerate().take(half_len) {
            let br_i = reverse_index(i, half_len as u64);
            let x = coset_offset * omega.pow(br_i as u64);
            let expected = x.inv().unwrap();
            let expected_u64 = F::canonical(expected.value());
            assert_eq!(
                *inv_x_val, expected_u64,
                "Domain inverse mismatch at index {} (bitrev={}): GPU={}, expected={}",
                i, br_i, inv_x_val, expected_u64
            );
        }
    }

    #[test]
    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn test_gpu_square_fri_inverses() {
        use lambdaworks_gpu::metal::abstractions::state::MetalState;
        use lambdaworks_math::field::element::FieldElement;
        use lambdaworks_math::field::fields::u64_goldilocks_field::Goldilocks64Field;
        use lambdaworks_math::field::traits::IsFFTField;

        type F = Goldilocks64Field;
        type FE = FieldElement<F>;

        let domain_size: usize = 64;
        let half_len = domain_size / 2;
        let coset_offset = FE::from(7u64);

        let order = domain_size.trailing_zeros() as u64;
        let omega = F::get_primitive_root_of_unity(order).unwrap();
        let h_inv = coset_offset.inv().unwrap();
        let omega_inv = omega.inv().unwrap();

        // Compute layer-0 inverses
        let inv_state = FriDomainInvState::new().unwrap();
        let inv_x_buffer =
            gpu_compute_fri_domain_inverses(&h_inv, &omega_inv, domain_size, &inv_state).unwrap();

        // Square them for the next layer
        let sq_state = FriSquareInvState::new().unwrap();
        let next_half = half_len / 2;
        let inv_x_sq_buffer = gpu_square_fri_inverses(&inv_x_buffer, next_half, &sq_state).unwrap();

        let inv_x_sq_u64: Vec<u64> = MetalState::retrieve_contents(&inv_x_sq_buffer);
        assert_eq!(inv_x_sq_u64.len(), next_half);

        // Verify: inv_x_sq[i] = inv_x[2*i]^2 (stride-2 gather + square)
        let inv_x_u64: Vec<u64> = MetalState::retrieve_contents(&inv_x_buffer);
        for i in 0..next_half {
            let v = FE::from(inv_x_u64[2 * i]);
            let expected = v * v;
            let expected_u64 = F::canonical(expected.value());
            assert_eq!(
                inv_x_sq_u64[i], expected_u64,
                "Stride-2 square inverse mismatch at index {}: GPU={}, expected={}",
                i, inv_x_sq_u64[i], expected_u64
            );
        }

        // Also verify these match the from-scratch domain inverses for the next layer.
        let h_inv_1 = h_inv * h_inv;
        let omega_inv_1 = omega_inv * omega_inv;
        let next_domain_size = domain_size / 2;
        let from_scratch =
            gpu_compute_fri_domain_inverses(&h_inv_1, &omega_inv_1, next_domain_size, &inv_state)
                .unwrap();
        let from_scratch_u64: Vec<u64> = MetalState::retrieve_contents(&from_scratch);
        for i in 0..next_half {
            assert_eq!(
                inv_x_sq_u64[i], from_scratch_u64[i],
                "Stride-2 squared inv_x[{}] != from-scratch inv_x[{}]: {} vs {}",
                i, i, inv_x_sq_u64[i], from_scratch_u64[i]
            );
        }
    }

    #[test]
    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn test_gpu_fold_eval_bitrev_matches_coeff_fold() {
        use lambdaworks_gpu::metal::abstractions::state::MetalState;
        use lambdaworks_math::fft::cpu::bit_reversing::in_place_bit_reverse_permute;
        use lambdaworks_math::field::element::FieldElement;
        use lambdaworks_math::field::fields::u64_goldilocks_field::Goldilocks64Field;
        use lambdaworks_math::field::traits::{IsFFTField, IsPrimeField};
        use lambdaworks_math::polynomial::Polynomial;

        type F = Goldilocks64Field;
        type FE = FieldElement<F>;

        // Create a random-ish polynomial of degree 63
        let degree: usize = 63;
        let coeffs: Vec<FE> = (0..=degree)
            .map(|i| FE::from((i * 17 + 3) as u64))
            .collect();
        let poly = Polynomial::new(&coeffs);

        let domain_size: usize = 128;
        let coset_offset = FE::from(7u64);

        // Generate coset domain in natural order, then bit-reverse
        let order = domain_size.trailing_zeros() as u64;
        let omega = F::get_primitive_root_of_unity(order).unwrap();
        let mut domain_nat = Vec::with_capacity(domain_size);
        let mut w_i = FE::one();
        for _ in 0..domain_size {
            domain_nat.push(coset_offset * w_i);
            w_i *= omega;
        }

        // Evaluate polynomial on natural-order domain, then bit-reverse
        let mut evals: Vec<FE> = domain_nat.iter().map(|x| poly.evaluate(x)).collect();
        in_place_bit_reverse_permute(&mut evals);

        let beta = FE::from(42u64);
        let half_domain_size = domain_size / 2;

        // === Path 1: Coefficient-domain fold → evaluate on squared domain → bit-reverse ===
        let half_degree = degree.div_ceil(2);
        let mut folded_coeffs = Vec::with_capacity(half_degree);
        for k in 0..half_degree {
            let even = &coeffs[2 * k];
            let odd = if 2 * k < degree {
                &coeffs[2 * k + 1]
            } else {
                &FE::zero()
            };
            folded_coeffs.push(FE::from(2u64) * (even + beta * odd));
        }
        let folded_poly = Polynomial::new(&folded_coeffs);

        // Evaluate folded polynomial on squared coset domain in natural order, then bit-reverse
        let folded_offset = coset_offset * coset_offset;
        let omega_half_order = half_domain_size.trailing_zeros() as u64;
        let omega_half = F::get_primitive_root_of_unity(omega_half_order).unwrap();
        let mut folded_domain = Vec::with_capacity(half_domain_size);
        let mut w_j = FE::one();
        for _ in 0..half_domain_size {
            folded_domain.push(folded_offset * w_j);
            w_j *= omega_half;
        }
        let mut expected_evals: Vec<FE> = folded_domain
            .iter()
            .map(|x| folded_poly.evaluate(x))
            .collect();
        in_place_bit_reverse_permute(&mut expected_evals);

        // === Path 2: Eval-domain fold on GPU (bit-reversed) ===
        let fold_state = FriFoldEvalState::new().unwrap();
        let inv_state = FriDomainInvState::new().unwrap();

        // Upload bit-reversed evaluations
        let evals_u64: Vec<u64> = evals.iter().map(|e| F::canonical(e.value())).collect();
        let evals_buffer = fold_state.state.alloc_buffer_with_data(&evals_u64).unwrap();

        // Compute domain inverses using known constants
        let h_inv = coset_offset.inv().unwrap();
        let omega_inv = omega.inv().unwrap();
        let inv_x_buffer =
            gpu_compute_fri_domain_inverses(&h_inv, &omega_inv, domain_size, &inv_state).unwrap();

        // Fold evaluations
        let (result_buffer, result_len) = gpu_fold_evaluations(
            &evals_buffer,
            domain_size,
            &beta,
            &inv_x_buffer,
            &fold_state,
        )
        .unwrap();

        assert_eq!(result_len, half_domain_size);

        // Read back and compare
        let result_u64: Vec<u64> = MetalState::retrieve_contents(&result_buffer);
        assert_eq!(result_u64.len(), half_domain_size);

        for i in 0..half_domain_size {
            let expected_u64 = F::canonical(expected_evals[i].value());
            assert_eq!(
                result_u64[i], expected_u64,
                "Eval-domain fold mismatch at index {}: GPU={}, expected={}",
                i, result_u64[i], expected_u64
            );
        }
    }
}
