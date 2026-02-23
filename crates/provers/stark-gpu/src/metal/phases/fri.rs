//! GPU Phase 4: DEEP Composition Polynomial + FRI + Queries.

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
use crate::metal::fft::{gpu_coset_shift_buffer_to_buffer, CosetShiftState};
#[cfg(all(target_os = "macos", feature = "metal"))]
use crate::metal::merkle::{gpu_fri_layer_commit_from_buffer, gpu_generate_nonce, GpuMerkleState};
#[cfg(all(target_os = "macos", feature = "metal"))]
use crate::metal::state::StarkMetalState;
#[cfg(all(target_os = "macos", feature = "metal"))]
use lambdaworks_math::fft::gpu::metal::ops::{fft_buffer_to_buffer, gen_twiddles_to_buffer};
#[cfg(all(target_os = "macos", feature = "metal"))]
use lambdaworks_math::field::fields::u64_goldilocks_field::{
    Degree3GoldilocksExtensionField, Goldilocks64Field,
};
#[cfg(all(target_os = "macos", feature = "metal"))]
use lambdaworks_math::field::traits::{IsPrimeField, RootsConfig};

#[cfg(all(target_os = "macos", feature = "metal"))]
type Fp3 = Degree3GoldilocksExtensionField;
#[cfg(all(target_os = "macos", feature = "metal"))]
type Fp3E = FieldElement<Fp3>;

#[cfg(all(target_os = "macos", feature = "metal"))]
use lambdaworks_gpu::metal::abstractions::{
    errors::MetalError,
    state::{DynamicMetalState, MetalState},
};

/// Check that a Metal command buffer completed successfully.
#[cfg(all(target_os = "macos", feature = "metal"))]
fn check_command_buffer(cb: &metal::CommandBufferRef, context: &str) -> Result<(), MetalError> {
    if cb.status() == metal::MTLCommandBufferStatus::Error {
        return Err(MetalError::ExecutionError(context.to_string()));
    }
    Ok(())
}

/// Dispatch a single compute kernel synchronously.
///
/// Handles the boilerplate of creating a command buffer, encoder, setting the pipeline,
/// dispatching thread groups, and waiting for completion. The caller provides a closure
/// that sets the required buffers on the encoder.
#[cfg(all(target_os = "macos", feature = "metal"))]
fn dispatch_kernel(
    state: &DynamicMetalState,
    pipeline: &metal::ComputePipelineStateRef,
    max_threads: u64,
    work_items: usize,
    context: &str,
    set_buffers: impl FnOnce(&metal::ComputeCommandEncoderRef),
) -> Result<(), MetalError> {
    use metal::MTLSize;

    let threads_per_group = max_threads.min(256);
    let thread_groups = (work_items as u64).div_ceil(threads_per_group);

    let command_buffer = state.command_queue().new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    set_buffers(encoder);
    encoder.dispatch_thread_groups(
        MTLSize::new(thread_groups, 1, 1),
        MTLSize::new(threads_per_group, 1, 1),
    );
    encoder.end_encoding();

    command_buffer.commit();
    command_buffer.wait_until_completed();
    check_command_buffer(command_buffer, context)
}

/// Macro to define a shader state struct with `new()` and `from_device_and_queue()`.
#[cfg(all(target_os = "macos", feature = "metal"))]
macro_rules! define_shader_state {
    ($(#[$meta:meta])* $vis:vis struct $Name:ident, header: $header:expr, shader: $shader:expr, pipeline: $pipeline:expr, max_threads_field: $field:ident) => {
        $(#[$meta])*
        #[cfg(all(target_os = "macos", feature = "metal"))]
        $vis struct $Name {
            state: DynamicMetalState,
            $field: u64,
        }

        #[cfg(all(target_os = "macos", feature = "metal"))]
        impl $Name {
            pub fn new() -> Result<Self, MetalError> {
                let combined_source = format!("{}\n{}", $header, $shader);
                let mut state = DynamicMetalState::new()?;
                state.load_library(&combined_source)?;
                let max_t = state.prepare_pipeline($pipeline)?;
                Ok(Self { state, $field: max_t })
            }

            pub fn from_device_and_queue(
                device: &metal::Device,
                queue: &metal::CommandQueue,
            ) -> Result<Self, MetalError> {
                let combined_source = format!("{}\n{}", $header, $shader);
                let mut state = DynamicMetalState::from_device_and_queue(device, queue);
                state.load_library(&combined_source)?;
                let max_t = state.prepare_pipeline($pipeline)?;
                Ok(Self { state, $field: max_t })
            }
        }
    };
}

#[cfg(all(target_os = "macos", feature = "metal"))]
const FRI_GOLDILOCKS_FIELD_HEADER: &str =
    include_str!("../../../../../math/src/gpu/metal/shaders/field/fp_u64.h.metal");

#[cfg(all(target_os = "macos", feature = "metal"))]
const FRI_FOLD_SHADER: &str = include_str!("../shaders/fri_fold.metal");

define_shader_state! {
    /// Pre-compiled Metal state for the `goldilocks_fri_fold` kernel.
    pub struct FriFoldState,
    header: FRI_GOLDILOCKS_FIELD_HEADER,
    shader: FRI_FOLD_SHADER,
    pipeline: "goldilocks_fri_fold",
    max_threads_field: fri_fold_max_threads
}

/// FRI fold on GPU: `result[k] = 2 * (coeffs[2k] + beta * coeffs[2k+1])`.
/// Returns `(output_buffer, half_len)`.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn gpu_fold_polynomial(
    coeffs_buffer: &metal::Buffer,
    num_coeffs: usize,
    beta: &FieldElement<Goldilocks64Field>,
    state: &FriFoldState,
) -> Result<(metal::Buffer, usize), MetalError> {
    if num_coeffs <= 1 {
        let val: u64 = if num_coeffs == 1 {
            let vals: Vec<u64> = MetalState::retrieve_contents(coeffs_buffer);
            let fe = FieldElement::<Goldilocks64Field>::from(vals.first().copied().unwrap_or(0));
            Goldilocks64Field::canonical(
                (FieldElement::<Goldilocks64Field>::from(2u64) * fe).value(),
            )
        } else {
            0u64
        };
        let buf = state
            .state
            .alloc_buffer_with_data(std::slice::from_ref(&val))?;
        return Ok((buf, 1));
    }

    // Pad to even if odd number of coefficients.
    let (input_buf_owned, input_ref, padded_len) = if !num_coeffs.is_multiple_of(2) {
        let padded_len = num_coeffs + 1;
        let buf_padded = state
            .state
            .alloc_buffer(padded_len * std::mem::size_of::<u64>())?;
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
        (Some(buf_padded), None, padded_len)
    } else {
        (None, Some(coeffs_buffer), num_coeffs)
    };

    let actual_input: &metal::Buffer = input_buf_owned.as_ref().or(input_ref).unwrap();

    let half_len = padded_len / 2;

    let beta_u64 = Goldilocks64Field::canonical(beta.value());

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

    let pipeline = state
        .state
        .get_pipeline_ref("goldilocks_fri_fold")
        .ok_or_else(|| MetalError::FunctionError("goldilocks_fri_fold".to_string()))?;

    dispatch_kernel(
        &state.state,
        pipeline,
        state.fri_fold_max_threads,
        half_len,
        "GPU FRI fold error",
        |encoder| {
            encoder.set_buffer(0, Some(actual_input), 0);
            encoder.set_buffer(1, Some(&buf_output), 0);
            encoder.set_buffer(2, Some(&buf_beta), 0);
            encoder.set_buffer(3, Some(&buf_half_len), 0);
        },
    )?;

    Ok((buf_output, half_len))
}

/// FRI fold on GPU from CPU polynomial data. Uploads coefficients then dispatches kernel.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn gpu_fold_polynomial_from_cpu(
    poly: &Polynomial<FieldElement<Goldilocks64Field>>,
    beta: &FieldElement<Goldilocks64Field>,
    state: &FriFoldState,
) -> Result<(metal::Buffer, usize), MetalError> {
    let coeffs = poly.coefficients();
    let num_coeffs = coeffs.len();
    let coeffs_u64: Vec<u64> = coeffs
        .iter()
        .map(|fe| Goldilocks64Field::canonical(fe.value()))
        .collect();

    let buf_input = state.state.alloc_buffer_with_data(&coeffs_u64)?;

    gpu_fold_polynomial(&buf_input, num_coeffs, beta, state)
}

#[cfg(all(target_os = "macos", feature = "metal"))]
const FRI_FOLD_EVAL_SHADER: &str = include_str!("../shaders/fri_fold_eval.metal");
#[cfg(all(target_os = "macos", feature = "metal"))]
const FRI_DOMAIN_INV_SHADER: &str = include_str!("../shaders/fri_domain_inv.metal");
#[cfg(all(target_os = "macos", feature = "metal"))]
const FRI_SQUARE_INV_SHADER: &str = include_str!("../shaders/fri_square_inv.metal");

#[cfg(all(target_os = "macos", feature = "metal"))]
#[repr(C)]
#[derive(Copy, Clone)]
struct FriFoldEvalParams {
    half_len: u32,
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[repr(C)]
#[derive(Copy, Clone)]
struct FriDomainInvParams {
    half_len: u32,
    log_half_len: u32,
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[repr(C)]
#[derive(Copy, Clone)]
struct FriSquareInvParams {
    len: u32,
}

define_shader_state! {
    /// Pre-compiled Metal state for the `goldilocks_fri_fold_eval` kernel.
    pub struct FriFoldEvalState,
    header: FRI_GOLDILOCKS_FIELD_HEADER,
    shader: FRI_FOLD_EVAL_SHADER,
    pipeline: "goldilocks_fri_fold_eval",
    max_threads_field: max_threads
}

/// Eval-domain FRI fold on GPU (bit-reversed order). Returns `(output_buffer, half_len)`.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn gpu_fold_evaluations(
    evals_buffer: &metal::Buffer,
    num_evals: usize,
    beta: &FieldElement<Goldilocks64Field>,
    inv_x_buffer: &metal::Buffer,
    state: &FriFoldEvalState,
) -> Result<(metal::Buffer, usize), MetalError> {
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

    dispatch_kernel(
        &state.state,
        pipeline,
        state.max_threads,
        half_len,
        "GPU eval-domain FRI fold error",
        |encoder| {
            encoder.set_buffer(0, Some(evals_buffer), 0);
            encoder.set_buffer(1, Some(&buf_output), 0);
            encoder.set_buffer(2, Some(&buf_beta), 0);
            encoder.set_buffer(3, Some(inv_x_buffer), 0);
            encoder.set_buffer(4, Some(&buf_params), 0);
        },
    )?;

    Ok((buf_output, half_len))
}

/// Fused domain inverse + eval-domain fold in a single GPU command buffer.
/// Returns `(folded_buffer, half_len, inv_x_buffer)`.
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

    let inv_pipeline = domain_inv_state
        .state
        .get_pipeline_ref("compute_fri_domain_inverses")
        .ok_or_else(|| MetalError::FunctionError("compute_fri_domain_inverses".to_string()))?;

    let fold_pipeline = fold_eval_state
        .state
        .get_pipeline_ref("goldilocks_fri_fold_eval")
        .ok_or_else(|| MetalError::FunctionError("goldilocks_fri_fold_eval".to_string()))?;

    let command_buffer = domain_inv_state.state.command_queue().new_command_buffer();

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
    check_command_buffer(command_buffer, "GPU fused domain-inv+fold error")?;

    Ok((buf_output, half_len, buf_inv_x))
}

/// Fused stride-2-square-inverse + eval-domain fold in a single GPU command buffer.
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

    let buf_inv_x = square_inv_state
        .state
        .alloc_buffer(half_len * std::mem::size_of::<u64>())?;
    let sq_params = FriSquareInvParams {
        len: half_len as u32,
    };
    let buf_sq_params = square_inv_state
        .state
        .alloc_buffer_with_data(std::slice::from_ref(&sq_params))?;

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

    let sq_pipeline = square_inv_state
        .state
        .get_pipeline_ref("fri_square_inverses")
        .ok_or_else(|| MetalError::FunctionError("fri_square_inverses".to_string()))?;

    let fold_pipeline = fold_eval_state
        .state
        .get_pipeline_ref("goldilocks_fri_fold_eval")
        .ok_or_else(|| MetalError::FunctionError("goldilocks_fri_fold_eval".to_string()))?;

    let command_buffer = square_inv_state.state.command_queue().new_command_buffer();

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
    check_command_buffer(command_buffer, "GPU fused stride-2-square-inv+fold error")?;

    Ok((buf_output, half_len, buf_inv_x))
}

/// Fused stride-2-square-inverse + eval-domain fold + Merkle commit.
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

    let buf_inv_x = square_inv_state
        .state
        .alloc_buffer(half_len * std::mem::size_of::<u64>())?;
    let sq_params = FriSquareInvParams {
        len: half_len as u32,
    };
    let buf_sq_params = square_inv_state
        .state
        .alloc_buffer_with_data(std::slice::from_ref(&sq_params))?;

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

    let num_leaves = half_len / 2;
    let num_cols = 2usize;
    let leaves_len = num_leaves.next_power_of_two();
    let total_nodes = 2 * leaves_len - 1;
    let tree_buf = keccak_state.state.device().new_buffer(
        (total_nodes * 32) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let sq_pipeline = square_inv_state
        .state
        .get_pipeline_ref("fri_square_inverses")
        .ok_or_else(|| MetalError::FunctionError("fri_square_inverses".to_string()))?;
    let fold_pipeline = fold_eval_state
        .state
        .get_pipeline_ref("goldilocks_fri_fold_eval")
        .ok_or_else(|| MetalError::FunctionError("goldilocks_fri_fold_eval".to_string()))?;

    let command_buffer = square_inv_state.state.command_queue().new_command_buffer();

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

    crate::metal::merkle::encode_hash_and_build_tree(
        command_buffer,
        &buf_output,
        num_leaves,
        num_cols,
        &tree_buf,
        leaves_len,
        keccak_state,
    )?;

    command_buffer.commit();
    command_buffer.wait_until_completed();
    if command_buffer.status() == MTLCommandBufferStatus::Error {
        return Err(MetalError::ExecutionError(
            "GPU fused fold+commit error".to_string(),
        ));
    }

    let nodes =
        crate::metal::merkle::read_tree_nodes(&tree_buf, total_nodes, num_leaves, leaves_len);
    let root = nodes[0];
    let tree = BatchedMerkleTree::<Goldilocks64Field>::from_nodes(nodes)
        .ok_or_else(|| MetalError::ExecutionError("Failed to build FRI Merkle tree".into()))?;

    Ok((buf_output, half_len, buf_inv_x, tree, root))
}

define_shader_state! {
    /// Pre-compiled Metal state for the `compute_fri_domain_inverses` kernel.
    pub struct FriDomainInvState,
    header: FRI_GOLDILOCKS_FIELD_HEADER,
    shader: FRI_DOMAIN_INV_SHADER,
    pipeline: "compute_fri_domain_inverses",
    max_threads_field: max_threads
}

/// Compute FRI domain inverses on GPU: `inv_x[i] = h^{-1} * omega^{-bitrev(i)}`.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn gpu_compute_fri_domain_inverses(
    h_inv: &FieldElement<Goldilocks64Field>,
    omega_inv: &FieldElement<Goldilocks64Field>,
    domain_size: usize,
    state: &FriDomainInvState,
) -> Result<metal::Buffer, MetalError> {
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

    dispatch_kernel(
        &state.state,
        pipeline,
        state.max_threads,
        half_len,
        "GPU FRI domain inverse error",
        |encoder| {
            encoder.set_buffer(0, Some(&buf_h_inv), 0);
            encoder.set_buffer(1, Some(&buf_omega_inv), 0);
            encoder.set_buffer(2, Some(&buf_output), 0);
            encoder.set_buffer(3, Some(&buf_params), 0);
        },
    )?;

    Ok(buf_output)
}

define_shader_state! {
    /// Pre-compiled Metal state for the `fri_square_inverses` kernel.
    pub struct FriSquareInvState,
    header: FRI_GOLDILOCKS_FIELD_HEADER,
    shader: FRI_SQUARE_INV_SHADER,
    pipeline: "fri_square_inverses",
    max_threads_field: max_threads
}

/// Stride-2 square FRI domain inverses: `inv_x_next[j] = inv_x_prev[2*j]^2`.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn gpu_square_fri_inverses(
    inv_x_buffer: &metal::Buffer,
    output_len: usize,
    state: &FriSquareInvState,
) -> Result<metal::Buffer, MetalError> {
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

    dispatch_kernel(
        &state.state,
        pipeline,
        state.max_threads,
        output_len,
        "GPU FRI square inverses error",
        |encoder| {
            encoder.set_buffer(0, Some(inv_x_buffer), 0);
            encoder.set_buffer(1, Some(&buf_output), 0);
            encoder.set_buffer(2, Some(&buf_params), 0);
        },
    )?;

    Ok(buf_output)
}

/// Bit-reverse permutation on a Metal buffer (u64 values).
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
    check_command_buffer(command_buffer, "GPU bitrev permutation error")?;

    Ok(output_buffer)
}

#[cfg(all(target_os = "macos", feature = "metal"))]
const FRI_FOLD_FP3_SHADER: &str = include_str!("../shaders/fri_fold_fp3.metal");

/// Pre-compiled Metal state for the `goldilocks_fp3_fri_fold` kernel.
///
/// Note: Uses `combined_fp3_source` instead of the standard header, so cannot use
/// `define_shader_state!`.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub struct FriFoldFp3State {
    state: DynamicMetalState,
    fri_fold_max_threads: u64,
}

#[cfg(all(target_os = "macos", feature = "metal"))]
impl FriFoldFp3State {
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

/// Fp3 FRI fold on GPU: `result[k] = 2 * (coeffs[2k] + beta * coeffs[2k+1])` in Fp3.
/// Returns `(output_buffer, half_len)`.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn gpu_fold_polynomial_fp3(
    coeffs_buffer: &metal::Buffer,
    num_coeffs: usize,
    beta: &Fp3E,
    state: &FriFoldFp3State,
) -> Result<(metal::Buffer, usize), MetalError> {
    if num_coeffs <= 1 {
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

    let actual_input: &metal::Buffer = input_buf_owned.as_ref().or(input_ref).unwrap();

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

    dispatch_kernel(
        &state.state,
        pipeline,
        state.fri_fold_max_threads,
        half_len,
        "GPU Fp3 FRI fold error",
        |encoder| {
            encoder.set_buffer(0, Some(actual_input), 0);
            encoder.set_buffer(1, Some(&buf_output), 0);
            encoder.set_buffer(2, Some(&buf_beta), 0);
            encoder.set_buffer(3, Some(&buf_half_len), 0);
        },
    )?;

    Ok((buf_output, half_len))
}

/// Fp3 FRI fold on GPU from CPU polynomial data.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn gpu_fold_polynomial_fp3_from_cpu(
    poly: &Polynomial<Fp3E>,
    beta: &Fp3E,
    state: &FriFoldFp3State,
) -> Result<(metal::Buffer, usize), MetalError> {
    let coeffs = poly.coefficients();
    let num_coeffs = coeffs.len();
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

pub type DeepPolynomialOpenings<F, E> = Vec<DeepPolynomialOpening<F, E>>;

/// Result of GPU Phase 4 (FRI round).
pub struct GpuRound4Result<F: IsField>
where
    FieldElement<F>: AsBytes,
{
    pub fri_last_value: FieldElement<F>,
    pub fri_layers_merkle_roots: Vec<Commitment>,
    pub deep_poly_openings: DeepPolynomialOpenings<F, F>,
    pub query_list: Vec<FriDecommitment<F>>,
    pub nonce: Option<u64>,
}

/// GPU Phase 4: DEEP composition + FRI commit/query + grinding + Merkle openings.
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
    let gamma = transcript.sample_field_element();

    let n_terms_composition_poly = round_2_result.lde_composition_poly_evaluations.len();
    let num_terms_trace =
        air.context().transition_offsets.len() * air.step_size() * air.context().trace_columns;

    let mut deep_composition_coefficients: Vec<_> =
        core::iter::successors(Some(FieldElement::one()), |x| Some(x * &gamma))
            .take(n_terms_composition_poly + num_terms_trace)
            .collect();

    let trace_term_coeffs: Vec<_> = deep_composition_coefficients
        .drain(..num_terms_trace)
        .collect::<Vec<_>>()
        .chunks(air.context().transition_offsets.len() * air.step_size())
        .map(|chunk| chunk.to_vec())
        .collect();

    let composition_gammas = deep_composition_coefficients;

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

    let domain_size = domain.lde_roots_of_unity_coset.len();
    let (fri_last_value, fri_layers) = fri::commit_phase::<F, F>(
        domain.root_order as usize,
        deep_composition_poly,
        transcript,
        &domain.coset_offset,
        domain_size,
    )?;

    let security_bits = air.context().proof_options.grinding_factor;
    let mut nonce = None;
    if security_bits > 0 {
        let nonce_value = grinding::generate_nonce(&transcript.state(), security_bits)
            .ok_or(ProvingError::NonceNotFound(security_bits))?;
        transcript.append_bytes(&nonce_value.to_be_bytes());
        nonce = Some(nonce_value);
    }

    let number_of_queries = air.options().fri_number_of_queries;
    let domain_size_u64 = domain_size as u64;
    let iotas: Vec<usize> = (0..number_of_queries)
        .map(|_| transcript.sample_u64(domain_size_u64 >> 1) as usize)
        .collect();

    let query_list = fri::query_phase(&fri_layers, &iotas)?;

    let fri_layers_merkle_roots: Vec<_> = fri_layers
        .iter()
        .map(|layer| layer.merkle_tree.root)
        .collect();

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

/// Computes the DEEP composition polynomial via Ruffini division.
///
/// Supports both `F = E` (base field only) and `F != E` (e.g. Goldilocks + Fp3 extension).
/// `primitive_root` lives in the base field `F`; all other field elements live in `E`.
#[allow(clippy::too_many_arguments)]
fn compute_deep_composition_poly<F, E>(
    trace_polys: &[Polynomial<FieldElement<E>>],
    composition_poly_parts: &[Polynomial<FieldElement<E>>],
    trace_ood_evaluations: &Table<E>,
    composition_poly_ood_evaluations: &[FieldElement<E>],
    z: &FieldElement<E>,
    primitive_root: &FieldElement<F>,
    composition_gammas: &[FieldElement<E>],
    trace_term_coeffs: &[Vec<FieldElement<E>>],
) -> Polynomial<FieldElement<E>>
where
    F: IsFFTField + IsSubFieldOf<E>,
    E: IsField,
{
    let z_power = crate::metal::exp_power_of_2(z, composition_poly_parts.len().trailing_zeros());

    let mut h_terms = Polynomial::zero();
    for (i, part) in composition_poly_parts.iter().enumerate() {
        let h_i_eval = &composition_poly_ood_evaluations[i];
        let h_i_term = composition_gammas[i].clone() * (part - h_i_eval);
        h_terms += h_i_term;
    }
    debug_assert_eq!(h_terms.evaluate(&z_power), FieldElement::zero());
    h_terms.ruffini_division_inplace(&z_power);

    let trace_evaluations_columns = trace_ood_evaluations.columns();
    let num_offsets = trace_ood_evaluations.height;
    let z_shifted_values: Vec<FieldElement<E>> = {
        let mut vals = Vec::with_capacity(num_offsets);
        let mut g_pow = FieldElement::<F>::one();
        for _ in 0..num_offsets {
            vals.push(g_pow.clone() * z);
            g_pow *= primitive_root;
        }
        vals
    };

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

/// Opens trace polynomials at a given query index (symmetric pair + Merkle proofs).
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

    let (evaluations, evaluations_sym): (Vec<_>, Vec<_>) = lde_composition_poly_evaluations
        .iter()
        .map(|part| {
            (
                part[reverse_index(index * 2, part.len() as u64)].clone(),
                part[reverse_index(index * 2 + 1, part.len() as u64)].clone(),
            )
        })
        .unzip();

    Ok(PolynomialOpenings {
        proof: proof.clone(),
        proof_sym: proof,
        evaluations,
        evaluations_sym,
    })
}

/// Like [`open_trace_polys`] but reads from GPU Metal buffers via UMA zero-copy.
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

/// Like [`open_composition_poly`] but reads from GPU Metal buffers via UMA zero-copy.
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

    let (evaluations, evaluations_sym): (Vec<_>, Vec<_>) = lde_composition_buffers
        .iter()
        .map(|buf| {
            let part_len = buf.length() as usize / std::mem::size_of::<u64>();
            (
                read_element_from_buffer(buf, reverse_index(index * 2, part_len as u64)),
                read_element_from_buffer(buf, reverse_index(index * 2 + 1, part_len as u64)),
            )
        })
        .unzip();

    Ok(PolynomialOpenings {
        proof: proof.clone(),
        proof_sym: proof,
        evaluations,
        evaluations_sym,
    })
}

/// Opens the deep composition polynomial at a set of query indexes.
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
    let mut openings = Vec::with_capacity(indexes_to_open.len());

    for index in indexes_to_open.iter() {
        let main_trace_opening = open_trace_polys(
            domain_size,
            &round_1_result.main_merkle_tree,
            &round_1_result.main_lde_evaluations,
            *index,
        )?;
        let composition_openings = open_composition_poly(
            &round_2_result.composition_poly_merkle_tree,
            &round_2_result.lde_composition_poly_evaluations,
            *index,
        )?;
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

/// Like [`open_deep_composition_poly`] but reads from GPU Metal buffers via UMA zero-copy.
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

    let mut openings = Vec::with_capacity(indexes_to_open.len());

    for index in indexes_to_open.iter() {
        let main_trace_opening = open_trace_polys_from_buffers(
            domain_size,
            &round_1_result.main_merkle_tree,
            main_bufs,
            *index,
        )?;
        let composition_openings = open_composition_poly_from_buffers(
            &round_2_result.composition_poly_merkle_tree,
            comp_bufs,
            *index,
        )?;
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

#[cfg(all(target_os = "macos", feature = "metal"))]
const GPU_FRI_THRESHOLD: usize = 4096;

/// Create a FRI layer on GPU from a Metal buffer: coset shift -> FFT -> Merkle hash.
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
    let shifted_buffer = gpu_coset_shift_buffer_to_buffer(
        coeffs_buffer,
        num_coeffs,
        coset_offset,
        domain_size,
        coset_state,
    )
    .map_err(|e| ProvingError::FieldOperationError(format!("GPU coset shift in FRI: {e}")))?;

    let order = domain_size.trailing_zeros() as u64;
    let twiddles_buffer = gen_twiddles_to_buffer::<Goldilocks64Field>(
        order,
        RootsConfig::BitReverse,
        gpu_state.inner(),
    )
    .map_err(|e| ProvingError::FieldOperationError(format!("GPU twiddle gen in FRI: {e}")))?;

    let eval_buffer = fft_buffer_to_buffer::<Goldilocks64Field>(
        &shifted_buffer,
        domain_size,
        &twiddles_buffer,
        gpu_state.inner(),
    )
    .map_err(|e| ProvingError::FieldOperationError(format!("GPU FFT in FRI: {e}")))?;

    let (merkle_tree, _root) =
        gpu_fri_layer_commit_from_buffer(&eval_buffer, domain_size, keccak_state)
            .map_err(|e| ProvingError::MerkleTreeError(format!("GPU Merkle in FRI: {e}")))?;

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

/// Initial input for the GPU FRI commit phase: either a CPU polynomial or a GPU buffer.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub enum FriCommitInput {
    /// CPU polynomial coefficients (will be uploaded on first fold).
    Poly(Polynomial<FieldElement<Goldilocks64Field>>),
    /// Pre-existing GPU buffer with coefficient count.
    Buffer(metal::Buffer, usize),
}

/// GPU-accelerated FRI commit phase for Goldilocks (coeff-domain fold).
///
/// Accepts either a CPU polynomial or a GPU buffer as the initial input via [`FriCommitInput`].
#[cfg(all(target_os = "macos", feature = "metal"))]
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub fn gpu_fri_commit_phase_goldilocks(
    number_layers: usize,
    input: FriCommitInput,
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

    // On the first iteration the poly variant uploads from CPU; after that we always use buffers.
    let mut current_buffer: Option<(metal::Buffer, usize)> = None;
    let mut current_poly_cpu: Option<Polynomial<FpE>> = None;
    match input {
        FriCommitInput::Poly(p) => current_poly_cpu = Some(p),
        FriCommitInput::Buffer(buf, len) => current_buffer = Some((buf, len)),
    }

    for _ in 1..number_layers {
        let zeta = transcript.sample_field_element();
        coset_offset = coset_offset.square();
        domain_size /= 2;

        let (folded_buf, folded_len) = if let Some(poly) = current_poly_cpu.take() {
            gpu_fold_polynomial_from_cpu(&poly, &zeta, fri_fold_state)
                .map_err(|e| ProvingError::FieldOperationError(format!("GPU FRI fold: {e}")))?
        } else {
            let (buf, len) = current_buffer.take().unwrap();
            gpu_fold_polynomial(&buf, len, &zeta, fri_fold_state)
                .map_err(|e| ProvingError::FieldOperationError(format!("GPU FRI fold: {e}")))?
        };

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
            let coeffs_u64: Vec<u64> = MetalState::retrieve_contents(&folded_buf);
            let coeffs: Vec<FpE> = coeffs_u64.into_iter().map(FpE::from).collect();
            fri::new_fri_layer(&Polynomial::new(&coeffs), &coset_offset, domain_size)?
        };

        current_buffer = Some((folded_buf, folded_len));
        let commitment = current_layer.merkle_tree.root;
        fri_layer_list.push(current_layer);
        transcript.append_bytes(&commitment);
    }

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

/// GPU-backed FRI layer: stores evaluations as a Metal buffer (UMA zero-copy reads).
#[cfg(all(target_os = "macos", feature = "metal"))]
pub struct GpuFriLayer {
    pub evaluation_buffer: metal::Buffer,
    pub evaluation_len: usize,
    pub merkle_tree: BatchedMerkleTree<Goldilocks64Field>,
    pub coset_offset: FieldElement<Goldilocks64Field>,
    pub domain_size: usize,
}

/// GPU-backed FRI query phase using [`GpuFriLayer`] (reads elements via UMA pointer).
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn gpu_query_phase(
    fri_layers: &[GpuFriLayer],
    iotas: &[usize],
) -> Result<Vec<FriDecommitment<Goldilocks64Field>>, ProvingError> {
    if fri_layers.is_empty() {
        return Ok(vec![]);
    }

    let mut query_list = Vec::with_capacity(iotas.len());

    for iota_s in iotas {
        let mut layers_evaluations_sym = Vec::new();
        let mut layers_auth_paths_sym = Vec::new();

        let mut index = *iota_s;
        for layer in fri_layers {
            let evaluation_sym = crate::metal::phases::rap::read_element_from_buffer(
                &layer.evaluation_buffer,
                index ^ 1,
            );
            let auth_path_sym =
                layer
                    .merkle_tree
                    .get_proof_by_pos(index >> 1)
                    .ok_or_else(|| {
                        ProvingError::MerkleTreeError(format!(
                            "Failed to get proof at position {}",
                            index >> 1
                        ))
                    })?;
            layers_evaluations_sym.push(evaluation_sym);
            layers_auth_paths_sym.push(auth_path_sym);

            index >>= 1;
        }

        query_list.push(FriDecommitment {
            layers_auth_paths: layers_auth_paths_sym,
            layers_evaluations_sym,
        });
    }

    Ok(query_list)
}

/// GPU FRI commit phase using eval-domain fold (no IFFT, no per-layer FFT).
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
) -> Result<(FieldElement<Goldilocks64Field>, Vec<GpuFriLayer>), ProvingError> {
    type F = Goldilocks64Field;
    type FpE = FieldElement<F>;

    let mut domain_size = domain_size;
    let mut fri_layer_list: Vec<GpuFriLayer> = Vec::with_capacity(number_layers);
    let mut coset_offset = *coset_offset;

    let mut current_evals =
        gpu_bitrev_buffer_to_buffer(&evals_buffer, evals_len, gpu_state.inner())
            .map_err(|e| ProvingError::FieldOperationError(format!("GPU bitrev: {e}")))?;
    let mut current_len = evals_len;

    let lde_log_order = domain_size.trailing_zeros() as u64;
    let omega = F::get_primitive_root_of_unity(lde_log_order).map_err(|_| {
        ProvingError::FieldOperationError("Failed to get LDE primitive root".to_string())
    })?;
    let omega_inv = omega.inv().unwrap();
    let h_inv = coset_offset.inv().unwrap();

    let mut prev_inv_x: Option<metal::Buffer> = None;

    for layer_idx in 1..number_layers {
        let zeta = transcript.sample_field_element();
        coset_offset = coset_offset.square();
        domain_size /= 2;

        let (folded_buf, folded_len, inv_x, merkle_tree) = if layer_idx == 1 {
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

        let commitment = merkle_tree.root;
        fri_layer_list.push(GpuFriLayer {
            evaluation_buffer: folded_buf.clone(),
            evaluation_len: folded_len,
            merkle_tree,
            coset_offset,
            domain_size,
        });
        transcript.append_bytes(&commitment);

        current_evals = folded_buf;
        current_len = folded_len;
        prev_inv_x = Some(inv_x);
    }

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

/// GPU-optimized Phase 4 for Goldilocks: GPU DEEP composition + eval-domain FRI.
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
            round_1_result.lde_coset_gpu_buffer.as_ref(),
        )?;

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

    let security_bits = air.context().proof_options.grinding_factor;
    let mut nonce = None;
    if security_bits > 0 {
        let nonce_value = gpu_generate_nonce(&transcript.state(), security_bits, keccak_state)
            .or_else(|| grinding::generate_nonce(&transcript.state(), security_bits))
            .ok_or(ProvingError::NonceNotFound(security_bits))?;
        transcript.append_bytes(&nonce_value.to_be_bytes());
        nonce = Some(nonce_value);
    }

    let number_of_queries = air.options().fri_number_of_queries;
    let domain_size_u64 = domain_size as u64;
    let iotas: Vec<usize> = (0..number_of_queries)
        .map(|_| transcript.sample_u64(domain_size_u64 >> 1) as usize)
        .collect();

    let query_list = gpu_query_phase(&fri_layers, &iotas)?;

    let fri_layers_merkle_roots: Vec<_> = fri_layers
        .iter()
        .map(|layer| layer.merkle_tree.root)
        .collect();

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

/// GPU Phase 4 for Fp3 extension field: CPU DEEP composition + GPU FRI fold.
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

    let security_bits = air.context().proof_options.grinding_factor;
    let mut nonce = None;
    if security_bits > 0 {
        let nonce_value = grinding::generate_nonce(&transcript.state(), security_bits)
            .ok_or(ProvingError::NonceNotFound(security_bits))?;
        transcript.append_bytes(&nonce_value.to_be_bytes());
        nonce = Some(nonce_value);
    }

    let number_of_queries = air.options().fri_number_of_queries;
    let domain_size_u64 = domain_size as u64;
    let iotas: Vec<usize> = (0..number_of_queries)
        .map(|_| transcript.sample_u64(domain_size_u64 >> 1) as usize)
        .collect();

    let query_list = fri::query_phase(&fri_layers, &iotas)?;
    let fri_layers_merkle_roots: Vec<Commitment> = fri_layers
        .iter()
        .map(|layer| layer.merkle_tree.root)
        .collect();

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

/// Opens the deep composition polynomial at query indexes for Fp3 proofs.
fn open_deep_composition_poly_fp3(
    domain: &Domain<Goldilocks64Field>,
    round_1_result: &crate::metal::phases::fp3_types::GpuRound1ResultFp3,
    round_2_result: &crate::metal::phases::fp3_types::GpuRound2ResultFp3,
    indexes_to_open: &[usize],
) -> Result<Vec<DeepPolynomialOpening<Goldilocks64Field, Fp3>>, ProvingError> {
    let domain_size = domain.lde_roots_of_unity_coset.len();
    let mut openings = Vec::with_capacity(indexes_to_open.len());

    for index in indexes_to_open.iter() {
        let main_trace_opening = open_trace_polys(
            domain_size,
            &round_1_result.main_merkle_tree,
            &round_1_result.main_lde_evaluations,
            *index,
        )?;

        let composition_openings = open_composition_poly(
            &round_2_result.composition_poly_merkle_tree,
            &round_2_result.lde_composition_poly_evaluations,
            *index,
        )?;

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

        assert_ne!(round_4.fri_last_value, FpE::zero());
        assert!(!round_4.fri_layers_merkle_roots.is_empty());
        assert_eq!(
            round_4.query_list.len(),
            proof_options.fri_number_of_queries
        );
        assert_eq!(
            round_4.deep_poly_openings.len(),
            proof_options.fri_number_of_queries
        );
        for opening in &round_4.deep_poly_openings {
            assert!(!opening.main_trace_polys.evaluations.is_empty());
            assert!(!opening.composition_poly.evaluations.is_empty());
            assert!(opening.aux_trace_polys.is_some());
        }
        assert!(round_4.nonce.is_some());
    }

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
        let z_power = crate::metal::exp_power_of_2(
            &round_3.z,
            round_2.composition_poly_parts.len().trailing_zeros(),
        );
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

    #[test]
    fn gpu_fold_matches_cpu() {
        use stark_platinum_prover::fri::fri_functions::fold_polynomial;

        let fri_state = FriFoldState::new().unwrap();

        // Create a polynomial with 64 coefficients
        let coeffs: Vec<FpE> = (0..64).map(|i| FpE::from(i as u64 * 31 + 17)).collect();
        let poly = Polynomial::new(&coeffs);
        let beta = FpE::from(42u64);

        let cpu_folded = FpE::from(2u64) * fold_polynomial(&poly, &beta);
        let (gpu_buffer, gpu_len) = gpu_fold_polynomial_from_cpu(&poly, &beta, &fri_state).unwrap();
        assert_eq!(gpu_len, 32);

        let gpu_u64: Vec<u64> = unsafe { fri_state.state.read_buffer(&gpu_buffer, gpu_len) };
        let gpu_coeffs: Vec<FpE> = gpu_u64.iter().map(|&v| FpE::from(v)).collect();

        let cpu_coeffs = cpu_folded.coefficients();
        assert_eq!(cpu_coeffs.len(), gpu_coeffs.len());
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

        let coeffs: Vec<FpE> = (0..128).map(|i| FpE::from(i as u64 * 7 + 3)).collect();
        let poly = Polynomial::new(&coeffs);
        let beta1 = FpE::from(13u64);
        let beta2 = FpE::from(99u64);

        let cpu_fold1 = FpE::from(2u64) * fold_polynomial(&poly, &beta1);
        let cpu_fold2 = FpE::from(2u64) * fold_polynomial(&cpu_fold1, &beta2);

        let (gpu_buffer1, gpu_len1) =
            gpu_fold_polynomial_from_cpu(&poly, &beta1, &fri_state).unwrap();
        assert_eq!(gpu_len1, 64);
        let (gpu_buffer2, gpu_len2) =
            gpu_fold_polynomial(&gpu_buffer1, gpu_len1, &beta2, &fri_state).unwrap();
        assert_eq!(gpu_len2, 32);

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

    type Fp3 =
        lambdaworks_math::field::fields::u64_goldilocks_field::Degree3GoldilocksExtensionField;
    type Fp3E = FieldElement<Fp3>;

    fn fp3(a: u64, b: u64, c: u64) -> Fp3E {
        Fp3E::new([FpE::from(a), FpE::from(b), FpE::from(c)])
    }

    fn fp3_to_u64s(e: &Fp3E) -> [u64; 3] {
        let comps = e.value();
        [*comps[0].value(), *comps[1].value(), *comps[2].value()]
    }

    #[test]
    fn gpu_fp3_fold_matches_cpu() {
        use stark_platinum_prover::fri::fri_functions::fold_polynomial;

        let fri_state = FriFoldFp3State::new().unwrap();

        let coeffs: Vec<Fp3E> = (0..64)
            .map(|i| fp3(i as u64 * 31 + 17, i as u64 * 7 + 3, i as u64 * 13 + 5))
            .collect();
        let poly = Polynomial::new(&coeffs);
        let beta = fp3(42, 7, 99);

        let two = fp3(2, 0, 0);
        let cpu_folded = two * fold_polynomial(&poly, &beta);
        let (gpu_buffer, gpu_len) =
            gpu_fold_polynomial_fp3_from_cpu(&poly, &beta, &fri_state).unwrap();
        assert_eq!(gpu_len, 32);

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
        assert_eq!(cpu_coeffs.len(), gpu_coeffs.len());
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

        let coeffs: Vec<Fp3E> = (0..128)
            .map(|i| fp3(i as u64 * 7 + 3, i as u64 * 11 + 2, i as u64 * 3 + 1))
            .collect();
        let poly = Polynomial::new(&coeffs);
        let beta1 = fp3(13, 0, 5);
        let beta2 = fp3(99, 77, 0);
        let two = fp3(2, 0, 0);

        let cpu_fold1 = two * fold_polynomial(&poly, &beta1);
        let cpu_fold2 = two * fold_polynomial(&cpu_fold1, &beta2);

        let (gpu_buffer1, gpu_len1) =
            gpu_fold_polynomial_fp3_from_cpu(&poly, &beta1, &fri_state).unwrap();
        assert_eq!(gpu_len1, 64);
        let (gpu_buffer2, gpu_len2) =
            gpu_fold_polynomial_fp3(&gpu_buffer1, gpu_len1, &beta2, &fri_state).unwrap();
        assert_eq!(gpu_len2, 32);

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

        let order = domain_size.trailing_zeros() as u64;
        let omega = F::get_primitive_root_of_unity(order).unwrap();
        let h_inv = coset_offset.inv().unwrap();
        let omega_inv = omega.inv().unwrap();

        let state = FriDomainInvState::new().unwrap();
        let inv_x_buffer =
            gpu_compute_fri_domain_inverses(&h_inv, &omega_inv, domain_size, &state).unwrap();

        let inv_x_u64: Vec<u64> = MetalState::retrieve_contents(&inv_x_buffer);
        assert_eq!(inv_x_u64.len(), half_len);

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

        let inv_state = FriDomainInvState::new().unwrap();
        let inv_x_buffer =
            gpu_compute_fri_domain_inverses(&h_inv, &omega_inv, domain_size, &inv_state).unwrap();

        let sq_state = FriSquareInvState::new().unwrap();
        let next_half = half_len / 2;
        let inv_x_sq_buffer = gpu_square_fri_inverses(&inv_x_buffer, next_half, &sq_state).unwrap();

        let inv_x_sq_u64: Vec<u64> = MetalState::retrieve_contents(&inv_x_sq_buffer);
        assert_eq!(inv_x_sq_u64.len(), next_half);

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

        let degree: usize = 63;
        let coeffs: Vec<FE> = (0..=degree)
            .map(|i| FE::from((i * 17 + 3) as u64))
            .collect();
        let poly = Polynomial::new(&coeffs);

        let domain_size: usize = 128;
        let coset_offset = FE::from(7u64);

        let order = domain_size.trailing_zeros() as u64;
        let omega = F::get_primitive_root_of_unity(order).unwrap();
        let mut domain_nat = Vec::with_capacity(domain_size);
        let mut w_i = FE::one();
        for _ in 0..domain_size {
            domain_nat.push(coset_offset * w_i);
            w_i *= omega;
        }

        let mut evals: Vec<FE> = domain_nat.iter().map(|x| poly.evaluate(x)).collect();
        in_place_bit_reverse_permute(&mut evals);

        let beta = FE::from(42u64);
        let half_domain_size = domain_size / 2;

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

        let fold_state = FriFoldEvalState::new().unwrap();
        let inv_state = FriDomainInvState::new().unwrap();

        let evals_u64: Vec<u64> = evals.iter().map(|e| F::canonical(e.value())).collect();
        let evals_buffer = fold_state.state.alloc_buffer_with_data(&evals_u64).unwrap();

        let h_inv = coset_offset.inv().unwrap();
        let omega_inv = omega.inv().unwrap();
        let inv_x_buffer =
            gpu_compute_fri_domain_inverses(&h_inv, &omega_inv, domain_size, &inv_state).unwrap();

        let (result_buffer, result_len) = gpu_fold_evaluations(
            &evals_buffer,
            domain_size,
            &beta,
            &inv_x_buffer,
            &fold_state,
        )
        .unwrap();

        assert_eq!(result_len, half_domain_size);

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
