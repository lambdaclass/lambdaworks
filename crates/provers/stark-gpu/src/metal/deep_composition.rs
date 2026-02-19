//! GPU DEEP composition polynomial for Fibonacci RAP.
//!
//! This module provides a GPU-accelerated alternative to the CPU
//! `compute_deep_composition_poly()`. Instead of Ruffini division on
//! polynomial coefficients, it uses an evaluation-domain approach:
//!
//! 1. Pre-compute batch inversions 1/(x_i - z^N) and 1/(x_i - z*g^k) on CPU
//! 2. Evaluate the DEEP composition at each LDE domain point on GPU (embarrassingly parallel)
//! 3. IFFT back to coefficient form using CPU FFT

#[cfg(all(target_os = "macos", feature = "metal"))]
use lambdaworks_gpu::metal::abstractions::state::DynamicMetalState;
#[cfg(all(target_os = "macos", feature = "metal"))]
use lambdaworks_math::field::{
    element::FieldElement, fields::u64_goldilocks_field::Goldilocks64Field, traits::IsPrimeField,
};
#[cfg(all(target_os = "macos", feature = "metal"))]
use lambdaworks_math::polynomial::Polynomial;

#[cfg(all(target_os = "macos", feature = "metal"))]
use crate::metal::phases::composition::GpuRound2Result;
use crate::metal::phases::ood::GpuRound3Result;
use crate::metal::phases::rap::GpuRound1Result;

/// Embedded Metal shader source for the DEEP composition kernel.
#[cfg(all(target_os = "macos", feature = "metal"))]
const DEEP_COMPOSITION_SHADER: &str = include_str!("shaders/deep_composition.metal");

/// Parameters struct matching the Metal shader's `DeepCompParams`.
///
/// Must be `#[repr(C)]` to match Metal's memory layout exactly.
#[cfg(all(target_os = "macos", feature = "metal"))]
#[repr(C)]
#[derive(Copy, Clone)]
struct DeepCompParams {
    num_rows: u32,
    num_trace_polys: u32,
    num_offsets: u32,
    num_comp_parts: u32,
}

/// Pre-compiled Metal state for DEEP composition polynomial evaluation.
///
/// Create once and reuse across multiple calls to avoid shader recompilation.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub struct DeepCompositionState {
    state: DynamicMetalState,
    max_threads: u64,
}

#[cfg(all(target_os = "macos", feature = "metal"))]
impl DeepCompositionState {
    /// Compile the DEEP composition shader and prepare the pipeline.
    ///
    /// # Errors
    ///
    /// Returns `MetalError` if the shader fails to compile or the pipeline cannot be created.
    pub fn new() -> Result<Self, lambdaworks_gpu::metal::abstractions::errors::MetalError> {
        let mut state = DynamicMetalState::new()?;
        state.load_library(DEEP_COMPOSITION_SHADER)?;
        let max_threads = state.prepare_pipeline("deep_composition_eval")?;
        Ok(Self { state, max_threads })
    }
}

/// Compute the DEEP composition polynomial using GPU for the evaluation-domain computation.
///
/// This is a GPU-accelerated alternative to the CPU `compute_deep_composition_poly()`.
/// Instead of Ruffini division on polynomial coefficients, it:
/// 1. Pre-computes batch inversions 1/(x_i - z^N) and 1/(x_i - z*g^k) on CPU
/// 2. Evaluates the DEEP composition at each LDE domain point on GPU (embarrassingly parallel)
/// 3. IFFTs back to coefficient form using CPU FFT
///
/// If `precompiled` is `Some`, uses the pre-compiled shader state to avoid
/// recompiling the Metal shader on each call.
#[cfg(all(target_os = "macos", feature = "metal"))]
#[allow(clippy::too_many_arguments)]
pub fn gpu_compute_deep_composition_poly(
    round_1_result: &GpuRound1Result<Goldilocks64Field>,
    round_2_result: &GpuRound2Result<Goldilocks64Field>,
    round_3_result: &GpuRound3Result<Goldilocks64Field>,
    domain: &stark_platinum_prover::domain::Domain<Goldilocks64Field>,
    composition_gammas: &[FieldElement<Goldilocks64Field>],
    trace_term_coeffs: &[Vec<FieldElement<Goldilocks64Field>>],
    _gpu_state: &crate::metal::state::StarkMetalState,
    precompiled: Option<&DeepCompositionState>,
) -> Result<Polynomial<FieldElement<Goldilocks64Field>>, stark_platinum_prover::prover::ProvingError>
{
    type F = Goldilocks64Field;
    type FpE = FieldElement<F>;

    let num_rows = domain.lde_roots_of_unity_coset.len();
    let num_offsets = round_3_result.trace_ood_evaluations.height; // 3 for Fibonacci RAP
    let num_comp_parts = round_2_result.composition_poly_parts.len(); // 1 for Fibonacci RAP

    // Combine main + aux trace LDE evaluations (column-major).
    let mut all_trace_lde = round_1_result.main_lde_evaluations.clone();
    all_trace_lde.extend(round_1_result.aux_lde_evaluations.iter().cloned());
    let num_trace_polys = all_trace_lde.len(); // 3 for Fibonacci RAP

    // --- Pre-compute batch inversions on CPU ---
    let z_power = round_3_result.z.pow(num_comp_parts);
    let primitive_root = &domain.trace_primitive_root;

    // 1/(x_i - z^N) for each domain point
    let mut inv_z_power_vec: Vec<FpE> = domain
        .lde_roots_of_unity_coset
        .iter()
        .map(|x| x - z_power)
        .collect();
    FieldElement::inplace_batch_inverse(&mut inv_z_power_vec)
        .map_err(|_| stark_platinum_prover::prover::ProvingError::BatchInversionFailed)?;

    // z*g^k for each offset k
    let z_shifted_values: Vec<FpE> = (0..num_offsets)
        .map(|k| primitive_root.pow(k) * round_3_result.z)
        .collect();

    // 1/(x_i - z*g^k) for each offset k
    let mut inv_z_shifted_vecs: Vec<Vec<FpE>> = z_shifted_values
        .iter()
        .map(|z_shifted| {
            domain
                .lde_roots_of_unity_coset
                .iter()
                .map(|x| x - z_shifted)
                .collect()
        })
        .collect();
    for inv_vec in &mut inv_z_shifted_vecs {
        FieldElement::inplace_batch_inverse(inv_vec)
            .map_err(|_| stark_platinum_prover::prover::ProvingError::BatchInversionFailed)?;
    }

    // --- Pack scalar data (gammas + OOD evals) as raw u64 ---
    // Layout: [gamma_h * num_comp_parts,
    //          gamma_t * (num_trace_polys * num_offsets),
    //          h_ood * num_comp_parts,
    //          t_ood * (num_trace_polys * num_offsets)]
    let trace_ood_columns = round_3_result.trace_ood_evaluations.columns();
    let mut scalars: Vec<u64> = Vec::new();

    // Composition gammas
    for gamma in composition_gammas.iter().take(num_comp_parts) {
        scalars.push(F::canonical(gamma.value()));
    }
    // Trace gammas (for each trace poly, for each offset)
    for gammas in trace_term_coeffs.iter() {
        for g in gammas.iter() {
            scalars.push(F::canonical(g.value()));
        }
    }
    // Composition OOD evaluations
    for eval in round_3_result.composition_poly_parts_ood_evaluation.iter() {
        scalars.push(F::canonical(eval.value()));
    }
    // Trace OOD evaluations (for each trace poly, for each offset)
    for col in trace_ood_columns.iter() {
        for eval in col.iter() {
            scalars.push(F::canonical(eval.value()));
        }
    }

    // --- Convert trace and composition LDE data to raw u64 ---
    let trace_raw: Vec<Vec<u64>> = all_trace_lde
        .iter()
        .map(|col| col.iter().map(|fe| F::canonical(fe.value())).collect())
        .collect();
    let comp_raw: Vec<Vec<u64>> = round_2_result
        .lde_composition_poly_evaluations
        .iter()
        .map(|col| col.iter().map(|fe| F::canonical(fe.value())).collect())
        .collect();
    let inv_zp_raw: Vec<u64> = inv_z_power_vec
        .iter()
        .map(|fe| F::canonical(fe.value()))
        .collect();
    let inv_zs_raw: Vec<Vec<u64>> = inv_z_shifted_vecs
        .iter()
        .map(|v| v.iter().map(|fe| F::canonical(fe.value())).collect())
        .collect();

    let params = DeepCompParams {
        num_rows: num_rows as u32,
        num_trace_polys: num_trace_polys as u32,
        num_offsets: num_offsets as u32,
        num_comp_parts: num_comp_parts as u32,
    };

    // --- Dispatch Metal kernel ---
    let mut owned_state;
    let (dyn_state, max_threads) = match precompiled {
        Some(pre) => (&pre.state, pre.max_threads),
        None => {
            owned_state = DynamicMetalState::new().map_err(|e| {
                stark_platinum_prover::prover::ProvingError::FieldOperationError(e.to_string())
            })?;
            owned_state
                .load_library(DEEP_COMPOSITION_SHADER)
                .map_err(|e| {
                    stark_platinum_prover::prover::ProvingError::FieldOperationError(e.to_string())
                })?;
            let mt = owned_state
                .prepare_pipeline("deep_composition_eval")
                .map_err(|e| {
                    stark_platinum_prover::prover::ProvingError::FieldOperationError(e.to_string())
                })?;
            (&owned_state, mt)
        }
    };

    let buf_trace_0 = dyn_state
        .alloc_buffer_with_data(&trace_raw[0])
        .map_err(|e| {
            stark_platinum_prover::prover::ProvingError::FieldOperationError(e.to_string())
        })?;
    let buf_trace_1 = dyn_state
        .alloc_buffer_with_data(&trace_raw[1])
        .map_err(|e| {
            stark_platinum_prover::prover::ProvingError::FieldOperationError(e.to_string())
        })?;
    let buf_trace_2 = dyn_state
        .alloc_buffer_with_data(&trace_raw[2])
        .map_err(|e| {
            stark_platinum_prover::prover::ProvingError::FieldOperationError(e.to_string())
        })?;
    let buf_comp_0 = dyn_state
        .alloc_buffer_with_data(&comp_raw[0])
        .map_err(|e| {
            stark_platinum_prover::prover::ProvingError::FieldOperationError(e.to_string())
        })?;
    let buf_inv_zp = dyn_state.alloc_buffer_with_data(&inv_zp_raw).map_err(|e| {
        stark_platinum_prover::prover::ProvingError::FieldOperationError(e.to_string())
    })?;
    let buf_inv_zs0 = dyn_state
        .alloc_buffer_with_data(&inv_zs_raw[0])
        .map_err(|e| {
            stark_platinum_prover::prover::ProvingError::FieldOperationError(e.to_string())
        })?;
    let buf_inv_zs1 = dyn_state
        .alloc_buffer_with_data(&inv_zs_raw[1])
        .map_err(|e| {
            stark_platinum_prover::prover::ProvingError::FieldOperationError(e.to_string())
        })?;
    let buf_inv_zs2 = dyn_state
        .alloc_buffer_with_data(&inv_zs_raw[2])
        .map_err(|e| {
            stark_platinum_prover::prover::ProvingError::FieldOperationError(e.to_string())
        })?;
    let buf_scalars = dyn_state.alloc_buffer_with_data(&scalars).map_err(|e| {
        stark_platinum_prover::prover::ProvingError::FieldOperationError(e.to_string())
    })?;
    let buf_params = dyn_state
        .alloc_buffer_with_data(std::slice::from_ref(&params))
        .map_err(|e| {
            stark_platinum_prover::prover::ProvingError::FieldOperationError(e.to_string())
        })?;
    let buf_output = dyn_state
        .alloc_buffer(num_rows * std::mem::size_of::<u64>())
        .map_err(|e| {
            stark_platinum_prover::prover::ProvingError::FieldOperationError(e.to_string())
        })?;
    dyn_state
        .execute_compute(
            "deep_composition_eval",
            &[
                &buf_trace_0,
                &buf_trace_1,
                &buf_trace_2,
                &buf_comp_0,
                &buf_inv_zp,
                &buf_inv_zs0,
                &buf_inv_zs1,
                &buf_inv_zs2,
                &buf_scalars,
                &buf_params,
                &buf_output,
            ],
            num_rows as u64,
            max_threads,
        )
        .map_err(|e| {
            stark_platinum_prover::prover::ProvingError::FieldOperationError(e.to_string())
        })?;

    // --- Read GPU results ---
    let output_raw: Vec<u64> = unsafe { dyn_state.read_buffer(&buf_output, num_rows) };
    let deep_poly_evals: Vec<FpE> = output_raw.into_iter().map(FpE::from).collect();

    // --- IFFT to get polynomial coefficients ---
    // The evaluations are on the LDE coset {offset * w^i}, so we use coset IFFT.
    let deep_poly = Polynomial::interpolate_offset_fft(&deep_poly_evals, &domain.coset_offset)?;

    Ok(deep_poly)
}

/// Compute the DEEP composition polynomial on GPU, returning coefficients as a Metal Buffer.
///
/// Like [`gpu_compute_deep_composition_poly`] but keeps everything on GPU:
/// - The DEEP composition evaluations stay on a GPU buffer (no CPU readback)
/// - The coset IFFT runs on GPU via `gpu_interpolate_offset_fft_buffer_to_buffer`
/// - Returns `(metal::Buffer, usize)` instead of `Polynomial`
///
/// This eliminates the largest CPU↔GPU transfer in the prover: the full LDE-sized
/// evaluation vector that was previously read back to CPU for IFFT.
#[cfg(all(target_os = "macos", feature = "metal"))]
#[allow(clippy::too_many_arguments)]
pub fn gpu_compute_deep_composition_poly_to_buffer(
    round_1_result: &GpuRound1Result<Goldilocks64Field>,
    round_2_result: &GpuRound2Result<Goldilocks64Field>,
    round_3_result: &GpuRound3Result<Goldilocks64Field>,
    domain: &stark_platinum_prover::domain::Domain<Goldilocks64Field>,
    composition_gammas: &[FieldElement<Goldilocks64Field>],
    trace_term_coeffs: &[Vec<FieldElement<Goldilocks64Field>>],
    _gpu_state: &crate::metal::state::StarkMetalState,
    precompiled: Option<&DeepCompositionState>,
    coset_state: &crate::metal::fft::CosetShiftState,
) -> Result<(metal::Buffer, usize), stark_platinum_prover::prover::ProvingError>
{
    type F = Goldilocks64Field;
    type FpE = FieldElement<F>;

    let num_rows = domain.lde_roots_of_unity_coset.len();
    let num_offsets = round_3_result.trace_ood_evaluations.height;
    let num_comp_parts = round_2_result.composition_poly_parts.len();
    let num_trace_polys = round_1_result.main_lde_evaluations.len()
        + round_1_result.aux_lde_evaluations.len();

    // --- Pre-compute batch inversions on CPU ---
    let z_power = round_3_result.z.pow(num_comp_parts);
    let primitive_root = &domain.trace_primitive_root;

    let mut inv_z_power_vec: Vec<FpE> = domain
        .lde_roots_of_unity_coset
        .iter()
        .map(|x| x - &z_power)
        .collect();
    FieldElement::inplace_batch_inverse(&mut inv_z_power_vec)
        .map_err(|_| stark_platinum_prover::prover::ProvingError::BatchInversionFailed)?;

    let z_shifted_values: Vec<FpE> = (0..num_offsets)
        .map(|k| primitive_root.pow(k) * &round_3_result.z)
        .collect();

    let mut inv_z_shifted_vecs: Vec<Vec<FpE>> = z_shifted_values
        .iter()
        .map(|z_shifted| {
            domain
                .lde_roots_of_unity_coset
                .iter()
                .map(|x| x - z_shifted)
                .collect()
        })
        .collect();
    for inv_vec in &mut inv_z_shifted_vecs {
        FieldElement::inplace_batch_inverse(inv_vec)
            .map_err(|_| stark_platinum_prover::prover::ProvingError::BatchInversionFailed)?;
    }

    // --- Pack scalar data ---
    let trace_ood_columns = round_3_result.trace_ood_evaluations.columns();
    let mut scalars: Vec<u64> = Vec::new();
    for gamma in composition_gammas.iter().take(num_comp_parts) {
        scalars.push(F::canonical(gamma.value()));
    }
    for gammas in trace_term_coeffs.iter() {
        for g in gammas.iter() {
            scalars.push(F::canonical(g.value()));
        }
    }
    for eval in round_3_result.composition_poly_parts_ood_evaluation.iter() {
        scalars.push(F::canonical(eval.value()));
    }
    for col in trace_ood_columns.iter() {
        for eval in col.iter() {
            scalars.push(F::canonical(eval.value()));
        }
    }

    let params = DeepCompParams {
        num_rows: num_rows as u32,
        num_trace_polys: num_trace_polys as u32,
        num_offsets: num_offsets as u32,
        num_comp_parts: num_comp_parts as u32,
    };

    // --- Dispatch Metal kernel ---
    let mut owned_state;
    let (dyn_state, max_threads) = match precompiled {
        Some(pre) => (&pre.state, pre.max_threads),
        None => {
            owned_state = DynamicMetalState::new().map_err(|e| {
                stark_platinum_prover::prover::ProvingError::FieldOperationError(e.to_string())
            })?;
            owned_state
                .load_library(DEEP_COMPOSITION_SHADER)
                .map_err(|e| {
                    stark_platinum_prover::prover::ProvingError::FieldOperationError(e.to_string())
                })?;
            let mt = owned_state
                .prepare_pipeline("deep_composition_eval")
                .map_err(|e| {
                    stark_platinum_prover::prover::ProvingError::FieldOperationError(e.to_string())
                })?;
            (&owned_state, mt)
        }
    };

    let alloc_err = |e: lambdaworks_gpu::metal::abstractions::errors::MetalError| {
        stark_platinum_prover::prover::ProvingError::FieldOperationError(e.to_string())
    };

    // --- Allocate trace + composition buffers ---
    // Use retained GPU buffers from Phase 1/Phase 2 when available (zero-copy),
    // otherwise fall back to clone + convert + upload.
    let has_gpu_trace_bufs = round_1_result.main_lde_gpu_buffers.as_ref().is_some_and(|b| b.len() >= 2)
        && round_1_result.aux_lde_gpu_buffers.as_ref().is_some_and(|b| !b.is_empty());
    let has_gpu_comp_bufs = round_2_result.lde_composition_gpu_buffers.as_ref().is_some_and(|b| !b.is_empty());

    // Owned buffers to keep alive for the fallback path (dropped at end of scope).
    let mut _owned_trace_bufs: Vec<metal::Buffer> = Vec::new();
    let mut _owned_comp_bufs: Vec<metal::Buffer> = Vec::new();

    let (trace_buf_0, trace_buf_1, trace_buf_2, comp_buf_0) = if has_gpu_trace_bufs && has_gpu_comp_bufs {
        // Fast path: use retained GPU buffers directly (no clone, no convert, no upload).
        let main_bufs = round_1_result.main_lde_gpu_buffers.as_ref().unwrap();
        let aux_bufs = round_1_result.aux_lde_gpu_buffers.as_ref().unwrap();
        let comp_bufs = round_2_result.lde_composition_gpu_buffers.as_ref().unwrap();
        (&main_bufs[0], &main_bufs[1], &aux_bufs[0], &comp_bufs[0])
    } else {
        // Fallback: clone LDE data, convert to u64, upload to GPU.
        let mut all_trace_lde = round_1_result.main_lde_evaluations.clone();
        all_trace_lde.extend(round_1_result.aux_lde_evaluations.iter().cloned());

        let trace_raw: Vec<Vec<u64>> = all_trace_lde
            .iter()
            .map(|col| col.iter().map(|fe| F::canonical(fe.value())).collect())
            .collect();
        let comp_raw: Vec<Vec<u64>> = round_2_result
            .lde_composition_poly_evaluations
            .iter()
            .map(|col| col.iter().map(|fe| F::canonical(fe.value())).collect())
            .collect();

        _owned_trace_bufs.push(dyn_state.alloc_buffer_with_data(&trace_raw[0]).map_err(alloc_err)?);
        _owned_trace_bufs.push(dyn_state.alloc_buffer_with_data(&trace_raw[1]).map_err(alloc_err)?);
        _owned_trace_bufs.push(dyn_state.alloc_buffer_with_data(&trace_raw[2]).map_err(alloc_err)?);
        _owned_comp_bufs.push(dyn_state.alloc_buffer_with_data(&comp_raw[0]).map_err(alloc_err)?);

        (&_owned_trace_bufs[0], &_owned_trace_bufs[1], &_owned_trace_bufs[2], &_owned_comp_bufs[0])
    };

    // --- Inversions always computed fresh (different per z) ---
    let inv_zp_raw: Vec<u64> = inv_z_power_vec
        .iter()
        .map(|fe| F::canonical(fe.value()))
        .collect();
    let inv_zs_raw: Vec<Vec<u64>> = inv_z_shifted_vecs
        .iter()
        .map(|v| v.iter().map(|fe| F::canonical(fe.value())).collect())
        .collect();

    let buf_inv_zp = dyn_state.alloc_buffer_with_data(&inv_zp_raw).map_err(alloc_err)?;
    let buf_inv_zs0 = dyn_state.alloc_buffer_with_data(&inv_zs_raw[0]).map_err(alloc_err)?;
    let buf_inv_zs1 = dyn_state.alloc_buffer_with_data(&inv_zs_raw[1]).map_err(alloc_err)?;
    let buf_inv_zs2 = dyn_state.alloc_buffer_with_data(&inv_zs_raw[2]).map_err(alloc_err)?;
    let buf_scalars = dyn_state.alloc_buffer_with_data(&scalars).map_err(alloc_err)?;
    let buf_params = dyn_state
        .alloc_buffer_with_data(std::slice::from_ref(&params))
        .map_err(alloc_err)?;
    let buf_output = dyn_state
        .alloc_buffer(num_rows * std::mem::size_of::<u64>())
        .map_err(alloc_err)?;

    dyn_state
        .execute_compute(
            "deep_composition_eval",
            &[
                trace_buf_0,
                trace_buf_1,
                trace_buf_2,
                comp_buf_0,
                &buf_inv_zp,
                &buf_inv_zs0,
                &buf_inv_zs1,
                &buf_inv_zs2,
                &buf_scalars,
                &buf_params,
                &buf_output,
            ],
            num_rows as u64,
            max_threads,
        )
        .map_err(|e| {
            stark_platinum_prover::prover::ProvingError::FieldOperationError(e.to_string())
        })?;

    // --- GPU IFFT: keep evaluations on GPU, IFFT to coefficients on GPU ---
    let coeffs_buffer = crate::metal::fft::gpu_interpolate_offset_fft_buffer_to_buffer(
        &buf_output,
        num_rows,
        &domain.coset_offset,
        coset_state,
        _gpu_state.inner(),
    )
    .map_err(|e| {
        stark_platinum_prover::prover::ProvingError::FieldOperationError(
            format!("GPU IFFT in deep composition: {e}"),
        )
    })?;

    Ok((coeffs_buffer, num_rows))
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
    use lambdaworks_math::field::element::FieldElement;
    use lambdaworks_math::field::fields::u64_goldilocks_field::Goldilocks64Field;
    use lambdaworks_math::polynomial::Polynomial;
    use stark_platinum_prover::domain::Domain;
    use stark_platinum_prover::examples::fibonacci_rap::{
        fibonacci_rap_trace, FibonacciRAP, FibonacciRAPPublicInputs,
    };
    use stark_platinum_prover::proof::options::ProofOptions;
    use stark_platinum_prover::traits::AIR;

    type F = Goldilocks64Field;
    type FpE = FieldElement<F>;

    /// Differential test: compare GPU DEEP composition poly against the CPU version.
    ///
    /// The GPU path uses the evaluation-domain approach (evaluate at each LDE point,
    /// then IFFT), while the CPU reference uses Ruffini division on polynomial coefficients.
    /// Both must produce the same polynomial.
    #[test]
    fn gpu_deep_composition_matches_cpu() {
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
        let gamma: FpE = transcript.sample_field_element();
        let n_terms_composition_poly = round_2.lde_composition_poly_evaluations.len();
        let num_terms_trace =
            air.context().transition_offsets.len() * air.step_size() * air.context().trace_columns;

        let mut deep_composition_coefficients: Vec<_> =
            core::iter::successors(Some(FpE::one()), |x| Some(x * &gamma))
                .take(n_terms_composition_poly + num_terms_trace)
                .collect();

        let trace_term_coeffs: Vec<_> = deep_composition_coefficients
            .drain(..num_terms_trace)
            .collect::<Vec<_>>()
            .chunks(air.context().transition_offsets.len() * air.step_size())
            .map(|chunk| chunk.to_vec())
            .collect();
        let composition_gammas = deep_composition_coefficients;

        // --- CPU reference path (using Ruffini division) ---
        let mut all_trace_polys_cpu = round_1.main_trace_polys.clone();
        all_trace_polys_cpu.extend(round_1.aux_trace_polys.iter().cloned());

        let z_power = round_3.z.pow(round_2.composition_poly_parts.len());
        let primitive_root = &domain.trace_primitive_root;

        let mut h_terms = Polynomial::zero();
        for (i, part) in round_2.composition_poly_parts.iter().enumerate() {
            let h_i_eval = &round_3.composition_poly_parts_ood_evaluation[i];
            let h_i_term = &composition_gammas[i] * (part - h_i_eval);
            h_terms = h_terms + h_i_term;
        }
        h_terms.ruffini_division_inplace(&z_power);

        let trace_evaluations_columns = round_3.trace_ood_evaluations.columns();
        let num_offsets = round_3.trace_ood_evaluations.height;
        let z_shifted_values: Vec<FpE> = (0..num_offsets)
            .map(|offset| primitive_root.pow(offset) * &round_3.z)
            .collect();

        let trace_terms = all_trace_polys_cpu.iter().enumerate().fold(
            Polynomial::zero(),
            |accumulator, (i, t_j)| {
                let gammas_i = &trace_term_coeffs[i];
                let trace_evals_i = &trace_evaluations_columns[i];
                let trace_int = trace_evals_i
                    .iter()
                    .zip(&z_shifted_values)
                    .zip(gammas_i)
                    .fold(
                        Polynomial::zero(),
                        |trace_agg, ((eval, z_shifted), trace_gamma)| {
                            let mut poly = t_j - eval;
                            poly.ruffini_division_inplace(z_shifted);
                            trace_agg + poly * trace_gamma
                        },
                    );
                accumulator + trace_int
            },
        );

        let cpu_deep_poly = h_terms + trace_terms;

        // --- GPU path (evaluation-domain approach) ---
        let gpu_deep_poly = gpu_compute_deep_composition_poly(
            &round_1,
            &round_2,
            &round_3,
            &domain,
            &composition_gammas,
            &trace_term_coeffs,
            &state,
            None,
        )
        .unwrap();

        // Compare polynomial coefficients
        let cpu_coeffs = cpu_deep_poly.coefficients();
        let gpu_coeffs = gpu_deep_poly.coefficients();

        let max_len = cpu_coeffs.len().max(gpu_coeffs.len());
        for i in 0..max_len {
            let cpu_c = if i < cpu_coeffs.len() {
                &cpu_coeffs[i]
            } else {
                &FpE::zero()
            };
            let gpu_c = if i < gpu_coeffs.len() {
                &gpu_coeffs[i]
            } else {
                &FpE::zero()
            };
            assert_eq!(
                cpu_c, gpu_c,
                "DEEP composition poly coefficient {i} mismatch: CPU={cpu_c:?} GPU={gpu_c:?}"
            );
        }
    }
}
