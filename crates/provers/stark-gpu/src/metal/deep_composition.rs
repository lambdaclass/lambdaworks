//! GPU DEEP composition polynomial for Fibonacci RAP.
//!
//! Evaluation-domain approach: pre-compute batch inversions, evaluate the DEEP
//! composition at each LDE domain point on GPU, then optionally IFFT back to
//! coefficient form.

#[cfg(all(target_os = "macos", feature = "metal"))]
use lambdaworks_gpu::metal::abstractions::state::DynamicMetalState;
#[cfg(all(target_os = "macos", feature = "metal"))]
use lambdaworks_math::field::{
    element::FieldElement, fields::u64_goldilocks_field::Goldilocks64Field,
};
#[cfg(all(target_os = "macos", feature = "metal"))]
use lambdaworks_math::polynomial::Polynomial;

#[cfg(all(target_os = "macos", feature = "metal"))]
use crate::metal::phases::composition::GpuRound2Result;
use crate::metal::phases::ood::GpuRound3Result;
use crate::metal::phases::rap::GpuRound1Result;

#[cfg(all(target_os = "macos", feature = "metal"))]
const DEEP_COMPOSITION_SHADER: &str = include_str!("shaders/deep_composition.metal");

#[cfg(all(target_os = "macos", feature = "metal"))]
const DOMAIN_INV_SHADER: &str = include_str!("shaders/domain_inversions.metal");

/// Convert a `MetalError` to `ProvingError`.
#[cfg(all(target_os = "macos", feature = "metal"))]
fn metal_err(
    e: lambdaworks_gpu::metal::abstractions::errors::MetalError,
) -> stark_platinum_prover::prover::ProvingError {
    stark_platinum_prover::prover::ProvingError::FieldOperationError(e.to_string())
}

/// Parameters struct matching the Metal shader's `DeepCompParams`.
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
#[cfg(all(target_os = "macos", feature = "metal"))]
pub struct DeepCompositionState {
    state: DynamicMetalState,
    max_threads: u64,
}

#[cfg(all(target_os = "macos", feature = "metal"))]
impl DeepCompositionState {
    pub fn new() -> Result<Self, lambdaworks_gpu::metal::abstractions::errors::MetalError> {
        let mut state = DynamicMetalState::new()?;
        state.load_library(DEEP_COMPOSITION_SHADER)?;
        let max_threads = state.prepare_pipeline("deep_composition_eval")?;
        Ok(Self { state, max_threads })
    }
}

/// Parameters struct matching the Metal shader's `BatchDomainInvParams`.
#[cfg(all(target_os = "macos", feature = "metal"))]
#[repr(C)]
#[derive(Copy, Clone)]
struct BatchDomainInvParams {
    num_rows: u32,
    chunk_size: u32,
}

#[cfg(all(target_os = "macos", feature = "metal"))]
const BATCH_INV_CHUNK_SIZE: u32 = 16;

/// Pre-compiled Metal state for GPU domain inversions (base field).
#[cfg(all(target_os = "macos", feature = "metal"))]
pub struct DomainInversionState {
    state: DynamicMetalState,
    batch_max_threads: u64,
}

#[cfg(all(target_os = "macos", feature = "metal"))]
impl DomainInversionState {
    pub fn new() -> Result<Self, lambdaworks_gpu::metal::abstractions::errors::MetalError> {
        let combined_source = build_domain_inv_source();
        let mut state = DynamicMetalState::new()?;
        state.load_library(&combined_source)?;
        let batch_max_threads = state.prepare_pipeline("batch_domain_inversions")?;
        Ok(Self {
            state,
            batch_max_threads,
        })
    }
}

#[cfg(all(target_os = "macos", feature = "metal"))]
fn build_domain_inv_source() -> String {
    let shader_clean = DOMAIN_INV_SHADER.replace(
        "#include <metal_stdlib>\nusing namespace metal;",
        "// (metal_stdlib included via header)",
    );
    format!(
        "{}\n{}",
        crate::metal::fp3::FP_U64_HEADER_SOURCE,
        shader_clean
    )
}

/// Pack scalar data (gammas + OOD evals) as raw u64 values for the GPU kernel.
///
/// Layout: [composition_gammas, trace_gammas, composition_ood_evals, trace_ood_evals]
#[cfg(all(target_os = "macos", feature = "metal"))]
fn pack_scalars_base(
    composition_gammas: &[FieldElement<Goldilocks64Field>],
    trace_term_coeffs: &[Vec<FieldElement<Goldilocks64Field>>],
    composition_ood_evals: &[FieldElement<Goldilocks64Field>],
    trace_ood_columns: &[Vec<FieldElement<Goldilocks64Field>>],
    num_comp_parts: usize,
) -> Vec<u64> {
    let mut scalars = Vec::new();
    for gamma in composition_gammas.iter().take(num_comp_parts) {
        scalars.push(canonical(gamma));
    }
    for gammas in trace_term_coeffs {
        for g in gammas {
            scalars.push(canonical(g));
        }
    }
    for eval in composition_ood_evals {
        scalars.push(canonical(eval));
    }
    for col in trace_ood_columns {
        for eval in col {
            scalars.push(canonical(eval));
        }
    }
    scalars
}

/// GPU DEEP composition with CPU batch inversions, returning a `Polynomial`.
///
/// Used only by tests to validate against the CPU Ruffini-based reference.
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
    let num_offsets = round_3_result.trace_ood_evaluations.height;
    let num_comp_parts = round_2_result.composition_poly_parts.len();

    let mut all_trace_lde = round_1_result.main_lde_evaluations.clone();
    all_trace_lde.extend(round_1_result.aux_lde_evaluations.iter().cloned());
    let num_trace_polys = all_trace_lde.len();

    // CPU batch inversions
    let z_power = crate::metal::exp_power_of_2(&round_3_result.z, num_comp_parts.trailing_zeros());
    let primitive_root = &domain.trace_primitive_root;

    let mut inv_z_power_vec: Vec<FpE> = domain
        .lde_roots_of_unity_coset
        .iter()
        .map(|x| x - z_power)
        .collect();
    FieldElement::inplace_batch_inverse_parallel(&mut inv_z_power_vec)
        .map_err(|_| stark_platinum_prover::prover::ProvingError::BatchInversionFailed)?;

    let z_shifted_values: Vec<FpE> = {
        let mut vals = Vec::with_capacity(num_offsets);
        let mut g_pow = FpE::one();
        for _ in 0..num_offsets {
            vals.push(g_pow * round_3_result.z);
            g_pow *= primitive_root;
        }
        vals
    };

    use rayon::prelude::*;
    let inv_z_shifted_vecs: Vec<Vec<FpE>> = z_shifted_values
        .par_iter()
        .map(|z_shifted| {
            let mut denoms: Vec<FpE> = domain
                .lde_roots_of_unity_coset
                .iter()
                .map(|x| x - z_shifted)
                .collect();
            FieldElement::inplace_batch_inverse_parallel(&mut denoms)
                .expect("batch inverse must succeed: coset offset ensures no zeros");
            denoms
        })
        .collect();

    let trace_ood_columns = round_3_result.trace_ood_evaluations.columns();
    let scalars = pack_scalars_base(
        composition_gammas,
        trace_term_coeffs,
        &round_3_result.composition_poly_parts_ood_evaluation,
        &trace_ood_columns,
        num_comp_parts,
    );

    let trace_raw: Vec<Vec<u64>> = all_trace_lde.iter().map(|col| to_raw_u64(col)).collect();
    let comp_raw: Vec<Vec<u64>> = round_2_result
        .lde_composition_poly_evaluations
        .iter()
        .map(|col| to_raw_u64(col))
        .collect();
    let inv_zp_raw: Vec<u64> = to_raw_u64(&inv_z_power_vec);
    let inv_zs_raw: Vec<Vec<u64>> = inv_z_shifted_vecs.iter().map(|v| to_raw_u64(v)).collect();

    let params = DeepCompParams {
        num_rows: num_rows as u32,
        num_trace_polys: num_trace_polys as u32,
        num_offsets: num_offsets as u32,
        num_comp_parts: num_comp_parts as u32,
    };

    let mut owned_state;
    let (dyn_state, max_threads) = match precompiled {
        Some(pre) => (&pre.state, pre.max_threads),
        None => {
            owned_state = DynamicMetalState::new().map_err(metal_err)?;
            owned_state
                .load_library(DEEP_COMPOSITION_SHADER)
                .map_err(metal_err)?;
            let mt = owned_state
                .prepare_pipeline("deep_composition_eval")
                .map_err(metal_err)?;
            (&owned_state, mt)
        }
    };

    let buf_trace_0 = dyn_state
        .alloc_buffer_with_data(&trace_raw[0])
        .map_err(metal_err)?;
    let buf_trace_1 = dyn_state
        .alloc_buffer_with_data(&trace_raw[1])
        .map_err(metal_err)?;
    let buf_trace_2 = dyn_state
        .alloc_buffer_with_data(&trace_raw[2])
        .map_err(metal_err)?;
    let buf_comp_0 = dyn_state
        .alloc_buffer_with_data(&comp_raw[0])
        .map_err(metal_err)?;
    let buf_inv_zp = dyn_state
        .alloc_buffer_with_data(&inv_zp_raw)
        .map_err(metal_err)?;
    let buf_inv_zs0 = dyn_state
        .alloc_buffer_with_data(&inv_zs_raw[0])
        .map_err(metal_err)?;
    let buf_inv_zs1 = dyn_state
        .alloc_buffer_with_data(&inv_zs_raw[1])
        .map_err(metal_err)?;
    let buf_inv_zs2 = dyn_state
        .alloc_buffer_with_data(&inv_zs_raw[2])
        .map_err(metal_err)?;
    let buf_scalars = dyn_state
        .alloc_buffer_with_data(&scalars)
        .map_err(metal_err)?;
    let buf_params = dyn_state
        .alloc_buffer_with_data(std::slice::from_ref(&params))
        .map_err(metal_err)?;
    let buf_output = dyn_state
        .alloc_buffer(num_rows * std::mem::size_of::<u64>())
        .map_err(metal_err)?;

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
        .map_err(metal_err)?;

    let output_raw: Vec<u64> = unsafe { dyn_state.read_buffer(&buf_output, num_rows) };
    let deep_poly_evals: Vec<FpE> = output_raw.into_iter().map(FpE::from).collect();

    Ok(Polynomial::interpolate_offset_fft(
        &deep_poly_evals,
        &domain.coset_offset,
    )?)
}

/// Compute DEEP composition evaluations on GPU, returning the raw evaluations buffer.
///
/// Shared by `gpu_compute_deep_composition_evals_to_buffer` (eval-domain FRI path).
#[cfg(all(target_os = "macos", feature = "metal"))]
#[allow(clippy::too_many_arguments)]
fn gpu_compute_deep_composition_evals_internal(
    round_1_result: &GpuRound1Result<Goldilocks64Field>,
    round_2_result: &GpuRound2Result<Goldilocks64Field>,
    round_3_result: &GpuRound3Result<Goldilocks64Field>,
    domain: &stark_platinum_prover::domain::Domain<Goldilocks64Field>,
    composition_gammas: &[FieldElement<Goldilocks64Field>],
    trace_term_coeffs: &[Vec<FieldElement<Goldilocks64Field>>],
    precompiled: Option<&DeepCompositionState>,
    domain_inv_state: Option<&DomainInversionState>,
    lde_coset_buf: Option<&metal::Buffer>,
) -> Result<(metal::Buffer, usize), stark_platinum_prover::prover::ProvingError> {
    let num_rows = domain.lde_roots_of_unity_coset.len();
    let num_offsets = round_3_result.trace_ood_evaluations.height;
    let num_comp_parts = round_2_result.composition_poly_parts.len();
    let num_trace_polys =
        round_1_result.main_trace_polys.len() + round_1_result.aux_trace_polys.len();

    let trace_ood_columns = round_3_result.trace_ood_evaluations.columns();
    let scalars = pack_scalars_base(
        composition_gammas,
        trace_term_coeffs,
        &round_3_result.composition_poly_parts_ood_evaluation,
        &trace_ood_columns,
        num_comp_parts,
    );

    let params = DeepCompParams {
        num_rows: num_rows as u32,
        num_trace_polys: num_trace_polys as u32,
        num_offsets: num_offsets as u32,
        num_comp_parts: num_comp_parts as u32,
    };

    let mut owned_state;
    let (dyn_state, max_threads) = match precompiled {
        Some(pre) => (&pre.state, pre.max_threads),
        None => {
            owned_state = DynamicMetalState::new().map_err(metal_err)?;
            owned_state
                .load_library(DEEP_COMPOSITION_SHADER)
                .map_err(metal_err)?;
            let mt = owned_state
                .prepare_pipeline("deep_composition_eval")
                .map_err(metal_err)?;
            (&owned_state, mt)
        }
    };

    // Use retained GPU buffers from Phase 1/Phase 2 when available (zero-copy),
    // otherwise fall back to clone + convert + upload.
    let has_gpu_trace_bufs = round_1_result
        .main_lde_gpu_buffers
        .as_ref()
        .is_some_and(|b| b.len() >= 2)
        && round_1_result
            .aux_lde_gpu_buffers
            .as_ref()
            .is_some_and(|b| !b.is_empty());
    let has_gpu_comp_bufs = round_2_result
        .lde_composition_gpu_buffers
        .as_ref()
        .is_some_and(|b| !b.is_empty());

    let mut _owned_trace_bufs: Vec<metal::Buffer> = Vec::new();
    let mut _owned_comp_bufs: Vec<metal::Buffer> = Vec::new();

    let (trace_buf_0, trace_buf_1, trace_buf_2, comp_buf_0) = if has_gpu_trace_bufs
        && has_gpu_comp_bufs
    {
        let main_bufs = round_1_result.main_lde_gpu_buffers.as_ref().unwrap();
        let aux_bufs = round_1_result.aux_lde_gpu_buffers.as_ref().unwrap();
        let comp_bufs = round_2_result.lde_composition_gpu_buffers.as_ref().unwrap();
        (&main_bufs[0], &main_bufs[1], &aux_bufs[0], &comp_bufs[0])
    } else {
        let mut all_trace_lde = round_1_result.main_lde_evaluations.clone();
        all_trace_lde.extend(round_1_result.aux_lde_evaluations.iter().cloned());

        let trace_raw: Vec<Vec<u64>> = all_trace_lde.iter().map(|col| to_raw_u64(col)).collect();
        let comp_raw: Vec<Vec<u64>> = round_2_result
            .lde_composition_poly_evaluations
            .iter()
            .map(|col| to_raw_u64(col))
            .collect();

        for raw in &trace_raw {
            _owned_trace_bufs.push(dyn_state.alloc_buffer_with_data(raw).map_err(metal_err)?);
        }
        _owned_comp_bufs.push(
            dyn_state
                .alloc_buffer_with_data(&comp_raw[0])
                .map_err(metal_err)?,
        );

        (
            &_owned_trace_bufs[0],
            &_owned_trace_bufs[1],
            &_owned_trace_bufs[2],
            &_owned_comp_bufs[0],
        )
    };

    // Compute inversions on GPU
    let z_power = crate::metal::exp_power_of_2(&round_3_result.z, num_comp_parts.trailing_zeros());
    let primitive_root = &domain.trace_primitive_root;
    let z_shifted_0 = round_3_result.z;
    let z_shifted_1 = round_3_result.z * primitive_root;
    let z_shifted_2 = round_3_result.z * (primitive_root * primitive_root);

    let (buf_inv_zp, buf_inv_zs0, buf_inv_zs1, buf_inv_zs2) = gpu_compute_domain_inversions_base(
        domain,
        &z_power,
        &z_shifted_0,
        &z_shifted_1,
        &z_shifted_2,
        num_rows,
        domain_inv_state,
        lde_coset_buf,
    )
    .map_err(metal_err)?;

    let buf_scalars = dyn_state
        .alloc_buffer_with_data(&scalars)
        .map_err(metal_err)?;
    let buf_params = dyn_state
        .alloc_buffer_with_data(std::slice::from_ref(&params))
        .map_err(metal_err)?;
    let buf_output = dyn_state
        .alloc_buffer(num_rows * std::mem::size_of::<u64>())
        .map_err(metal_err)?;

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
        .map_err(metal_err)?;

    Ok((buf_output, num_rows))
}

/// Compute DEEP composition evaluations on GPU, returning raw evaluations as a Metal Buffer.
///
/// Skips the IFFT step, returning evaluations in natural (LDE coset) order.
/// Used by the eval-domain FRI path which operates directly on evaluations.
#[cfg(all(target_os = "macos", feature = "metal"))]
#[allow(clippy::too_many_arguments)]
pub fn gpu_compute_deep_composition_evals_to_buffer(
    round_1_result: &GpuRound1Result<Goldilocks64Field>,
    round_2_result: &GpuRound2Result<Goldilocks64Field>,
    round_3_result: &GpuRound3Result<Goldilocks64Field>,
    domain: &stark_platinum_prover::domain::Domain<Goldilocks64Field>,
    composition_gammas: &[FieldElement<Goldilocks64Field>],
    trace_term_coeffs: &[Vec<FieldElement<Goldilocks64Field>>],
    precompiled: Option<&DeepCompositionState>,
    domain_inv_state: Option<&DomainInversionState>,
    lde_coset_buf: Option<&metal::Buffer>,
) -> Result<(metal::Buffer, usize), stark_platinum_prover::prover::ProvingError> {
    gpu_compute_deep_composition_evals_internal(
        round_1_result,
        round_2_result,
        round_3_result,
        domain,
        composition_gammas,
        trace_term_coeffs,
        precompiled,
        domain_inv_state,
        lde_coset_buf,
    )
}

/// Dispatch GPU batch Montgomery inversions for the base field.
///
/// Returns 4 Metal Buffers: inv(x-z^N), inv(x-z*g^0), inv(x-z*g^1), inv(x-z*g^2).
#[cfg(all(target_os = "macos", feature = "metal"))]
#[allow(clippy::too_many_arguments)]
fn gpu_compute_domain_inversions_base(
    domain: &stark_platinum_prover::domain::Domain<Goldilocks64Field>,
    z_power: &FieldElement<Goldilocks64Field>,
    z_shifted_0: &FieldElement<Goldilocks64Field>,
    z_shifted_1: &FieldElement<Goldilocks64Field>,
    z_shifted_2: &FieldElement<Goldilocks64Field>,
    num_rows: usize,
    precompiled: Option<&DomainInversionState>,
    lde_coset_buf: Option<&metal::Buffer>,
) -> Result<
    (metal::Buffer, metal::Buffer, metal::Buffer, metal::Buffer),
    lambdaworks_gpu::metal::abstractions::errors::MetalError,
> {
    let mut owned_state;
    let (inv_state, inv_max_threads) = match precompiled {
        Some(pre) => (&pre.state, pre.batch_max_threads),
        None => {
            let combined_source = build_domain_inv_source();
            owned_state = DynamicMetalState::new()?;
            owned_state.load_library(&combined_source)?;
            let mt = owned_state.prepare_pipeline("batch_domain_inversions")?;
            (&owned_state, mt)
        }
    };

    let _owned_domain_buf;
    let buf_domain: &metal::Buffer = if let Some(buf) = lde_coset_buf {
        buf
    } else {
        let domain_raw: Vec<u64> = to_raw_u64(&domain.lde_roots_of_unity_coset);
        _owned_domain_buf = inv_state.alloc_buffer_with_data(&domain_raw)?;
        &_owned_domain_buf
    };

    let z_packed: [u64; 4] = [
        canonical(z_power),
        canonical(z_shifted_0),
        canonical(z_shifted_1),
        canonical(z_shifted_2),
    ];
    let buf_z_values = inv_state.alloc_buffer_with_data(&z_packed)?;

    let buf_size = num_rows * std::mem::size_of::<u64>();
    let buf_inv_zp = inv_state.alloc_buffer(buf_size)?;
    let buf_inv_zs0 = inv_state.alloc_buffer(buf_size)?;
    let buf_inv_zs1 = inv_state.alloc_buffer(buf_size)?;
    let buf_inv_zs2 = inv_state.alloc_buffer(buf_size)?;

    let inv_params = BatchDomainInvParams {
        num_rows: num_rows as u32,
        chunk_size: BATCH_INV_CHUNK_SIZE,
    };
    let buf_params = inv_state.alloc_buffer_with_data(std::slice::from_ref(&inv_params))?;

    let num_threads = (num_rows as u64).div_ceil(BATCH_INV_CHUNK_SIZE as u64);

    inv_state.execute_compute(
        "batch_domain_inversions",
        &[
            buf_domain,
            &buf_z_values,
            &buf_inv_zp,
            &buf_inv_zs0,
            &buf_inv_zs1,
            &buf_inv_zs2,
            &buf_params,
        ],
        num_threads,
        inv_max_threads,
    )?;

    Ok((buf_inv_zp, buf_inv_zs0, buf_inv_zs1, buf_inv_zs2))
}

// =============================================================================
// Fp3 DEEP Composition (degree-3 Goldilocks extension)
// =============================================================================

#[cfg(all(target_os = "macos", feature = "metal"))]
const DEEP_COMPOSITION_FP3_SHADER: &str = include_str!("shaders/deep_composition_fp3.metal");

#[cfg(all(target_os = "macos", feature = "metal"))]
const DOMAIN_INV_FP3_SHADER: &str = include_str!("shaders/domain_inversions_fp3.metal");

#[cfg(all(target_os = "macos", feature = "metal"))]
#[repr(C)]
#[derive(Copy, Clone)]
struct DomainInvFp3Params {
    num_rows: u32,
}

/// Pre-compiled Metal state for GPU domain inversions (Fp3 extension field).
#[cfg(all(target_os = "macos", feature = "metal"))]
pub struct DomainInversionFp3State {
    state: DynamicMetalState,
    max_threads: u64,
}

#[cfg(all(target_os = "macos", feature = "metal"))]
impl DomainInversionFp3State {
    pub fn new() -> Result<Self, lambdaworks_gpu::metal::abstractions::errors::MetalError> {
        let combined_source = crate::metal::fp3::combined_fp3_source(DOMAIN_INV_FP3_SHADER);
        let mut state = DynamicMetalState::new()?;
        state.load_library(&combined_source)?;
        let max_threads = state.prepare_pipeline("compute_domain_inversions_fp3")?;
        Ok(Self { state, max_threads })
    }
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[repr(C)]
#[derive(Copy, Clone)]
struct DeepCompFp3Params {
    num_rows: u32,
    num_trace_polys: u32,
    num_offsets: u32,
    num_comp_parts: u32,
}

/// Pre-compiled Metal state for the Fp3 DEEP composition kernel.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub struct DeepCompositionFp3State {
    state: DynamicMetalState,
    max_threads: u64,
}

#[cfg(all(target_os = "macos", feature = "metal"))]
impl DeepCompositionFp3State {
    pub fn new() -> Result<Self, lambdaworks_gpu::metal::abstractions::errors::MetalError> {
        let combined_source = crate::metal::fp3::combined_fp3_source(DEEP_COMPOSITION_FP3_SHADER);
        let mut state = DynamicMetalState::new()?;
        state.load_library(&combined_source)?;
        let max_threads = state.prepare_pipeline("deep_composition_fp3_eval")?;
        Ok(Self { state, max_threads })
    }
}

use crate::metal::{canonical, fp3_to_u64s, to_raw_u64};

/// Dispatch GPU domain inversions for Fp3 extension field.
///
/// Domain points are base field; z-values are Fp3. Returns 4 Metal Buffers,
/// each containing `num_rows * 3` u64s (3 components per Fp3 element).
#[cfg(all(target_os = "macos", feature = "metal"))]
fn gpu_compute_domain_inversions_fp3(
    domain: &stark_platinum_prover::domain::Domain<Goldilocks64Field>,
    z_power: &FieldElement<
        lambdaworks_math::field::fields::u64_goldilocks_field::Degree3GoldilocksExtensionField,
    >,
    z_shifted_0: &FieldElement<
        lambdaworks_math::field::fields::u64_goldilocks_field::Degree3GoldilocksExtensionField,
    >,
    z_shifted_1: &FieldElement<
        lambdaworks_math::field::fields::u64_goldilocks_field::Degree3GoldilocksExtensionField,
    >,
    z_shifted_2: &FieldElement<
        lambdaworks_math::field::fields::u64_goldilocks_field::Degree3GoldilocksExtensionField,
    >,
    num_rows: usize,
    precompiled: Option<&DomainInversionFp3State>,
) -> Result<
    (metal::Buffer, metal::Buffer, metal::Buffer, metal::Buffer),
    lambdaworks_gpu::metal::abstractions::errors::MetalError,
> {
    let mut owned_state;
    let (inv_state, inv_max_threads) = match precompiled {
        Some(pre) => (&pre.state, pre.max_threads),
        None => {
            let combined_source = crate::metal::fp3::combined_fp3_source(DOMAIN_INV_FP3_SHADER);
            owned_state = DynamicMetalState::new()?;
            owned_state.load_library(&combined_source)?;
            let mt = owned_state.prepare_pipeline("compute_domain_inversions_fp3")?;
            (&owned_state, mt)
        }
    };

    let domain_raw: Vec<u64> = to_raw_u64(&domain.lde_roots_of_unity_coset);
    let buf_domain = inv_state.alloc_buffer_with_data(&domain_raw)?;

    let buf_zp = inv_state.alloc_buffer_with_data(&fp3_to_u64s(z_power))?;
    let buf_zs0 = inv_state.alloc_buffer_with_data(&fp3_to_u64s(z_shifted_0))?;
    let buf_zs1 = inv_state.alloc_buffer_with_data(&fp3_to_u64s(z_shifted_1))?;
    let buf_zs2 = inv_state.alloc_buffer_with_data(&fp3_to_u64s(z_shifted_2))?;

    let buf_size = num_rows * 3 * std::mem::size_of::<u64>();
    let buf_inv_zp = inv_state.alloc_buffer(buf_size)?;
    let buf_inv_zs0 = inv_state.alloc_buffer(buf_size)?;
    let buf_inv_zs1 = inv_state.alloc_buffer(buf_size)?;
    let buf_inv_zs2 = inv_state.alloc_buffer(buf_size)?;

    let inv_params = DomainInvFp3Params {
        num_rows: num_rows as u32,
    };
    let buf_params = inv_state.alloc_buffer_with_data(std::slice::from_ref(&inv_params))?;

    inv_state.execute_compute(
        "compute_domain_inversions_fp3",
        &[
            &buf_domain,
            &buf_zp,
            &buf_zs0,
            &buf_zs1,
            &buf_zs2,
            &buf_inv_zp,
            &buf_inv_zs0,
            &buf_inv_zs1,
            &buf_inv_zs2,
            &buf_params,
        ],
        num_rows as u64,
        inv_max_threads,
    )?;

    Ok((buf_inv_zp, buf_inv_zs0, buf_inv_zs1, buf_inv_zs2))
}

/// Compute the Fp3 DEEP composition on GPU, returning evaluations in Fp3.
///
/// Trace LDE values remain in base field; gammas, z, OOD evals, inversions,
/// and output are all in Fp3 (degree-3 Goldilocks extension).
#[cfg(all(target_os = "macos", feature = "metal"))]
#[allow(clippy::too_many_arguments)]
pub fn gpu_compute_deep_composition_poly_fp3(
    round_1_result: &GpuRound1Result<Goldilocks64Field>,
    round_3_z: &FieldElement<
        lambdaworks_math::field::fields::u64_goldilocks_field::Degree3GoldilocksExtensionField,
    >,
    trace_ood_evaluations_fp3: &[Vec<
        FieldElement<
            lambdaworks_math::field::fields::u64_goldilocks_field::Degree3GoldilocksExtensionField,
        >,
    >],
    composition_ood_evaluations_fp3: &[FieldElement<
        lambdaworks_math::field::fields::u64_goldilocks_field::Degree3GoldilocksExtensionField,
    >],
    lde_composition_poly_evaluations_fp3: &[Vec<
        FieldElement<
            lambdaworks_math::field::fields::u64_goldilocks_field::Degree3GoldilocksExtensionField,
        >,
    >],
    domain: &stark_platinum_prover::domain::Domain<Goldilocks64Field>,
    composition_gammas_fp3: &[FieldElement<
        lambdaworks_math::field::fields::u64_goldilocks_field::Degree3GoldilocksExtensionField,
    >],
    trace_term_coeffs_fp3: &[Vec<
        FieldElement<
            lambdaworks_math::field::fields::u64_goldilocks_field::Degree3GoldilocksExtensionField,
        >,
    >],
    precompiled: Option<&DeepCompositionFp3State>,
    domain_inv_fp3_state: Option<&DomainInversionFp3State>,
) -> Result<
    Vec<
        FieldElement<
            lambdaworks_math::field::fields::u64_goldilocks_field::Degree3GoldilocksExtensionField,
        >,
    >,
    stark_platinum_prover::prover::ProvingError,
> {
    use lambdaworks_math::field::fields::u64_goldilocks_field::Degree3GoldilocksExtensionField;

    type Fp3 = Degree3GoldilocksExtensionField;
    type Fp3E = FieldElement<Fp3>;

    let num_rows = domain.lde_roots_of_unity_coset.len();
    let num_offsets = trace_ood_evaluations_fp3.len();
    let num_comp_parts = composition_ood_evaluations_fp3.len();
    let num_trace_polys =
        round_1_result.main_trace_polys.len() + round_1_result.aux_trace_polys.len();

    // Compute inversions on GPU (in Fp3)
    let z_power: Fp3E = crate::metal::exp_power_of_2(round_3_z, num_comp_parts.trailing_zeros());
    let primitive_root = &domain.trace_primitive_root;
    let z_shifted_0: Fp3E = *round_3_z;
    let z_shifted_1: Fp3E = primitive_root * round_3_z;
    let z_shifted_2: Fp3E = (primitive_root * primitive_root) * round_3_z;

    let (buf_inv_zp, buf_inv_zs0, buf_inv_zs1, buf_inv_zs2) = gpu_compute_domain_inversions_fp3(
        domain,
        &z_power,
        &z_shifted_0,
        &z_shifted_1,
        &z_shifted_2,
        num_rows,
        domain_inv_fp3_state,
    )
    .map_err(metal_err)?;

    // Pack scalar data (gammas + OOD evals) as raw u64 triples
    let mut scalars: Vec<u64> = Vec::new();
    for gamma in composition_gammas_fp3.iter().take(num_comp_parts) {
        scalars.extend_from_slice(&fp3_to_u64s(gamma));
    }
    for gammas in trace_term_coeffs_fp3 {
        for g in gammas {
            scalars.extend_from_slice(&fp3_to_u64s(g));
        }
    }
    for eval in composition_ood_evaluations_fp3 {
        scalars.extend_from_slice(&fp3_to_u64s(eval));
    }
    for col in trace_ood_evaluations_fp3 {
        for eval in col {
            scalars.extend_from_slice(&fp3_to_u64s(eval));
        }
    }

    // Convert trace LDE data to raw u64 (base field)
    let mut all_trace_lde = round_1_result.main_lde_evaluations.clone();
    all_trace_lde.extend(round_1_result.aux_lde_evaluations.iter().cloned());
    let trace_raw: Vec<Vec<u64>> = all_trace_lde.iter().map(|col| to_raw_u64(col)).collect();

    // Convert Fp3 composition LDE data to raw u64 triples
    let comp_raw: Vec<Vec<u64>> = lde_composition_poly_evaluations_fp3
        .iter()
        .map(|col| col.iter().flat_map(fp3_to_u64s).collect())
        .collect();

    let params = DeepCompFp3Params {
        num_rows: num_rows as u32,
        num_trace_polys: num_trace_polys as u32,
        num_offsets: num_offsets as u32,
        num_comp_parts: num_comp_parts as u32,
    };

    let mut owned_state;
    let (dyn_state, max_threads) = match precompiled {
        Some(pre) => (&pre.state, pre.max_threads),
        None => {
            let combined_source =
                crate::metal::fp3::combined_fp3_source(DEEP_COMPOSITION_FP3_SHADER);
            owned_state = DynamicMetalState::new().map_err(metal_err)?;
            owned_state
                .load_library(&combined_source)
                .map_err(metal_err)?;
            let mt = owned_state
                .prepare_pipeline("deep_composition_fp3_eval")
                .map_err(metal_err)?;
            (&owned_state, mt)
        }
    };

    let buf_trace_0 = dyn_state
        .alloc_buffer_with_data(&trace_raw[0])
        .map_err(metal_err)?;
    let buf_trace_1 = dyn_state
        .alloc_buffer_with_data(&trace_raw[1])
        .map_err(metal_err)?;
    let buf_trace_2 = dyn_state
        .alloc_buffer_with_data(&trace_raw[2])
        .map_err(metal_err)?;
    let buf_comp_0 = dyn_state
        .alloc_buffer_with_data(&comp_raw[0])
        .map_err(metal_err)?;
    let buf_scalars = dyn_state
        .alloc_buffer_with_data(&scalars)
        .map_err(metal_err)?;
    let buf_params = dyn_state
        .alloc_buffer_with_data(std::slice::from_ref(&params))
        .map_err(metal_err)?;
    let buf_output = dyn_state
        .alloc_buffer(num_rows * 3 * std::mem::size_of::<u64>())
        .map_err(metal_err)?;

    dyn_state
        .execute_compute(
            "deep_composition_fp3_eval",
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
        .map_err(metal_err)?;

    let output_raw: Vec<u64> =
        lambdaworks_gpu::metal::abstractions::state::MetalState::retrieve_contents(&buf_output);
    let deep_poly_evals: Vec<Fp3E> = output_raw
        .chunks_exact(3)
        .map(|chunk| {
            Fp3E::new([
                FieldElement::from_raw(chunk[0]),
                FieldElement::from_raw(chunk[1]),
                FieldElement::from_raw(chunk[2]),
            ])
        })
        .collect();

    Ok(deep_poly_evals)
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

    /// Set up a Fibonacci RAP test fixture (trace, AIR, domain, rounds 1-3).
    fn setup_fibonacci_rap() -> (
        stark_platinum_prover::examples::fibonacci_rap::FibonacciRAP<F>,
        Domain<F>,
        StarkMetalState,
        GpuRound1Result<F>,
        GpuRound2Result<F>,
        GpuRound3Result<F>,
        DefaultTranscript<F>,
    ) {
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

        (air, domain, state, round_1, round_2, round_3, transcript)
    }

    /// Sample deep composition coefficients from a transcript, split into
    /// trace term coefficients and composition gammas.
    fn sample_deep_coefficients(
        transcript: &mut DefaultTranscript<F>,
        air: &impl AIR<Field = F, FieldExtension = F>,
        n_terms_composition_poly: usize,
    ) -> (Vec<Vec<FpE>>, Vec<FpE>) {
        let gamma: FpE = transcript.sample_field_element();
        let num_terms_trace =
            air.context().transition_offsets.len() * air.step_size() * air.context().trace_columns;

        let mut coefficients: Vec<FpE> =
            core::iter::successors(Some(FpE::one()), |x| Some(x * gamma))
                .take(n_terms_composition_poly + num_terms_trace)
                .collect();

        let trace_term_coeffs: Vec<Vec<FpE>> = coefficients
            .drain(..num_terms_trace)
            .collect::<Vec<_>>()
            .chunks(air.context().transition_offsets.len() * air.step_size())
            .map(|chunk| chunk.to_vec())
            .collect();

        (trace_term_coeffs, coefficients)
    }

    /// Differential test: GPU DEEP composition (evaluation-domain + IFFT) vs CPU Ruffini division.
    #[test]
    fn gpu_deep_composition_matches_cpu() {
        let (air, domain, state, round_1, round_2, round_3, mut transcript) = setup_fibonacci_rap();

        let (trace_term_coeffs, composition_gammas) = sample_deep_coefficients(
            &mut transcript,
            &air,
            round_2.lde_composition_poly_evaluations.len(),
        );

        // CPU reference (Ruffini division)
        let mut all_trace_polys_cpu = round_1.main_trace_polys.clone();
        all_trace_polys_cpu.extend(round_1.aux_trace_polys.iter().cloned());

        let z_power = crate::metal::exp_power_of_2(
            &round_3.z,
            round_2.composition_poly_parts.len().trailing_zeros(),
        );
        let primitive_root = &domain.trace_primitive_root;

        let mut h_terms = Polynomial::zero();
        for (i, part) in round_2.composition_poly_parts.iter().enumerate() {
            let h_i_eval = &round_3.composition_poly_parts_ood_evaluation[i];
            let h_i_term = composition_gammas[i] * (part - h_i_eval);
            h_terms += h_i_term;
        }
        h_terms.ruffini_division_inplace(&z_power);

        let trace_evaluations_columns = round_3.trace_ood_evaluations.columns();
        let num_offsets = round_3.trace_ood_evaluations.height;
        let z_shifted_values: Vec<FpE> = {
            let mut vals = Vec::with_capacity(num_offsets);
            let mut g_pow = FpE::one();
            for _ in 0..num_offsets {
                vals.push(g_pow * round_3.z);
                g_pow *= primitive_root;
            }
            vals
        };

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

        // GPU path
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

        let cpu_coeffs = cpu_deep_poly.coefficients();
        let gpu_coeffs = gpu_deep_poly.coefficients();

        let max_len = cpu_coeffs.len().max(gpu_coeffs.len());
        let zero = FpE::zero();
        for i in 0..max_len {
            let cpu_c = cpu_coeffs.get(i).unwrap_or(&zero);
            let gpu_c = gpu_coeffs.get(i).unwrap_or(&zero);
            assert_eq!(
                cpu_c, gpu_c,
                "DEEP composition poly coefficient {i} mismatch: CPU={cpu_c:?} GPU={gpu_c:?}"
            );
        }
    }

    /// Differential test: Fp3 DEEP composition GPU vs CPU.
    #[test]
    fn gpu_deep_composition_fp3_matches_cpu() {
        use lambdaworks_math::field::fields::u64_goldilocks_field::Degree3GoldilocksExtensionField;
        type Fp3 = Degree3GoldilocksExtensionField;
        type Fp3E = FieldElement<Fp3>;

        let (_air, domain, _state, round_1, round_2, round_3, mut transcript) =
            setup_fibonacci_rap();

        let z_fp3 = Fp3E::new([round_3.z, FpE::from(7u64), FpE::from(13u64)]);

        let gamma_base: FpE = transcript.sample_field_element();
        let gamma_fp3 = Fp3E::new([gamma_base, FpE::from(3u64), FpE::from(5u64)]);

        let n_terms_composition_poly = round_2.lde_composition_poly_evaluations.len();
        let num_offsets = round_3.trace_ood_evaluations.height;
        let num_trace_polys =
            round_1.main_lde_evaluations.len() + round_1.aux_lde_evaluations.len();
        let num_terms_trace = num_offsets * num_trace_polys;

        let mut deep_coeffs_fp3: Vec<Fp3E> =
            core::iter::successors(Some(Fp3E::one()), |x| Some(x * gamma_fp3))
                .take(n_terms_composition_poly + num_terms_trace)
                .collect();

        let trace_term_coeffs_fp3: Vec<Vec<Fp3E>> = deep_coeffs_fp3
            .drain(..num_terms_trace)
            .collect::<Vec<_>>()
            .chunks(num_offsets)
            .map(|chunk| chunk.to_vec())
            .collect();
        let composition_gammas_fp3 = deep_coeffs_fp3;

        let primitive_root = &domain.trace_primitive_root;
        let z_shifted_fp3: Vec<Fp3E> = {
            let mut vals = Vec::with_capacity(num_offsets);
            let mut g_pow = FieldElement::<Goldilocks64Field>::one();
            for _ in 0..num_offsets {
                vals.push(g_pow * z_fp3);
                g_pow *= primitive_root;
            }
            vals
        };

        let mut all_trace_polys = round_1.main_trace_polys.clone();
        all_trace_polys.extend(round_1.aux_trace_polys.iter().cloned());

        let trace_ood_fp3: Vec<Vec<Fp3E>> = (0..num_trace_polys)
            .map(|col| {
                z_shifted_fp3
                    .iter()
                    .map(|zs| all_trace_polys[col].evaluate(zs))
                    .collect()
            })
            .collect();

        let z_power_fp3: Fp3E = crate::metal::exp_power_of_2(
            &z_fp3,
            round_2.composition_poly_parts.len().trailing_zeros(),
        );
        let composition_ood_fp3: Vec<Fp3E> = round_2
            .composition_poly_parts
            .iter()
            .map(|part| part.evaluate(&z_power_fp3))
            .collect();

        let lde_comp_evals_fp3: Vec<Vec<Fp3E>> = round_2
            .lde_composition_poly_evaluations
            .iter()
            .map(|col| {
                col.iter()
                    .map(|fe| Fp3E::new([*fe, FpE::zero(), FpE::zero()]))
                    .collect()
            })
            .collect();

        // CPU reference: evaluate DEEP composition in Fp3
        let num_rows = domain.lde_roots_of_unity_coset.len();

        let mut inv_z_power: Vec<Fp3E> = domain
            .lde_roots_of_unity_coset
            .iter()
            .map(|x| x - z_power_fp3)
            .collect();
        FieldElement::inplace_batch_inverse(&mut inv_z_power).unwrap();

        let mut inv_z_shifted: Vec<Vec<Fp3E>> = z_shifted_fp3
            .iter()
            .map(|zs| {
                domain
                    .lde_roots_of_unity_coset
                    .iter()
                    .map(|x| x - zs)
                    .collect()
            })
            .collect();
        for v in &mut inv_z_shifted {
            FieldElement::inplace_batch_inverse(v).unwrap();
        }

        let mut all_trace_lde = round_1.main_lde_evaluations.clone();
        all_trace_lde.extend(round_1.aux_lde_evaluations.iter().cloned());

        let mut cpu_deep_evals: Vec<Fp3E> = vec![Fp3E::zero(); num_rows];
        for i in 0..num_rows {
            let mut acc = Fp3E::zero();
            for (k, comp_eval_col) in lde_comp_evals_fp3.iter().enumerate() {
                acc += composition_gammas_fp3[k]
                    * (&comp_eval_col[i] - &composition_ood_fp3[k])
                    * inv_z_power[i];
            }
            for (col, trace_col) in all_trace_lde.iter().enumerate() {
                let t_x_bf = &trace_col[i];
                for (offset, zs_inv) in inv_z_shifted.iter().enumerate() {
                    let diff: Fp3E = t_x_bf - &trace_ood_fp3[col][offset];
                    acc += trace_term_coeffs_fp3[col][offset] * diff * zs_inv[i];
                }
            }
            cpu_deep_evals[i] = acc;
        }

        // GPU path
        let gpu_deep_evals = gpu_compute_deep_composition_poly_fp3(
            &round_1,
            &z_fp3,
            &trace_ood_fp3,
            &composition_ood_fp3,
            &lde_comp_evals_fp3,
            &domain,
            &composition_gammas_fp3,
            &trace_term_coeffs_fp3,
            None,
            None,
        )
        .unwrap();

        assert_eq!(gpu_deep_evals.len(), cpu_deep_evals.len());
        for (i, (gpu, cpu)) in gpu_deep_evals.iter().zip(cpu_deep_evals.iter()).enumerate() {
            assert_eq!(gpu, cpu, "Fp3 DEEP composition eval mismatch at point {i}");
        }
    }
}
