//! GPU constraint evaluation for the Fibonacci RAP AIR.
//!
//! Uses Metal shaders to evaluate transition and boundary constraints in parallel
//! across all LDE domain points.

#[cfg(all(target_os = "macos", feature = "metal"))]
use lambdaworks_gpu::metal::abstractions::{errors::MetalError, state::DynamicMetalState};
#[cfg(all(target_os = "macos", feature = "metal"))]
use lambdaworks_math::field::{
    element::FieldElement, fields::u64_goldilocks_field::Goldilocks64Field, traits::IsPrimeField,
};

#[cfg(all(target_os = "macos", feature = "metal"))]
const FIBONACCI_RAP_SHADER: &str = include_str!("shaders/fibonacci_rap_constraints.metal");

/// Parameters matching the Metal shader's `FibRapParams`.
#[cfg(all(target_os = "macos", feature = "metal"))]
#[repr(C)]
#[derive(Copy, Clone)]
struct FibRapParams {
    lde_step_size: u32,
    num_rows: u32,
    zerofier_0_len: u32,
    zerofier_1_len: u32,
    gamma: u64,
    transition_coeff_0: u64,
    transition_coeff_1: u64,
}

/// Converts a slice of field elements to their canonical u64 representations.
#[cfg(all(target_os = "macos", feature = "metal"))]
fn to_raw_u64(elems: &[FieldElement<Goldilocks64Field>]) -> Vec<u64> {
    elems
        .iter()
        .map(|fe| Goldilocks64Field::canonical(fe.value()))
        .collect()
}

/// Shorthand for extracting a single canonical u64 value.
#[cfg(all(target_os = "macos", feature = "metal"))]
fn canonical(fe: &FieldElement<Goldilocks64Field>) -> u64 {
    Goldilocks64Field::canonical(fe.value())
}

/// Builds `FibRapParams` from common arguments.
#[cfg(all(target_os = "macos", feature = "metal"))]
fn build_fib_rap_params(
    lde_step_size: usize,
    num_rows: usize,
    z0_len: usize,
    z1_len: usize,
    gamma: &FieldElement<Goldilocks64Field>,
    transition_coefficients: &[FieldElement<Goldilocks64Field>],
) -> FibRapParams {
    FibRapParams {
        lde_step_size: lde_step_size as u32,
        num_rows: num_rows as u32,
        zerofier_0_len: z0_len as u32,
        zerofier_1_len: z1_len as u32,
        gamma: canonical(gamma),
        transition_coeff_0: canonical(&transition_coefficients[0]),
        transition_coeff_1: canonical(&transition_coefficients[1]),
    }
}

/// Resolves a precompiled Metal state or creates a fresh one for the given kernel.
///
/// Returns a reference to the `DynamicMetalState` and the max thread count.
/// When `precompiled` is `None`, the caller must keep `owned_state` alive.
#[cfg(all(target_os = "macos", feature = "metal"))]
fn resolve_state<'a, S: HasMetalState>(
    precompiled: Option<&'a S>,
    owned_state: &'a mut Option<DynamicMetalState>,
    kernel_name: &str,
) -> Result<(&'a DynamicMetalState, u64), MetalError> {
    if let Some(pre) = precompiled {
        Ok((pre.metal_state(), pre.metal_max_threads()))
    } else {
        let mut state = DynamicMetalState::new()?;
        state.load_library(FIBONACCI_RAP_SHADER)?;
        let mt = state.prepare_pipeline(kernel_name)?;
        *owned_state = Some(state);
        Ok((owned_state.as_ref().unwrap(), mt))
    }
}

/// Trait for pre-compiled Metal state types to share the state resolution logic.
#[cfg(all(target_os = "macos", feature = "metal"))]
trait HasMetalState {
    fn metal_state(&self) -> &DynamicMetalState;
    fn metal_max_threads(&self) -> u64;
}

/// Pre-compiled Metal state for Fibonacci RAP constraint evaluation.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub struct FibRapConstraintState {
    state: DynamicMetalState,
    max_threads: u64,
}

#[cfg(all(target_os = "macos", feature = "metal"))]
impl HasMetalState for FibRapConstraintState {
    fn metal_state(&self) -> &DynamicMetalState {
        &self.state
    }
    fn metal_max_threads(&self) -> u64 {
        self.max_threads
    }
}

#[cfg(all(target_os = "macos", feature = "metal"))]
impl FibRapConstraintState {
    pub fn new() -> Result<Self, MetalError> {
        let mut state = DynamicMetalState::new()?;
        state.load_library(FIBONACCI_RAP_SHADER)?;
        let max_threads = state.prepare_pipeline("fibonacci_rap_constraint_eval")?;
        Ok(Self { state, max_threads })
    }
}

/// Evaluates Fibonacci RAP constraints on the GPU, returning CPU `Vec`.
///
/// Boundary constraint evaluations must be pre-computed on CPU and passed in.
/// If `precompiled` is `Some`, reuses the compiled pipeline; otherwise compiles from scratch.
#[cfg(all(target_os = "macos", feature = "metal"))]
#[allow(clippy::too_many_arguments)]
pub fn gpu_evaluate_fibonacci_rap_constraints(
    main_col_0: &[FieldElement<Goldilocks64Field>],
    main_col_1: &[FieldElement<Goldilocks64Field>],
    aux_col_0: &[FieldElement<Goldilocks64Field>],
    zerofier_evals: &[Vec<FieldElement<Goldilocks64Field>>],
    boundary_evals: &[FieldElement<Goldilocks64Field>],
    gamma: &FieldElement<Goldilocks64Field>,
    transition_coefficients: &[FieldElement<Goldilocks64Field>],
    lde_step_size: usize,
    precompiled: Option<&FibRapConstraintState>,
) -> Result<Vec<FieldElement<Goldilocks64Field>>, MetalError> {
    let num_rows = main_col_0.len();
    assert_eq!(main_col_1.len(), num_rows, "main_col_1 length mismatch");
    assert_eq!(aux_col_0.len(), num_rows, "aux_col_0 length mismatch");
    assert_eq!(
        boundary_evals.len(),
        num_rows,
        "boundary_evals length mismatch"
    );
    assert!(
        zerofier_evals.len() >= 2,
        "need at least 2 zerofier evaluation vectors"
    );
    assert!(
        transition_coefficients.len() >= 2,
        "need at least 2 transition coefficients"
    );

    let col0_raw = to_raw_u64(main_col_0);
    let col1_raw = to_raw_u64(main_col_1);
    let aux0_raw = to_raw_u64(aux_col_0);
    let z0_raw = to_raw_u64(&zerofier_evals[0]);
    let z1_raw = to_raw_u64(&zerofier_evals[1]);
    let boundary_raw = to_raw_u64(boundary_evals);

    let params = build_fib_rap_params(
        lde_step_size,
        num_rows,
        zerofier_evals[0].len(),
        zerofier_evals[1].len(),
        gamma,
        transition_coefficients,
    );

    let mut owned_state = None;
    let (dyn_state, max_threads) = resolve_state(
        precompiled,
        &mut owned_state,
        "fibonacci_rap_constraint_eval",
    )?;

    let buf_col0 = dyn_state.alloc_buffer_with_data(&col0_raw)?;
    let buf_col1 = dyn_state.alloc_buffer_with_data(&col1_raw)?;
    let buf_aux0 = dyn_state.alloc_buffer_with_data(&aux0_raw)?;
    let buf_z0 = dyn_state.alloc_buffer_with_data(&z0_raw)?;
    let buf_z1 = dyn_state.alloc_buffer_with_data(&z1_raw)?;
    let buf_params = dyn_state.alloc_buffer_with_data(std::slice::from_ref(&params))?;
    let buf_boundary = dyn_state.alloc_buffer_with_data(&boundary_raw)?;
    let buf_output = dyn_state.alloc_buffer(num_rows * std::mem::size_of::<u64>())?;

    dyn_state.execute_compute(
        "fibonacci_rap_constraint_eval",
        &[
            &buf_col0,
            &buf_col1,
            &buf_aux0,
            &buf_z0,
            &buf_z1,
            &buf_params,
            &buf_boundary,
            &buf_output,
        ],
        num_rows as u64,
        max_threads,
    )?;

    let output_raw: Vec<u64> = unsafe { dyn_state.read_buffer(&buf_output, num_rows) };
    Ok(output_raw.into_iter().map(FieldElement::from).collect())
}

/// Like [`gpu_evaluate_fibonacci_rap_constraints`] but keeps the result on GPU as a Metal buffer.
#[cfg(all(target_os = "macos", feature = "metal"))]
#[allow(clippy::too_many_arguments)]
pub fn gpu_evaluate_fibonacci_rap_constraints_to_buffer(
    main_col_0: &[FieldElement<Goldilocks64Field>],
    main_col_1: &[FieldElement<Goldilocks64Field>],
    aux_col_0: &[FieldElement<Goldilocks64Field>],
    zerofier_evals: &[Vec<FieldElement<Goldilocks64Field>>],
    boundary_evals: &[FieldElement<Goldilocks64Field>],
    gamma: &FieldElement<Goldilocks64Field>,
    transition_coefficients: &[FieldElement<Goldilocks64Field>],
    lde_step_size: usize,
    precompiled: Option<&FibRapConstraintState>,
) -> Result<(metal::Buffer, usize), MetalError> {
    let num_rows = main_col_0.len();
    assert_eq!(main_col_1.len(), num_rows, "main_col_1 length mismatch");
    assert_eq!(aux_col_0.len(), num_rows, "aux_col_0 length mismatch");
    assert_eq!(
        boundary_evals.len(),
        num_rows,
        "boundary_evals length mismatch"
    );
    assert!(
        zerofier_evals.len() >= 2,
        "need at least 2 zerofier evaluation vectors"
    );
    assert!(
        transition_coefficients.len() >= 2,
        "need at least 2 transition coefficients"
    );

    let col0_raw = to_raw_u64(main_col_0);
    let col1_raw = to_raw_u64(main_col_1);
    let aux0_raw = to_raw_u64(aux_col_0);
    let z0_raw = to_raw_u64(&zerofier_evals[0]);
    let z1_raw = to_raw_u64(&zerofier_evals[1]);
    let boundary_raw = to_raw_u64(boundary_evals);

    let params = build_fib_rap_params(
        lde_step_size,
        num_rows,
        zerofier_evals[0].len(),
        zerofier_evals[1].len(),
        gamma,
        transition_coefficients,
    );

    let mut owned_state = None;
    let (dyn_state, max_threads) = resolve_state(
        precompiled,
        &mut owned_state,
        "fibonacci_rap_constraint_eval",
    )?;

    let buf_col0 = dyn_state.alloc_buffer_with_data(&col0_raw)?;
    let buf_col1 = dyn_state.alloc_buffer_with_data(&col1_raw)?;
    let buf_aux0 = dyn_state.alloc_buffer_with_data(&aux0_raw)?;
    let buf_z0 = dyn_state.alloc_buffer_with_data(&z0_raw)?;
    let buf_z1 = dyn_state.alloc_buffer_with_data(&z1_raw)?;
    let buf_params = dyn_state.alloc_buffer_with_data(std::slice::from_ref(&params))?;
    let buf_boundary = dyn_state.alloc_buffer_with_data(&boundary_raw)?;
    let buf_output = dyn_state.alloc_buffer(num_rows * std::mem::size_of::<u64>())?;

    dyn_state.execute_compute(
        "fibonacci_rap_constraint_eval",
        &[
            &buf_col0,
            &buf_col1,
            &buf_aux0,
            &buf_z0,
            &buf_z1,
            &buf_params,
            &buf_boundary,
            &buf_output,
        ],
        num_rows as u64,
        max_threads,
    )?;

    Ok((buf_output, num_rows))
}

/// Parameters for a single boundary constraint in the fused kernel.
/// Matches the Metal `FusedBoundaryParam` struct.
#[cfg(all(target_os = "macos", feature = "metal"))]
#[repr(C)]
#[derive(Copy, Clone)]
struct FusedBoundaryParam {
    g_pow_step: u64,
    value: u64,
    coefficient: u64,
    col: u32,
    _pad: u32,
}

/// Parameters for the fused transition + boundary kernel.
/// Matches the Metal `FusedParams` struct.
#[cfg(all(target_os = "macos", feature = "metal"))]
#[repr(C)]
#[derive(Copy, Clone)]
struct FusedParams {
    lde_step_size: u32,
    num_rows: u32,
    zerofier_0_len: u32,
    zerofier_1_len: u32,
    gamma: u64,
    transition_coeff_0: u64,
    transition_coeff_1: u64,
    num_boundary_constraints: u32,
    _pad2: u32,
}

/// Pre-compiled Metal state for the fused constraint evaluation kernel.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub struct FusedConstraintState {
    state: DynamicMetalState,
    max_threads: u64,
}

#[cfg(all(target_os = "macos", feature = "metal"))]
impl HasMetalState for FusedConstraintState {
    fn metal_state(&self) -> &DynamicMetalState {
        &self.state
    }
    fn metal_max_threads(&self) -> u64 {
        self.max_threads
    }
}

#[cfg(all(target_os = "macos", feature = "metal"))]
impl FusedConstraintState {
    pub fn new() -> Result<Self, MetalError> {
        let mut state = DynamicMetalState::new()?;
        state.load_library(FIBONACCI_RAP_SHADER)?;
        let max_threads = state.prepare_pipeline("fibonacci_rap_fused_eval")?;
        Ok(Self { state, max_threads })
    }

    pub fn from_device_and_queue(
        device: &metal::Device,
        queue: &metal::CommandQueue,
    ) -> Result<Self, MetalError> {
        let mut state = DynamicMetalState::from_device_and_queue(device, queue);
        state.load_library(FIBONACCI_RAP_SHADER)?;
        let max_threads = state.prepare_pipeline("fibonacci_rap_fused_eval")?;
        Ok(Self { state, max_threads })
    }
}

/// Fused boundary + transition constraint evaluation on GPU.
///
/// Combines both steps into a single kernel dispatch, eliminating separate boundary
/// kernel launch overhead, intermediate buffer, and redundant trace column reads.
/// Boundary zerofier inversions are computed inline via Fermat's little theorem.
#[cfg(all(target_os = "macos", feature = "metal"))]
#[allow(clippy::too_many_arguments)]
pub fn gpu_evaluate_fused_constraints(
    main_col_0_buf: &metal::Buffer,
    main_col_1_buf: &metal::Buffer,
    aux_col_0_buf: &metal::Buffer,
    num_rows: usize,
    zerofier_bufs: &[(metal::Buffer, usize)],
    gamma: &FieldElement<Goldilocks64Field>,
    transition_coefficients: &[FieldElement<Goldilocks64Field>],
    lde_step_size: usize,
    boundary_constraints: &[(usize, usize, FieldElement<Goldilocks64Field>)],
    boundary_coefficients: &[FieldElement<Goldilocks64Field>],
    lde_coset_points: &[FieldElement<Goldilocks64Field>],
    trace_primitive_root: &FieldElement<Goldilocks64Field>,
    precompiled: Option<&FusedConstraintState>,
    lde_coset_buf: Option<&metal::Buffer>,
) -> Result<(metal::Buffer, usize), MetalError> {
    assert!(zerofier_bufs.len() >= 2, "need at least 2 zerofier buffers");
    assert!(
        transition_coefficients.len() >= 2,
        "need at least 2 transition coefficients"
    );
    assert_eq!(
        boundary_constraints.len(),
        boundary_coefficients.len(),
        "boundary constraints and coefficients must have same length"
    );

    let mut owned_state = None;
    let (dyn_state, max_threads) =
        resolve_state(precompiled, &mut owned_state, "fibonacci_rap_fused_eval")?;

    // Use pre-existing LDE coset buffer when available, otherwise upload.
    let _owned_coset_buf;
    let buf_coset: &metal::Buffer = if let Some(buf) = lde_coset_buf {
        buf
    } else {
        let coset_raw = to_raw_u64(lde_coset_points);
        _owned_coset_buf = dyn_state.alloc_buffer_with_data(&coset_raw)?;
        &_owned_coset_buf
    };

    let bc_params: Vec<FusedBoundaryParam> = boundary_constraints
        .iter()
        .zip(boundary_coefficients.iter())
        .map(|(&(col, step, ref value), coeff)| FusedBoundaryParam {
            g_pow_step: canonical(&trace_primitive_root.pow(step as u64)),
            value: canonical(value),
            coefficient: canonical(coeff),
            col: col as u32,
            _pad: 0,
        })
        .collect();

    let params = FusedParams {
        lde_step_size: lde_step_size as u32,
        num_rows: num_rows as u32,
        zerofier_0_len: zerofier_bufs[0].1 as u32,
        zerofier_1_len: zerofier_bufs[1].1 as u32,
        gamma: canonical(gamma),
        transition_coeff_0: canonical(&transition_coefficients[0]),
        transition_coeff_1: canonical(&transition_coefficients[1]),
        num_boundary_constraints: bc_params.len() as u32,
        _pad2: 0,
    };

    let buf_params = dyn_state.alloc_buffer_with_data(std::slice::from_ref(&params))?;
    let buf_bc_params = dyn_state.alloc_buffer_with_data(&bc_params)?;
    let buf_output = dyn_state.alloc_buffer(num_rows * std::mem::size_of::<u64>())?;

    dyn_state.execute_compute(
        "fibonacci_rap_fused_eval",
        &[
            main_col_0_buf,
            main_col_1_buf,
            aux_col_0_buf,
            &zerofier_bufs[0].0,
            &zerofier_bufs[1].0,
            &buf_params,
            &buf_bc_params,
            buf_coset,
            &buf_output,
        ],
        num_rows as u64,
        max_threads,
    )?;

    Ok((buf_output, num_rows))
}

#[cfg(all(test, target_os = "macos", feature = "metal"))]
mod tests {
    use super::*;
    use lambdaworks_math::field::{
        element::FieldElement, fields::u64_goldilocks_field::Goldilocks64Field,
    };
    use stark_platinum_prover::constraints::evaluator::ConstraintEvaluator;
    use stark_platinum_prover::domain::Domain;
    use stark_platinum_prover::examples::fibonacci_rap::{
        fibonacci_rap_trace, FibonacciRAP, FibonacciRAPPublicInputs,
    };
    use stark_platinum_prover::proof::options::ProofOptions;
    use stark_platinum_prover::trace::LDETraceTable;
    use stark_platinum_prover::traits::AIR;

    use crate::metal::phases::rap::gpu_round_1;
    use crate::metal::state::StarkMetalState;
    use lambdaworks_crypto::fiat_shamir::default_transcript::DefaultTranscript;
    use lambdaworks_crypto::fiat_shamir::is_transcript::IsTranscript;

    type F = Goldilocks64Field;
    type FpE = FieldElement<F>;

    #[test]
    fn gpu_fibonacci_rap_constraints_match_cpu() {
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

        // Build coefficients the same way gpu_round_2 / the CPU prover does.
        let beta: FpE = transcript.sample_field_element();
        let num_boundary = air
            .boundary_constraints(&round_1.rap_challenges)
            .constraints
            .len();
        let num_transition = air.context().num_transition_constraints;

        let mut coefficients: Vec<FpE> =
            core::iter::successors(Some(FpE::one()), |x| Some(x * beta))
                .take(num_boundary + num_transition)
                .collect();
        let transition_coefficients: Vec<FpE> = coefficients.drain(..num_transition).collect();
        let boundary_coefficients = coefficients;

        let blowup_factor = air.blowup_factor() as usize;
        let lde_trace = LDETraceTable::from_columns(
            round_1.main_lde_evaluations.clone(),
            round_1.aux_lde_evaluations.clone(),
            air.step_size(),
            blowup_factor,
        );

        // --- CPU reference ---
        let evaluator = ConstraintEvaluator::<F, F, _>::new(&air, &round_1.rap_challenges);
        let cpu_result = evaluator
            .evaluate(
                &air,
                &lde_trace,
                &domain,
                &transition_coefficients,
                &boundary_coefficients,
                &round_1.rap_challenges,
            )
            .unwrap();

        // --- GPU path ---
        let num_lde_rows = lde_trace.num_rows();
        let main_col_0: Vec<FpE> = (0..num_lde_rows)
            .map(|r| *lde_trace.get_main(r, 0))
            .collect();
        let main_col_1: Vec<FpE> = (0..num_lde_rows)
            .map(|r| *lde_trace.get_main(r, 1))
            .collect();
        let aux_col_0: Vec<FpE> = (0..num_lde_rows)
            .map(|r| *lde_trace.get_aux(r, 0))
            .collect();

        let zerofier_evals = air.transition_zerofier_evaluations(&domain);

        // Pre-compute boundary evaluations on CPU.
        let boundary_constraints = air.boundary_constraints(&round_1.rap_challenges);
        use std::collections::HashMap;
        let mut zerofier_cache: HashMap<usize, Vec<FpE>> = HashMap::new();
        for bc in &boundary_constraints.constraints {
            zerofier_cache.entry(bc.step).or_insert_with(|| {
                let point = domain.trace_primitive_root.pow(bc.step as u64);
                let mut evals: Vec<FpE> = domain
                    .lde_roots_of_unity_coset
                    .iter()
                    .map(|v| v - point)
                    .collect();
                FpE::inplace_batch_inverse(&mut evals).unwrap();
                evals
            });
        }
        let boundary_zerofiers_refs: Vec<&Vec<FpE>> = boundary_constraints
            .constraints
            .iter()
            .map(|bc| zerofier_cache.get(&bc.step).unwrap())
            .collect();
        let boundary_poly_evals: Vec<Vec<FpE>> = boundary_constraints
            .constraints
            .iter()
            .map(|constraint| {
                if constraint.is_aux {
                    (0..num_lde_rows)
                        .map(|row| lde_trace.get_aux(row, constraint.col) - constraint.value)
                        .collect()
                } else {
                    (0..num_lde_rows)
                        .map(|row| lde_trace.get_main(row, constraint.col) - constraint.value)
                        .collect()
                }
            })
            .collect();
        let boundary_evals_vec: Vec<FpE> = (0..num_lde_rows)
            .map(|i| {
                boundary_zerofiers_refs
                    .iter()
                    .zip(boundary_poly_evals.iter())
                    .zip(boundary_coefficients.iter())
                    .fold(FpE::zero(), |acc, ((z, bp), beta)| {
                        acc + z[i] * beta * bp[i]
                    })
            })
            .collect();

        let lde_step_size = air.step_size() * blowup_factor;

        let gpu_result = gpu_evaluate_fibonacci_rap_constraints(
            &main_col_0,
            &main_col_1,
            &aux_col_0,
            &zerofier_evals,
            &boundary_evals_vec,
            &round_1.rap_challenges[0],
            &transition_coefficients,
            lde_step_size,
            None,
        )
        .expect("GPU constraint evaluation failed");

        assert_eq!(
            gpu_result.len(),
            cpu_result.len(),
            "GPU and CPU result lengths differ"
        );
        for (i, (gpu_val, cpu_val)) in gpu_result.iter().zip(cpu_result.iter()).enumerate() {
            assert_eq!(
                gpu_val, cpu_val,
                "Mismatch at LDE domain point {i}: GPU={gpu_val:?} CPU={cpu_val:?}"
            );
        }
    }
}
