//! GPU constraint evaluation for the Fibonacci RAP AIR.
//!
//! Uses a Metal shader to evaluate transition constraints in parallel across
//! all LDE domain points. Boundary constraints are still evaluated on CPU
//! and passed to the shader as pre-computed values.

#[cfg(all(target_os = "macos", feature = "metal"))]
use lambdaworks_gpu::metal::abstractions::{errors::MetalError, state::DynamicMetalState};
#[cfg(all(target_os = "macos", feature = "metal"))]
use lambdaworks_math::field::{
    element::FieldElement, fields::u64_goldilocks_field::Goldilocks64Field, traits::IsPrimeField,
};

/// Embedded Metal shader source for the Fibonacci RAP constraint kernel.
#[cfg(all(target_os = "macos", feature = "metal"))]
const FIBONACCI_RAP_SHADER: &str = include_str!("shaders/fibonacci_rap_constraints.metal");

/// Parameters struct matching the Metal shader's `FibRapParams`.
///
/// Must be `#[repr(C)]` to match Metal's memory layout exactly.
/// Fields must appear in the same order as in the shader struct.
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

/// Pre-compiled Metal state for Fibonacci RAP constraint evaluation.
///
/// Create once and reuse across multiple calls to avoid shader recompilation.
/// The compiled pipeline and max thread count are cached after construction.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub struct FibRapConstraintState {
    state: DynamicMetalState,
    max_threads: u64,
}

#[cfg(all(target_os = "macos", feature = "metal"))]
impl FibRapConstraintState {
    /// Compile the Fibonacci RAP constraint shader and prepare the pipeline.
    ///
    /// # Errors
    ///
    /// Returns `MetalError` if the shader fails to compile or the pipeline cannot be created.
    pub fn new() -> Result<Self, MetalError> {
        let mut state = DynamicMetalState::new()?;
        state.load_library(FIBONACCI_RAP_SHADER)?;
        let max_threads = state.prepare_pipeline("fibonacci_rap_constraint_eval")?;
        Ok(Self { state, max_threads })
    }
}

/// Evaluates Fibonacci RAP constraints on the GPU using a Metal compute shader.
///
/// This replaces the CPU `ConstraintEvaluator::evaluate()` for the specific
/// case of Fibonacci RAP on the Goldilocks field. The boundary constraint
/// evaluations must be pre-computed on the CPU and passed in via `boundary_evals`.
///
/// If `precompiled` is `Some`, uses the pre-compiled shader state to avoid
/// recompiling the Metal shader on each call. Otherwise, compiles the shader
/// from scratch (slower, but convenient for one-off use and tests).
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

    // Extract canonical u64 values for GPU buffers.
    let col0_raw: Vec<u64> = main_col_0
        .iter()
        .map(|fe| Goldilocks64Field::canonical(fe.value()))
        .collect();
    let col1_raw: Vec<u64> = main_col_1
        .iter()
        .map(|fe| Goldilocks64Field::canonical(fe.value()))
        .collect();
    let aux0_raw: Vec<u64> = aux_col_0
        .iter()
        .map(|fe| Goldilocks64Field::canonical(fe.value()))
        .collect();
    let z0_raw: Vec<u64> = zerofier_evals[0]
        .iter()
        .map(|fe| Goldilocks64Field::canonical(fe.value()))
        .collect();
    let z1_raw: Vec<u64> = zerofier_evals[1]
        .iter()
        .map(|fe| Goldilocks64Field::canonical(fe.value()))
        .collect();
    let boundary_raw: Vec<u64> = boundary_evals
        .iter()
        .map(|fe| Goldilocks64Field::canonical(fe.value()))
        .collect();

    let params = FibRapParams {
        lde_step_size: lde_step_size as u32,
        num_rows: num_rows as u32,
        zerofier_0_len: zerofier_evals[0].len() as u32,
        zerofier_1_len: zerofier_evals[1].len() as u32,
        gamma: Goldilocks64Field::canonical(gamma.value()),
        transition_coeff_0: Goldilocks64Field::canonical(transition_coefficients[0].value()),
        transition_coeff_1: Goldilocks64Field::canonical(transition_coefficients[1].value()),
    };

    // Use pre-compiled state or create a fresh one.
    let mut owned_state;
    let (dyn_state, max_threads) = match precompiled {
        Some(pre) => (&pre.state, pre.max_threads),
        None => {
            owned_state = DynamicMetalState::new()?;
            owned_state.load_library(FIBONACCI_RAP_SHADER)?;
            let mt = owned_state.prepare_pipeline("fibonacci_rap_constraint_eval")?;
            (&owned_state, mt)
        }
    };

    // Allocate GPU-accessible buffers.
    let buf_col0 = dyn_state.alloc_buffer_with_data(&col0_raw)?;
    let buf_col1 = dyn_state.alloc_buffer_with_data(&col1_raw)?;
    let buf_aux0 = dyn_state.alloc_buffer_with_data(&aux0_raw)?;
    let buf_z0 = dyn_state.alloc_buffer_with_data(&z0_raw)?;
    let buf_z1 = dyn_state.alloc_buffer_with_data(&z1_raw)?;
    let buf_params = dyn_state.alloc_buffer_with_data(std::slice::from_ref(&params))?;
    let buf_boundary = dyn_state.alloc_buffer_with_data(&boundary_raw)?;
    let buf_output = dyn_state.alloc_buffer(num_rows * std::mem::size_of::<u64>())?;

    // Dispatch the compute kernel over all LDE domain points.
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

    // Read results back from the GPU and wrap in FieldElements.
    let output_raw: Vec<u64> = unsafe { dyn_state.read_buffer(&buf_output, num_rows) };
    let result: Vec<FieldElement<Goldilocks64Field>> =
        output_raw.into_iter().map(FieldElement::from).collect();

    Ok(result)
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

    /// Differential test: compare GPU constraint evaluations against the CPU evaluator.
    ///
    /// This test:
    /// 1. Sets up a Fibonacci RAP trace
    /// 2. Runs GPU round 1 to obtain LDE trace and RAP challenges
    /// 3. Pre-computes boundary evaluations on CPU
    /// 4. Invokes the GPU constraint evaluator
    /// 5. Runs the CPU constraint evaluator
    /// 6. Compares results element-by-element
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

        // Run GPU round 1 to get the LDE trace and RAP challenges.
        let round_1 = gpu_round_1(&air, &mut trace, &domain, &mut transcript, &state).unwrap();

        // Build coefficients the same way gpu_round_2 / the CPU prover does.
        let beta: FpE = transcript.sample_field_element();
        let num_boundary = air
            .boundary_constraints(&round_1.rap_challenges)
            .constraints
            .len();
        let num_transition = air.context().num_transition_constraints;

        let mut coefficients: Vec<FpE> =
            core::iter::successors(Some(FpE::one()), |x| Some(x * &beta))
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

        // --- CPU reference path ---
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
        // Extract LDE trace columns for the GPU. The LDETraceTable stores data column-major.
        let num_lde_rows = lde_trace.num_rows();
        let main_col_0: Vec<FpE> = (0..num_lde_rows)
            .map(|r| lde_trace.get_main(r, 0).clone())
            .collect();
        let main_col_1: Vec<FpE> = (0..num_lde_rows)
            .map(|r| lde_trace.get_main(r, 1).clone())
            .collect();
        let aux_col_0: Vec<FpE> = (0..num_lde_rows)
            .map(|r| lde_trace.get_aux(r, 0).clone())
            .collect();

        // Pre-compute transition zerofier evaluations (same as CPU does internally).
        let zerofier_evals = air.transition_zerofier_evaluations(&domain);

        // Pre-compute boundary evaluations on CPU (the GPU shader adds them after transition acc).
        // Replicate the evaluator's boundary logic.
        let boundary_constraints = air.boundary_constraints(&round_1.rap_challenges);
        use std::collections::HashMap;
        let mut zerofier_cache: HashMap<usize, Vec<FpE>> = HashMap::new();
        for bc in &boundary_constraints.constraints {
            zerofier_cache.entry(bc.step).or_insert_with(|| {
                let point = domain.trace_primitive_root.pow(bc.step as u64);
                let mut evals: Vec<FpE> = domain
                    .lde_roots_of_unity_coset
                    .iter()
                    .map(|v| v - &point)
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
                        .map(|row| lde_trace.get_aux(row, constraint.col) - &constraint.value)
                        .collect()
                } else {
                    (0..num_lde_rows)
                        .map(|row| lde_trace.get_main(row, constraint.col) - &constraint.value)
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
                        acc + &z[i] * beta * &bp[i]
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

        // Compare results element-by-element.
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
