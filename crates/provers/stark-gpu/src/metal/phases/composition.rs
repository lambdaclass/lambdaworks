//! GPU Phase 2: Composition Polynomial.
//!
//! This module mirrors `round_2_compute_composition_polynomial` from the CPU
//! STARK prover but uses Metal GPU FFT for the LDE evaluation of composition
//! polynomial parts.
//!
//! Two variants are provided:
//! - [`gpu_round_2`]: Generic over field and AIR, uses CPU constraint evaluation.
//! - [`gpu_round_2_goldilocks`]: Concrete for Goldilocks field, uses GPU constraint evaluation.

use lambdaworks_math::fft::cpu::bit_reversing::in_place_bit_reverse_permute;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::{IsFFTField, IsField, IsSubFieldOf};
use lambdaworks_math::polynomial::Polynomial;
use lambdaworks_math::traits::AsBytes;

use lambdaworks_crypto::fiat_shamir::is_transcript::IsStarkTranscript;

use stark_platinum_prover::config::{BatchedMerkleTree, Commitment};
use stark_platinum_prover::constraints::evaluator::ConstraintEvaluator;
use stark_platinum_prover::domain::Domain;
use stark_platinum_prover::prover::ProvingError;
use stark_platinum_prover::trace::LDETraceTable;
use stark_platinum_prover::traits::AIR;

#[cfg(all(target_os = "macos", feature = "metal"))]
use crate::metal::fft::gpu_evaluate_offset_fft;
use crate::metal::merkle::cpu_batch_commit;
use crate::metal::phases::rap::GpuRound1Result;
#[cfg(all(target_os = "macos", feature = "metal"))]
use crate::metal::state::StarkMetalState;

#[cfg(all(target_os = "macos", feature = "metal"))]
use lambdaworks_math::field::fields::u64_goldilocks_field::Goldilocks64Field;
#[cfg(all(target_os = "macos", feature = "metal"))]
use crate::metal::merkle::{gpu_batch_commit_paired_goldilocks, GpuKeccakMerkleState};
#[cfg(all(target_os = "macos", feature = "metal"))]
use crate::metal::constraint_eval::{gpu_evaluate_fibonacci_rap_constraints, FibRapConstraintState};

/// Result of GPU Phase 2 (composition polynomial round).
///
/// This is the GPU equivalent of `Round2<FieldExtension>` from the CPU prover.
/// Contains the composition polynomial broken into parts, their LDE evaluations,
/// and the Merkle tree commitment.
pub struct GpuRound2Result<F: IsField>
where
    FieldElement<F>: AsBytes,
{
    /// The composition polynomial broken into parts.
    pub composition_poly_parts: Vec<Polynomial<FieldElement<F>>>,
    /// LDE evaluations of each composition poly part (column-major).
    pub lde_composition_poly_evaluations: Vec<Vec<FieldElement<F>>>,
    /// Merkle tree for the composition polynomial commitment.
    pub composition_poly_merkle_tree: BatchedMerkleTree<F>,
    /// Commitment root.
    pub composition_poly_root: Commitment,
}

/// Executes GPU Phase 2 of the STARK prover: composition polynomial computation.
///
/// This mirrors `round_2_compute_composition_polynomial` from the CPU prover:
///
/// 1. Sample beta from the transcript and compute transition/boundary coefficients
/// 2. Build an `LDETraceTable` from the round 1 LDE evaluations
/// 3. Evaluate constraints on CPU using `ConstraintEvaluator`
/// 4. Interpolate constraint evaluations to get the composition polynomial (CPU)
/// 5. Break the composition polynomial into parts
/// 6. Evaluate each part on the LDE domain using GPU FFT
/// 7. Commit the composition poly parts with the special layout (bit-reverse + pair rows)
///
/// # Type Parameters
///
/// - `F`: The base field (must equal the extension field for our GPU prover)
/// - `A`: An AIR whose `Field` and `FieldExtension` are both `F`
///
/// # Errors
///
/// Returns `ProvingError` if constraint evaluation, FFT, or commitment fails.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn gpu_round_2<F, A>(
    air: &A,
    domain: &Domain<F>,
    round_1_result: &GpuRound1Result<F>,
    transcript: &mut impl IsStarkTranscript<F, F>,
    state: &StarkMetalState,
) -> Result<GpuRound2Result<F>, ProvingError>
where
    F: IsFFTField + IsSubFieldOf<F> + Send + Sync,
    F::BaseType: Copy,
    FieldElement<F>: AsBytes + Sync + Send,
    A: AIR<Field = F, FieldExtension = F>,
{
    // Step 1: Sample beta and compute transition/boundary coefficients.
    // This matches the CPU prover's coefficient generation exactly.
    let beta = transcript.sample_field_element();
    let num_boundary = air
        .boundary_constraints(&round_1_result.rap_challenges)
        .constraints
        .len();
    let num_transition = air.context().num_transition_constraints;

    let mut coefficients: Vec<_> =
        core::iter::successors(Some(FieldElement::one()), |x| Some(x * &beta))
            .take(num_boundary + num_transition)
            .collect();
    let transition_coefficients: Vec<_> = coefficients.drain(..num_transition).collect();
    let boundary_coefficients = coefficients;

    // Step 2: Build LDETraceTable from GPU round 1 result.
    let blowup_factor = air.blowup_factor() as usize;
    let lde_trace = LDETraceTable::from_columns(
        round_1_result.main_lde_evaluations.clone(),
        round_1_result.aux_lde_evaluations.clone(),
        air.step_size(),
        blowup_factor,
    );

    // Step 3: Evaluate constraints on CPU.
    let evaluator =
        ConstraintEvaluator::<F, F, A::PublicInputs>::new(air, &round_1_result.rap_challenges);
    let constraint_evaluations = evaluator.evaluate(
        air,
        &lde_trace,
        domain,
        &transition_coefficients,
        &boundary_coefficients,
        &round_1_result.rap_challenges,
    )?;

    // Step 4: Interpolate constraint evaluations to get composition polynomial (CPU).
    let coset_offset = air.coset_offset();
    let composition_poly =
        Polynomial::interpolate_offset_fft(&constraint_evaluations, &coset_offset)?;

    // Step 5: Break into parts.
    let number_of_parts = air.composition_poly_degree_bound() / air.trace_length();
    if number_of_parts == 0 {
        return Err(ProvingError::WrongParameter(
            "composition_poly_degree_bound must be >= trace_length".to_string(),
        ));
    }
    let composition_poly_parts = composition_poly.break_in_parts(number_of_parts);

    // Step 6: Evaluate each part on the LDE domain using GPU FFT.
    let lde_evaluations: Vec<Vec<FieldElement<F>>> = composition_poly_parts
        .iter()
        .map(|part| {
            gpu_evaluate_offset_fft::<F>(
                part.coefficients(),
                blowup_factor,
                &coset_offset,
                state.inner(),
            )
            .map_err(|e| ProvingError::FieldOperationError(format!("GPU LDE FFT error: {e}")))
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Step 7: Commit composition poly parts with the special layout.
    // The CPU prover transposes, bit-reverse permutes, then pairs consecutive rows.
    let lde_len = lde_evaluations[0].len();

    // Transpose: column-major to row-major
    let mut rows: Vec<Vec<FieldElement<F>>> = (0..lde_len)
        .map(|i| lde_evaluations.iter().map(|col| col[i].clone()).collect())
        .collect();

    // Bit-reverse permute
    in_place_bit_reverse_permute(&mut rows);

    // Pair consecutive rows (merge row 2i and row 2i+1 into one row)
    let mut merged_rows = Vec::with_capacity(lde_len / 2);
    let mut iter = rows.into_iter();
    while let (Some(mut chunk0), Some(chunk1)) = (iter.next(), iter.next()) {
        chunk0.extend(chunk1);
        merged_rows.push(chunk0);
    }

    // Batch commit
    let (tree, root) = cpu_batch_commit(&merged_rows).ok_or(ProvingError::EmptyCommitment)?;

    Ok(GpuRound2Result {
        composition_poly_parts,
        lde_composition_poly_evaluations: lde_evaluations,
        composition_poly_merkle_tree: tree,
        composition_poly_root: root,
    })
}

/// GPU-optimized Phase 2 for Goldilocks field with GPU constraint evaluation.
///
/// This is a concrete version of [`gpu_round_2`] that uses the Metal GPU shader
/// for constraint evaluation instead of the CPU `ConstraintEvaluator`. Boundary
/// evaluations are pre-computed on CPU and passed to the GPU shader.
///
/// If `precompiled_constraint` is `Some`, uses the pre-compiled shader state.
#[cfg(all(target_os = "macos", feature = "metal"))]
#[allow(clippy::too_many_arguments)]
pub fn gpu_round_2_goldilocks<A>(
    air: &A,
    domain: &Domain<Goldilocks64Field>,
    round_1_result: &GpuRound1Result<Goldilocks64Field>,
    transcript: &mut impl IsStarkTranscript<Goldilocks64Field, Goldilocks64Field>,
    state: &StarkMetalState,
    precompiled_constraint: Option<&FibRapConstraintState>,
) -> Result<GpuRound2Result<Goldilocks64Field>, ProvingError>
where
    A: AIR<Field = Goldilocks64Field, FieldExtension = Goldilocks64Field>,
{
    type F = Goldilocks64Field;
    type FpE = FieldElement<F>;

    // Step 1: Sample beta and compute transition/boundary coefficients.
    let beta: FpE = transcript.sample_field_element();
    let num_boundary = air
        .boundary_constraints(&round_1_result.rap_challenges)
        .constraints
        .len();
    let num_transition = air.context().num_transition_constraints;

    let mut coefficients: Vec<FpE> =
        core::iter::successors(Some(FpE::one()), |x| Some(x * &beta))
            .take(num_boundary + num_transition)
            .collect();
    let transition_coefficients: Vec<FpE> = coefficients.drain(..num_transition).collect();
    let boundary_coefficients = coefficients;

    // Step 2: Build LDETraceTable from GPU round 1 result.
    let blowup_factor = air.blowup_factor() as usize;
    let lde_trace = LDETraceTable::from_columns(
        round_1_result.main_lde_evaluations.clone(),
        round_1_result.aux_lde_evaluations.clone(),
        air.step_size(),
        blowup_factor,
    );
    let num_lde_rows = lde_trace.num_rows();

    // Step 3a: Pre-compute boundary evaluations on CPU.
    let boundary_constraints = air.boundary_constraints(&round_1_result.rap_challenges);
    let mut zerofier_cache: std::collections::HashMap<usize, Vec<FpE>> =
        std::collections::HashMap::new();
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
    let boundary_evals: Vec<FpE> = (0..num_lde_rows)
        .map(|i| {
            boundary_zerofiers_refs
                .iter()
                .zip(boundary_poly_evals.iter())
                .zip(boundary_coefficients.iter())
                .fold(FpE::zero(), |acc, ((z, bp), coeff)| {
                    acc + &z[i] * coeff * &bp[i]
                })
        })
        .collect();

    // Step 3b: Extract LDE trace columns for GPU.
    let main_col_0: Vec<FpE> = (0..num_lde_rows)
        .map(|r| lde_trace.get_main(r, 0).clone())
        .collect();
    let main_col_1: Vec<FpE> = (0..num_lde_rows)
        .map(|r| lde_trace.get_main(r, 1).clone())
        .collect();
    let aux_col_0: Vec<FpE> = (0..num_lde_rows)
        .map(|r| lde_trace.get_aux(r, 0).clone())
        .collect();

    // Step 3c: Evaluate constraints on GPU.
    let zerofier_evals = air.transition_zerofier_evaluations(domain);
    let lde_step_size = air.step_size() * blowup_factor;
    let constraint_evaluations = gpu_evaluate_fibonacci_rap_constraints(
        &main_col_0,
        &main_col_1,
        &aux_col_0,
        &zerofier_evals,
        &boundary_evals,
        &round_1_result.rap_challenges[0],
        &transition_coefficients,
        lde_step_size,
        precompiled_constraint,
    )
    .map_err(|e| ProvingError::FieldOperationError(format!("GPU constraint eval error: {e}")))?;

    // Step 4: Interpolate constraint evaluations to get composition polynomial.
    let coset_offset = air.coset_offset();
    let composition_poly =
        Polynomial::interpolate_offset_fft(&constraint_evaluations, &coset_offset)?;

    // Step 5: Break into parts.
    let number_of_parts = air.composition_poly_degree_bound() / air.trace_length();
    if number_of_parts == 0 {
        return Err(ProvingError::WrongParameter(
            "composition_poly_degree_bound must be >= trace_length".to_string(),
        ));
    }
    let composition_poly_parts = composition_poly.break_in_parts(number_of_parts);

    // Step 6: Evaluate each part on the LDE domain using GPU FFT.
    let lde_evaluations: Vec<Vec<FpE>> = composition_poly_parts
        .iter()
        .map(|part| {
            gpu_evaluate_offset_fft::<F>(
                part.coefficients(),
                blowup_factor,
                &coset_offset,
                state.inner(),
            )
            .map_err(|e| ProvingError::FieldOperationError(format!("GPU LDE FFT error: {e}")))
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Step 7: Commit composition poly parts with the special layout.
    let lde_len = lde_evaluations[0].len();

    let mut rows: Vec<Vec<FpE>> = (0..lde_len)
        .map(|i| lde_evaluations.iter().map(|col| col[i].clone()).collect())
        .collect();

    in_place_bit_reverse_permute(&mut rows);

    let mut merged_rows = Vec::with_capacity(lde_len / 2);
    let mut iter = rows.into_iter();
    while let (Some(mut chunk0), Some(chunk1)) = (iter.next(), iter.next()) {
        chunk0.extend(chunk1);
        merged_rows.push(chunk0);
    }

    let (tree, root) = cpu_batch_commit(&merged_rows).ok_or(ProvingError::EmptyCommitment)?;

    Ok(GpuRound2Result {
        composition_poly_parts,
        lde_composition_poly_evaluations: lde_evaluations,
        composition_poly_merkle_tree: tree,
        composition_poly_root: root,
    })
}

/// GPU-optimized Phase 2 for Goldilocks field with GPU constraint eval AND GPU Merkle commit.
///
/// This extends [`gpu_round_2_goldilocks`] by also using the GPU Keccak256 shader
/// for the composition polynomial Merkle tree commit (paired-row layout).
#[cfg(all(target_os = "macos", feature = "metal"))]
#[allow(clippy::too_many_arguments)]
pub fn gpu_round_2_goldilocks_merkle<A>(
    air: &A,
    domain: &Domain<Goldilocks64Field>,
    round_1_result: &GpuRound1Result<Goldilocks64Field>,
    transcript: &mut impl IsStarkTranscript<Goldilocks64Field, Goldilocks64Field>,
    state: &StarkMetalState,
    precompiled_constraint: Option<&FibRapConstraintState>,
    keccak_state: &GpuKeccakMerkleState,
) -> Result<GpuRound2Result<Goldilocks64Field>, ProvingError>
where
    A: AIR<Field = Goldilocks64Field, FieldExtension = Goldilocks64Field>,
{
    type F = Goldilocks64Field;
    type FpE = FieldElement<F>;

    // Steps 1-6 are identical to gpu_round_2_goldilocks
    let beta: FpE = transcript.sample_field_element();
    let num_boundary = air
        .boundary_constraints(&round_1_result.rap_challenges)
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
    let lde_trace = stark_platinum_prover::trace::LDETraceTable::from_columns(
        round_1_result.main_lde_evaluations.clone(),
        round_1_result.aux_lde_evaluations.clone(),
        air.step_size(),
        blowup_factor,
    );
    let num_lde_rows = lde_trace.num_rows();

    // Boundary evaluations on CPU
    let boundary_constraints = air.boundary_constraints(&round_1_result.rap_challenges);
    let mut zerofier_cache: std::collections::HashMap<usize, Vec<FpE>> =
        std::collections::HashMap::new();
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
    let boundary_evals: Vec<FpE> = (0..num_lde_rows)
        .map(|i| {
            boundary_zerofiers_refs
                .iter()
                .zip(boundary_poly_evals.iter())
                .zip(boundary_coefficients.iter())
                .fold(FpE::zero(), |acc, ((z, bp), coeff)| {
                    acc + &z[i] * coeff * &bp[i]
                })
        })
        .collect();

    let main_col_0: Vec<FpE> = (0..num_lde_rows)
        .map(|r| lde_trace.get_main(r, 0).clone())
        .collect();
    let main_col_1: Vec<FpE> = (0..num_lde_rows)
        .map(|r| lde_trace.get_main(r, 1).clone())
        .collect();
    let aux_col_0: Vec<FpE> = (0..num_lde_rows)
        .map(|r| lde_trace.get_aux(r, 0).clone())
        .collect();

    let zerofier_evals = air.transition_zerofier_evaluations(domain);
    let lde_step_size = air.step_size() * blowup_factor;
    let constraint_evaluations = gpu_evaluate_fibonacci_rap_constraints(
        &main_col_0,
        &main_col_1,
        &aux_col_0,
        &zerofier_evals,
        &boundary_evals,
        &round_1_result.rap_challenges[0],
        &transition_coefficients,
        lde_step_size,
        precompiled_constraint,
    )
    .map_err(|e| ProvingError::FieldOperationError(format!("GPU constraint eval error: {e}")))?;

    let coset_offset = air.coset_offset();
    let composition_poly =
        Polynomial::interpolate_offset_fft(&constraint_evaluations, &coset_offset)?;

    let number_of_parts = air.composition_poly_degree_bound() / air.trace_length();
    if number_of_parts == 0 {
        return Err(ProvingError::WrongParameter(
            "composition_poly_degree_bound must be >= trace_length".to_string(),
        ));
    }
    let composition_poly_parts = composition_poly.break_in_parts(number_of_parts);

    let lde_evaluations: Vec<Vec<FpE>> = composition_poly_parts
        .iter()
        .map(|part| {
            gpu_evaluate_offset_fft::<F>(
                part.coefficients(),
                blowup_factor,
                &coset_offset,
                state.inner(),
            )
            .map_err(|e| ProvingError::FieldOperationError(format!("GPU LDE FFT error: {e}")))
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Step 7: GPU Merkle commit with paired-row layout
    let (tree, root) = gpu_batch_commit_paired_goldilocks(&lde_evaluations, keccak_state)
        .ok_or(ProvingError::EmptyCommitment)?;

    Ok(GpuRound2Result {
        composition_poly_parts,
        lde_composition_poly_evaluations: lde_evaluations,
        composition_poly_merkle_tree: tree,
        composition_poly_root: root,
    })
}

#[cfg(all(test, target_os = "macos", feature = "metal"))]
mod tests {
    use super::*;
    use crate::metal::phases::rap::gpu_round_1;
    use lambdaworks_crypto::fiat_shamir::default_transcript::DefaultTranscript;
    use lambdaworks_crypto::fiat_shamir::is_transcript::IsTranscript;
    use lambdaworks_math::field::fields::u64_goldilocks_field::Goldilocks64Field;
    use stark_platinum_prover::examples::fibonacci_rap::{
        fibonacci_rap_trace, FibonacciRAP, FibonacciRAPPublicInputs,
    };
    use stark_platinum_prover::proof::options::ProofOptions;

    type F = Goldilocks64Field;
    type FpE = FieldElement<F>;

    #[test]
    fn gpu_round_2_fibonacci_rap_goldilocks() {
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

        // Composition poly should have the expected number of parts
        let expected_parts = air.composition_poly_degree_bound() / air.trace_length();
        assert_eq!(round_2.composition_poly_parts.len(), expected_parts);

        // Merkle root should be non-zero
        assert_ne!(round_2.composition_poly_root, [0u8; 32]);

        // Each LDE evaluation length = part coefficient count * blowup_factor
        let blowup = proof_options.blowup_factor as usize;
        for (part, eval) in round_2
            .composition_poly_parts
            .iter()
            .zip(&round_2.lde_composition_poly_evaluations)
        {
            assert_eq!(eval.len(), part.coeff_len() * blowup);
        }
    }

    /// Differential test: compare GPU round 2 against CPU round 2 logic.
    #[test]
    fn gpu_round_2_matches_cpu_composition_poly() {
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

        // --- GPU path ---
        let mut gpu_transcript = DefaultTranscript::<F>::new(&[]);
        let gpu_round_1_result =
            gpu_round_1(&air, &mut trace, &domain, &mut gpu_transcript, &state).unwrap();
        let gpu_round_2_result = gpu_round_2(
            &air,
            &domain,
            &gpu_round_1_result,
            &mut gpu_transcript,
            &state,
        )
        .unwrap();

        // --- CPU path (reproduce round 2 manually) ---
        // Use a fresh transcript and re-run round 1 to get the same state.
        let mut cpu_transcript = DefaultTranscript::<F>::new(&[]);
        let cpu_round_1_result =
            gpu_round_1(&air, &mut trace, &domain, &mut cpu_transcript, &state).unwrap();

        // Sample beta the same way
        let beta: FieldElement<F> = cpu_transcript.sample_field_element();
        let num_boundary = air
            .boundary_constraints(&cpu_round_1_result.rap_challenges)
            .constraints
            .len();
        let num_transition = air.context().num_transition_constraints;

        let mut coefficients: Vec<_> =
            core::iter::successors(Some(FieldElement::one()), |x| Some(x * &beta))
                .take(num_boundary + num_transition)
                .collect();
        let transition_coefficients: Vec<_> = coefficients.drain(..num_transition).collect();
        let boundary_coefficients = coefficients;

        // Build LDE trace
        let blowup_factor = air.blowup_factor() as usize;
        let lde_trace = LDETraceTable::from_columns(
            cpu_round_1_result.main_lde_evaluations.clone(),
            cpu_round_1_result.aux_lde_evaluations.clone(),
            air.step_size(),
            blowup_factor,
        );

        // Evaluate constraints on CPU
        let evaluator =
            ConstraintEvaluator::<F, F, _>::new(&air, &cpu_round_1_result.rap_challenges);
        let constraint_evaluations = evaluator
            .evaluate(
                &air,
                &lde_trace,
                &domain,
                &transition_coefficients,
                &boundary_coefficients,
                &cpu_round_1_result.rap_challenges,
            )
            .unwrap();

        // Interpolate and break into parts (CPU)
        let coset_offset = air.coset_offset();
        let composition_poly =
            Polynomial::interpolate_offset_fft(&constraint_evaluations, &coset_offset).unwrap();
        let number_of_parts = air.composition_poly_degree_bound() / air.trace_length();
        let cpu_composition_poly_parts = composition_poly.break_in_parts(number_of_parts);

        // Compare composition polynomial parts
        assert_eq!(
            gpu_round_2_result.composition_poly_parts.len(),
            cpu_composition_poly_parts.len(),
            "Number of composition poly parts mismatch"
        );
        for (i, (gpu_part, cpu_part)) in gpu_round_2_result
            .composition_poly_parts
            .iter()
            .zip(&cpu_composition_poly_parts)
            .enumerate()
        {
            let gpu_coeffs = gpu_part.coefficients();
            let cpu_coeffs = cpu_part.coefficients();
            assert_eq!(
                gpu_coeffs.len(),
                cpu_coeffs.len(),
                "Part {i}: coefficient count mismatch"
            );
            for (j, (g, c)) in gpu_coeffs.iter().zip(cpu_coeffs).enumerate() {
                assert_eq!(g, c, "Part {i}, coeff {j}: value mismatch");
            }
        }

        // Compare LDE evaluations of the parts
        for (i, (gpu_eval, cpu_part)) in gpu_round_2_result
            .lde_composition_poly_evaluations
            .iter()
            .zip(&cpu_composition_poly_parts)
            .enumerate()
        {
            let cpu_eval =
                Polynomial::evaluate_offset_fft::<F>(cpu_part, blowup_factor, None, &coset_offset)
                    .unwrap();
            assert_eq!(
                gpu_eval.len(),
                cpu_eval.len(),
                "Part {i}: LDE eval count mismatch"
            );
            for (j, (g, c)) in gpu_eval.iter().zip(&cpu_eval).enumerate() {
                assert_eq!(g, c, "Part {i}, eval {j}: LDE value mismatch");
            }
        }
    }
}
