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
use crate::metal::fft::{
    gpu_break_in_parts_buffer_to_buffers, gpu_evaluate_offset_fft,
    gpu_evaluate_offset_fft_buffer_to_buffers_batch,
};
use crate::metal::merkle::cpu_batch_commit;
use crate::metal::phases::rap::GpuRound1Result;
#[cfg(all(target_os = "macos", feature = "metal"))]
use crate::metal::state::StarkMetalState;
#[cfg(all(target_os = "macos", feature = "metal"))]
use std::collections::HashMap;

#[cfg(all(target_os = "macos", feature = "metal"))]
use crate::metal::constraint_eval::{
    gpu_evaluate_fibonacci_rap_constraints, gpu_evaluate_fibonacci_rap_constraints_all_buffers,
    gpu_evaluate_fibonacci_rap_constraints_to_buffer, FibRapConstraintState,
};
#[cfg(all(target_os = "macos", feature = "metal"))]
use crate::metal::fft::{gpu_interpolate_offset_fft_buffer_to_buffer, CosetShiftState};
#[cfg(all(target_os = "macos", feature = "metal"))]
use crate::metal::merkle::{gpu_batch_commit_paired_from_column_buffers, GpuKeccakMerkleState};
#[cfg(all(target_os = "macos", feature = "metal"))]
use lambdaworks_math::field::fields::u64_goldilocks_field::Goldilocks64Field;

/// GPU-accelerated transition zerofier evaluation for Goldilocks field.
///
/// Mirrors `AIR::transition_zerofier_evaluations` from the CPU prover but uses
/// GPU FFT for the expensive end-exemptions polynomial evaluation (the main
/// bottleneck at large trace sizes). The base zerofier is computed on CPU since
/// it only has `blowup_factor * period` elements (typically 4).
#[cfg(all(target_os = "macos", feature = "metal"))]
fn gpu_transition_zerofier_evaluations<A>(
    air: &A,
    domain: &Domain<Goldilocks64Field>,
    state: &StarkMetalState,
) -> Vec<Vec<FieldElement<Goldilocks64Field>>>
where
    A: AIR<Field = Goldilocks64Field, FieldExtension = Goldilocks64Field>,
{
    type F = Goldilocks64Field;

    let constraints = air.transition_constraints();
    let blowup_factor = domain.blowup_factor;
    let trace_length = domain.trace_roots_of_unity.len();
    let trace_primitive_root = &domain.trace_primitive_root;
    let coset_offset = &domain.coset_offset;
    let lde_root_order = u64::from((blowup_factor * trace_length).trailing_zeros());
    let lde_root = F::get_primitive_root_of_unity(lde_root_order)
        .expect("primitive root of unity must exist for LDE domain size");

    // Step 1: Collect unique keys (same as CPU)
    type BaseZerofierKey = (usize, usize, Option<usize>, Option<usize>);
    type EndExemptionsKey = (usize, usize);

    let mut unique_base_keys: Vec<BaseZerofierKey> = Vec::new();
    let mut unique_end_exemptions_keys: Vec<EndExemptionsKey> = Vec::new();

    for c in constraints.iter() {
        let base_key: BaseZerofierKey = (
            c.period(),
            c.offset(),
            c.exemptions_period(),
            c.periodic_exemptions_offset(),
        );
        if !unique_base_keys.contains(&base_key) {
            unique_base_keys.push(base_key);
        }

        let end_key: EndExemptionsKey = (c.end_exemptions(), c.period());
        if !unique_end_exemptions_keys.contains(&end_key) {
            unique_end_exemptions_keys.push(end_key);
        }
    }

    // Step 2: Compute base zerofiers on CPU (trivial: only blowup_factor * period elements)
    let base_zerofiers: Vec<_> = unique_base_keys
        .iter()
        .map(
            |&(period, offset, exemptions_period, periodic_exemptions_offset)| {
                compute_base_zerofier_goldilocks(
                    period,
                    offset,
                    exemptions_period,
                    periodic_exemptions_offset,
                    blowup_factor,
                    trace_length,
                    trace_primitive_root,
                    coset_offset,
                    &lde_root,
                )
            },
        )
        .collect();

    let base_zerofier_map: HashMap<BaseZerofierKey, Vec<FieldElement<F>>> =
        unique_base_keys.into_iter().zip(base_zerofiers).collect();

    // Step 3: Compute end exemptions polynomial evaluations using GPU FFT
    let end_exemptions_evals: Vec<_> = unique_end_exemptions_keys
        .iter()
        .map(|&(end_exemptions, period)| {
            gpu_compute_end_exemptions_evals(
                end_exemptions,
                period,
                blowup_factor,
                trace_length,
                trace_primitive_root,
                coset_offset,
                domain.interpolation_domain_size,
                state,
            )
        })
        .collect();

    let end_exemptions_map: HashMap<EndExemptionsKey, Vec<FieldElement<F>>> =
        unique_end_exemptions_keys
            .into_iter()
            .zip(end_exemptions_evals)
            .collect();

    // Step 4: Build final zerofiers by combining base + end_exemptions (same as CPU)
    type ZerofierGroupKey = (usize, usize, Option<usize>, Option<usize>, usize);
    let mut evals = vec![Vec::new(); air.num_transition_constraints()];
    let mut full_zerofier_cache: HashMap<ZerofierGroupKey, Vec<FieldElement<F>>> = HashMap::new();

    for c in constraints.iter() {
        let period = c.period();
        let offset = c.offset();
        let exemptions_period = c.exemptions_period();
        let periodic_exemptions_offset = c.periodic_exemptions_offset();
        let end_exemptions = c.end_exemptions();

        let full_key = (
            period,
            offset,
            exemptions_period,
            periodic_exemptions_offset,
            end_exemptions,
        );

        if let Some(cached) = full_zerofier_cache.get(&full_key) {
            evals[c.constraint_idx()] = cached.clone();
            continue;
        }

        let base_key: BaseZerofierKey = (
            period,
            offset,
            exemptions_period,
            periodic_exemptions_offset,
        );
        let end_key: EndExemptionsKey = (end_exemptions, period);

        let base_zerofier = base_zerofier_map
            .get(&base_key)
            .expect("base_key was inserted into map in previous step");
        let end_exemptions_evals = end_exemptions_map
            .get(&end_key)
            .expect("end_key was inserted into map in previous step");

        let cycled_base = base_zerofier
            .iter()
            .cycle()
            .take(end_exemptions_evals.len());

        let final_zerofier: Vec<_> = std::iter::zip(cycled_base, end_exemptions_evals.iter())
            .map(|(base, exemption)| base * exemption)
            .collect();

        full_zerofier_cache.insert(full_key, final_zerofier.clone());
        evals[c.constraint_idx()] = final_zerofier;
    }

    evals
}

/// Compute base zerofier evaluations (without end exemptions) for Goldilocks field.
///
/// This is a copy of `compute_base_zerofier` from traits.rs, made concrete for Goldilocks.
/// Only computes `blowup_factor * period` elements (typically 4), so no GPU needed.
#[cfg(all(target_os = "macos", feature = "metal"))]
#[allow(clippy::too_many_arguments)]
fn compute_base_zerofier_goldilocks(
    period: usize,
    offset: usize,
    exemptions_period: Option<usize>,
    periodic_exemptions_offset: Option<usize>,
    blowup_factor: usize,
    trace_length: usize,
    trace_primitive_root: &FieldElement<Goldilocks64Field>,
    coset_offset: &FieldElement<Goldilocks64Field>,
    lde_root: &FieldElement<Goldilocks64Field>,
) -> Vec<FieldElement<Goldilocks64Field>> {
    type FpE = FieldElement<Goldilocks64Field>;

    if let Some(exemptions_period_val) = exemptions_period {
        let last_exponent = blowup_factor * exemptions_period_val;
        (0..last_exponent)
            .map(|exponent| {
                let x = lde_root.pow(exponent);
                let offset_times_x = coset_offset * x;
                let offset_exponent = trace_length
                    * periodic_exemptions_offset.expect(
                        "periodic_exemptions_offset must be Some when exemptions_period is Some",
                    )
                    / exemptions_period_val;

                let numerator = offset_times_x.pow(trace_length / exemptions_period_val)
                    - trace_primitive_root.pow(offset_exponent);
                let denominator = offset_times_x.pow(trace_length / period)
                    - trace_primitive_root.pow(offset * trace_length / period);

                use std::ops::Div;
                numerator.div(denominator).expect(
                    "zerofier denominator should be non-zero: coset offset ensures disjoint domains",
                )
            })
            .collect()
    } else {
        let last_exponent = blowup_factor * period;
        let mut evaluations: Vec<FpE> = (0..last_exponent)
            .map(|exponent| {
                let x = lde_root.pow(exponent);
                (coset_offset * x).pow(trace_length / period)
                    - trace_primitive_root.pow(offset * trace_length / period)
            })
            .collect();

        FpE::inplace_batch_inverse(&mut evaluations)
            .expect("batch inverse failed: zerofier evaluation contains zero element");
        evaluations
    }
}

/// Compute end exemptions polynomial evaluations using GPU FFT.
///
/// Builds the end exemptions polynomial on CPU (small, degree = `end_exemptions`),
/// then evaluates it on the LDE domain using GPU FFT instead of CPU FFT.
/// Falls back to CPU FFT if the GPU evaluation fails.
#[cfg(all(target_os = "macos", feature = "metal"))]
#[allow(clippy::too_many_arguments)]
fn gpu_compute_end_exemptions_evals(
    end_exemptions: usize,
    period: usize,
    blowup_factor: usize,
    trace_length: usize,
    trace_primitive_root: &FieldElement<Goldilocks64Field>,
    coset_offset: &FieldElement<Goldilocks64Field>,
    interpolation_domain_size: usize,
    state: &StarkMetalState,
) -> Vec<FieldElement<Goldilocks64Field>> {
    type F = Goldilocks64Field;
    type FpE = FieldElement<F>;

    // Build end exemptions polynomial on CPU (small: degree = end_exemptions)
    let one_poly = Polynomial::new_monomial(FpE::one(), 0);
    let end_exemptions_poly = if end_exemptions == 0 {
        one_poly
    } else {
        (1..=end_exemptions)
            .map(|exemption| trace_primitive_root.pow(trace_length - exemption * period))
            .fold(one_poly, |acc, offset| {
                acc * (Polynomial::new_monomial(FpE::one(), 1) - offset)
            })
    };

    // Evaluate on LDE domain using GPU FFT.
    // We need blowup_factor * interpolation_domain_size evaluation points.
    // gpu_evaluate_offset_fft uses coefficients.len() * blowup_factor as the domain size,
    // so we pad the coefficient vector to interpolation_domain_size.
    let eval_domain_size = blowup_factor * interpolation_domain_size;
    let coeffs = end_exemptions_poly.coefficients();

    // If polynomial is constant (end_exemptions == 0), return constant expansion
    if coeffs.len() <= 1 {
        let val = if coeffs.is_empty() {
            FpE::zero()
        } else {
            coeffs[0]
        };
        return vec![val; eval_domain_size];
    }

    // Pad coefficients to interpolation_domain_size so GPU FFT evaluates on the
    // correct domain (blowup_factor * interpolation_domain_size points).
    let mut padded_coeffs = coeffs.to_vec();
    padded_coeffs.resize(interpolation_domain_size, FpE::zero());

    // Try GPU FFT, fall back to CPU on failure
    match gpu_evaluate_offset_fft::<F>(&padded_coeffs, blowup_factor, coset_offset, state.inner()) {
        Ok(evaluations) => {
            // The GPU FFT should produce exactly eval_domain_size elements.
            // Subsample if larger (shouldn't happen with correct padding).
            let step = evaluations.len() / eval_domain_size;
            if step <= 1 {
                evaluations
            } else {
                evaluations.into_iter().step_by(step).collect()
            }
        }
        Err(_) => {
            // Fallback: CPU FFT
            Polynomial::evaluate_offset_fft(
                &end_exemptions_poly,
                blowup_factor,
                Some(interpolation_domain_size),
                coset_offset,
            )
            .expect("CPU FFT evaluation of end exemptions polynomial must succeed")
        }
    }
}

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
    /// Retained GPU buffers for composition poly LDE (used by DEEP composition to avoid re-upload).
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub lde_composition_gpu_buffers: Option<Vec<metal::Buffer>>,
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

    // Step 2: Build LDETraceTable from GPU round 1 result (borrow, no clone).
    let blowup_factor = air.blowup_factor() as usize;
    let lde_trace = LDETraceTable::from_columns_ref(
        &round_1_result.main_lde_evaluations,
        &round_1_result.aux_lde_evaluations,
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
        lde_composition_gpu_buffers: None,
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

    let mut coefficients: Vec<FpE> = core::iter::successors(Some(FpE::one()), |x| Some(x * beta))
        .take(num_boundary + num_transition)
        .collect();
    let transition_coefficients: Vec<FpE> = coefficients.drain(..num_transition).collect();
    let boundary_coefficients = coefficients;

    // Step 2: Build LDETraceTable from GPU round 1 result (borrow, no clone).
    let blowup_factor = air.blowup_factor() as usize;
    let lde_trace = LDETraceTable::from_columns_ref(
        &round_1_result.main_lde_evaluations,
        &round_1_result.aux_lde_evaluations,
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
    let boundary_evals: Vec<FpE> = (0..num_lde_rows)
        .map(|i| {
            boundary_zerofiers_refs
                .iter()
                .zip(boundary_poly_evals.iter())
                .zip(boundary_coefficients.iter())
                .fold(FpE::zero(), |acc, ((z, bp), coeff)| {
                    acc + z[i] * coeff * bp[i]
                })
        })
        .collect();

    // Step 3b: Extract LDE trace columns for GPU.
    let main_col_0: Vec<FpE> = (0..num_lde_rows)
        .map(|r| *lde_trace.get_main(r, 0))
        .collect();
    let main_col_1: Vec<FpE> = (0..num_lde_rows)
        .map(|r| *lde_trace.get_main(r, 1))
        .collect();
    let aux_col_0: Vec<FpE> = (0..num_lde_rows)
        .map(|r| *lde_trace.get_aux(r, 0))
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
        .map(|i| lde_evaluations.iter().map(|col| col[i]).collect())
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
        lde_composition_gpu_buffers: None,
    })
}

/// GPU-optimized Phase 2 for Goldilocks field with GPU constraint eval, GPU IFFT, and GPU Merkle commit.
///
/// This extends [`gpu_round_2_goldilocks`] by:
/// - Keeping constraint evaluations on GPU (no readback)
/// - Running the composition polynomial IFFT on GPU
/// - Using the GPU Keccak256 shader for Merkle tree commit (paired-row layout)
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
    coset_state: &CosetShiftState,
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

    let mut coefficients: Vec<FpE> = core::iter::successors(Some(FpE::one()), |x| Some(x * beta))
        .take(num_boundary + num_transition)
        .collect();
    let transition_coefficients: Vec<FpE> = coefficients.drain(..num_transition).collect();
    let boundary_coefficients = coefficients;

    let blowup_factor = air.blowup_factor() as usize;
    let num_lde_rows = domain.lde_roots_of_unity_coset.len();

    // Step 3a+3b: Zerofier + boundary + constraint evaluation.
    let t_zerofier = std::time::Instant::now();
    let zerofier_evals = gpu_transition_zerofier_evaluations(air, domain, state);
    eprintln!("    2b zerofier eval:  {:>10.2?}", t_zerofier.elapsed());
    let lde_step_size = air.step_size() * blowup_factor;

    let boundary_constraints = air.boundary_constraints(&round_1_result.rap_challenges);

    // Check if we have retained GPU buffers from Phase 1.
    let has_gpu_bufs = round_1_result
        .main_lde_gpu_buffers
        .as_ref()
        .is_some_and(|b| b.len() >= 2)
        && round_1_result
            .aux_lde_gpu_buffers
            .as_ref()
            .is_some_and(|b| !b.is_empty());

    let t_boundary = std::time::Instant::now();
    let t_constraint = std::time::Instant::now();
    let (constraint_eval_buffer, constraint_eval_len) = if has_gpu_bufs {
        // Fast path: GPU boundary eval + GPU constraint eval with retained buffers.
        let main_bufs = round_1_result
            .main_lde_gpu_buffers
            .as_ref()
            .expect("main_lde_gpu_buffers must be Some in fast path (checked by has_gpu_bufs)");
        let aux_bufs = round_1_result
            .aux_lde_gpu_buffers
            .as_ref()
            .expect("aux_lde_gpu_buffers must be Some in fast path (checked by has_gpu_bufs)");

        // GPU boundary evaluation: per-element inverse via addition chain.
        let bc_descriptors: Vec<(usize, usize, FpE)> = boundary_constraints
            .constraints
            .iter()
            .map(|bc| {
                let col_idx = if bc.is_aux { 2 } else { bc.col };
                (col_idx, bc.step, bc.value)
            })
            .collect();

        let (boundary_buf, _) = crate::metal::constraint_eval::gpu_evaluate_boundary_constraints(
            &domain.lde_roots_of_unity_coset,
            &main_bufs[0],
            &main_bufs[1],
            &aux_bufs[0],
            &bc_descriptors,
            &boundary_coefficients,
            &domain.trace_primitive_root,
            num_lde_rows,
            None,
        )
        .map_err(|e| ProvingError::FieldOperationError(format!("GPU boundary eval error: {e}")))?;

        eprintln!("    2a boundary GPU:   {:>10.2?}", t_boundary.elapsed());

        // Pass boundary buffer directly to constraint eval (no GPU→CPU→GPU round-trip).
        gpu_evaluate_fibonacci_rap_constraints_all_buffers(
            &main_bufs[0],
            &main_bufs[1],
            &aux_bufs[0],
            &boundary_buf,
            num_lde_rows,
            &zerofier_evals,
            &round_1_result.rap_challenges[0],
            &transition_coefficients,
            lde_step_size,
            precompiled_constraint,
        )
        .map_err(|e| ProvingError::FieldOperationError(format!("GPU constraint eval error: {e}")))?
    } else {
        // Fallback: CPU boundary eval + GPU constraint eval with column re-upload.
        // Build LDE trace table from CPU data (only needed in this fallback path).
        let lde_trace = stark_platinum_prover::trace::LDETraceTable::from_columns_ref(
            &round_1_result.main_lde_evaluations,
            &round_1_result.aux_lde_evaluations,
            air.step_size(),
            blowup_factor,
        );

        let mut zerofier_cache: HashMap<usize, Vec<FpE>> = HashMap::new();
        for bc in &boundary_constraints.constraints {
            zerofier_cache.entry(bc.step).or_insert_with(|| {
                let point = domain.trace_primitive_root.pow(bc.step as u64);
                let mut evals: Vec<FpE> = domain
                    .lde_roots_of_unity_coset
                    .iter()
                    .map(|v| v - point)
                    .collect();
                FpE::inplace_batch_inverse(&mut evals).expect(
                    "batch inverse must succeed: coset offset ensures no zeros in zerofier",
                );
                evals
            });
        }
        let boundary_zerofiers_refs: Vec<&Vec<FpE>> = boundary_constraints
            .constraints
            .iter()
            .map(|bc| {
                zerofier_cache
                    .get(&bc.step)
                    .expect("step was inserted into zerofier_cache in previous loop")
            })
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
        let boundary_evals: Vec<FpE> = (0..num_lde_rows)
            .map(|i| {
                boundary_zerofiers_refs
                    .iter()
                    .zip(boundary_poly_evals.iter())
                    .zip(boundary_coefficients.iter())
                    .fold(FpE::zero(), |acc, ((z, bp), coeff)| {
                        acc + z[i] * coeff * bp[i]
                    })
            })
            .collect();
        eprintln!("    2a boundary CPU:   {:>10.2?}", t_boundary.elapsed());

        let main_col_0: Vec<FpE> = (0..num_lde_rows)
            .map(|r| *lde_trace.get_main(r, 0))
            .collect();
        let main_col_1: Vec<FpE> = (0..num_lde_rows)
            .map(|r| *lde_trace.get_main(r, 1))
            .collect();
        let aux_col_0: Vec<FpE> = (0..num_lde_rows)
            .map(|r| *lde_trace.get_aux(r, 0))
            .collect();
        gpu_evaluate_fibonacci_rap_constraints_to_buffer(
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
        .map_err(|e| ProvingError::FieldOperationError(format!("GPU constraint eval error: {e}")))?
    };

    eprintln!("    2c constraint GPU: {:>10.2?}", t_constraint.elapsed());

    // Step 4: GPU coset IFFT — interpolate constraint evaluations entirely on GPU.
    let t_ifft = std::time::Instant::now();
    let coset_offset = air.coset_offset();
    let coeffs_buffer = gpu_interpolate_offset_fft_buffer_to_buffer(
        &constraint_eval_buffer,
        constraint_eval_len,
        &coset_offset,
        coset_state,
        state.inner(),
    )
    .map_err(|e| ProvingError::FieldOperationError(format!("GPU composition IFFT error: {e}")))?;

    eprintln!("    2d GPU IFFT:       {:>10.2?}", t_ifft.elapsed());

    // Step 5: Break into parts.
    let number_of_parts = air.composition_poly_degree_bound() / air.trace_length();
    if number_of_parts == 0 {
        return Err(ProvingError::WrongParameter(
            "composition_poly_degree_bound must be >= trace_length".to_string(),
        ));
    }

    // Step 5a: GPU stride — break coefficients into parts entirely on GPU.
    // IMPORTANT: Stride only the first `composition_poly_degree_bound` coefficients,
    // not the full IFFT output. The IFFT produces `constraint_eval_len` values, but
    // the composition polynomial has degree < `composition_poly_degree_bound`, so the
    // trailing coefficients are zeros. Striding over the full buffer would produce parts
    // with extra zeros, leading to a larger FFT domain and different Merkle root.
    let t_stride = std::time::Instant::now();
    let meaningful_coeffs = air.composition_poly_degree_bound();
    let part_buffers = gpu_break_in_parts_buffer_to_buffers(
        &coeffs_buffer,
        meaningful_coeffs,
        number_of_parts,
        state.inner(),
    )
    .map_err(|e| {
        ProvingError::FieldOperationError(format!("GPU stride (break_in_parts) error: {e}"))
    })?;

    eprintln!("    2e GPU stride:     {:>10.2?}", t_stride.elapsed());

    // Step 5b: Read back coefficients to CPU for OOD eval in Phase 3.
    // Uses UMA zero-copy: reads directly from Metal buffer shared memory,
    // avoiding the intermediate Vec<u64> allocation from retrieve_contents.
    let t_readback = std::time::Instant::now();
    let ptr = coeffs_buffer.contents() as *const u64;
    let coeffs_fe: Vec<FpE> = (0..meaningful_coeffs)
        .map(|i| FieldElement::from(unsafe { *ptr.add(i) }))
        .collect();
    let composition_poly = Polynomial::new(&coeffs_fe);
    let composition_poly_parts = composition_poly.break_in_parts(number_of_parts);

    eprintln!("    2e' OOD readback:  {:>10.2?}", t_readback.elapsed());

    // Step 6: Evaluate each part on LDE domain via GPU buffer-to-buffer FFT.
    let t_lde = std::time::Instant::now();
    let part_len = meaningful_coeffs / number_of_parts;
    let buffer_results = gpu_evaluate_offset_fft_buffer_to_buffers_batch(
        &part_buffers,
        part_len,
        blowup_factor,
        &coset_offset,
        coset_state,
        state.inner(),
    )
    .map_err(|e| ProvingError::FieldOperationError(format!("GPU LDE FFT error: {e}")))?;

    let mut lde_buffers: Vec<metal::Buffer> = Vec::with_capacity(number_of_parts);
    let mut lde_domain_size = 0;

    for (buffer, ds) in buffer_results {
        lde_domain_size = ds;
        lde_buffers.push(buffer);
    }

    eprintln!("    2f LDE FFT:        {:>10.2?}", t_lde.elapsed());

    // Step 7: GPU Merkle commit with paired-row layout directly from FFT buffers.
    let t_commit = std::time::Instant::now();
    let buffer_refs: Vec<&metal::Buffer> = lde_buffers.iter().collect();
    let (tree, root) =
        gpu_batch_commit_paired_from_column_buffers(&buffer_refs, lde_domain_size, keccak_state)
            .ok_or(ProvingError::EmptyCommitment)?;

    eprintln!("    2g Merkle commit:  {:>10.2?}", t_commit.elapsed());

    Ok(GpuRound2Result {
        composition_poly_parts,
        lde_composition_poly_evaluations: Vec::new(), // Not populated; use GPU buffers via UMA
        composition_poly_merkle_tree: tree,
        composition_poly_root: root,
        lde_composition_gpu_buffers: Some(lde_buffers),
    })
}

/// CPU Phase 2 for Fp3 extension field proofs.
///
/// Composition polynomial construction for F != E. Uses the CPU `ConstraintEvaluator`
/// for constraint evaluation and CPU FFT for all polynomial operations, since no
/// Fp3 GPU constraint evaluation shader exists.
#[cfg(all(target_os = "macos", feature = "metal"))]
#[allow(clippy::too_many_arguments)]
pub fn gpu_round_2_fp3<A>(
    air: &A,
    domain: &Domain<Goldilocks64Field>,
    round_1_result: &crate::metal::phases::fp3_types::GpuRound1ResultFp3,
    transcript: &mut impl IsStarkTranscript<
        lambdaworks_math::field::fields::u64_goldilocks_field::Degree3GoldilocksExtensionField,
        Goldilocks64Field,
    >,
) -> Result<crate::metal::phases::fp3_types::GpuRound2ResultFp3, ProvingError>
where
    A: AIR<
        Field = Goldilocks64Field,
        FieldExtension = lambdaworks_math::field::fields::u64_goldilocks_field::Degree3GoldilocksExtensionField,
    >,
{
    use lambdaworks_math::field::fields::u64_goldilocks_field::Degree3GoldilocksExtensionField;
    type F = Goldilocks64Field;
    type Fp3 = Degree3GoldilocksExtensionField;
    type Fp3E = FieldElement<Fp3>;

    // Step 1: Sample beta and compute transition/boundary coefficients (in Fp3).
    let beta: Fp3E = transcript.sample_field_element();
    let num_boundary = air
        .boundary_constraints(&round_1_result.rap_challenges)
        .constraints
        .len();
    let num_transition = air.context().num_transition_constraints;

    let mut coefficients: Vec<Fp3E> = core::iter::successors(Some(Fp3E::one()), |x| Some(x * beta))
        .take(num_boundary + num_transition)
        .collect();
    let transition_coefficients: Vec<Fp3E> = coefficients.drain(..num_transition).collect();
    let boundary_coefficients = coefficients;

    // Step 2: Build LDETraceTable from round 1 result (F main, Fp3 aux).
    let blowup_factor = air.blowup_factor() as usize;
    let lde_trace = stark_platinum_prover::trace::LDETraceTable::from_columns_ref(
        &round_1_result.main_lde_evaluations,
        &round_1_result.aux_lde_evaluations,
        air.step_size(),
        blowup_factor,
    );

    // Step 3: Evaluate constraints on CPU.
    let evaluator =
        ConstraintEvaluator::<F, Fp3, A::PublicInputs>::new(air, &round_1_result.rap_challenges);
    let constraint_evaluations = evaluator.evaluate(
        air,
        &lde_trace,
        domain,
        &transition_coefficients,
        &boundary_coefficients,
        &round_1_result.rap_challenges,
    )?;

    // Step 4: CPU IFFT → composition polynomial coefficients (in Fp3).
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

    // Step 6: CPU FFT LDE each part.
    let lde_evaluations: Vec<Vec<Fp3E>> = composition_poly_parts
        .iter()
        .map(|part| {
            Polynomial::evaluate_offset_fft::<F>(
                part,
                blowup_factor,
                Some(domain.interpolation_domain_size),
                &coset_offset,
            )
            .map(|evals| {
                let target_len = domain.interpolation_domain_size * blowup_factor;
                let step = evals.len() / target_len;
                if step <= 1 {
                    evals
                } else {
                    evals.into_iter().step_by(step).collect()
                }
            })
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| ProvingError::FieldOperationError(format!("CPU composition LDE: {e}")))?;

    // Step 7: Commit composition poly parts with the special paired-row layout.
    let lde_len = lde_evaluations[0].len();

    // Transpose: column-major to row-major
    let mut rows: Vec<Vec<Fp3E>> = (0..lde_len)
        .map(|i| lde_evaluations.iter().map(|col| col[i]).collect())
        .collect();

    // Bit-reverse permute
    in_place_bit_reverse_permute(&mut rows);

    // Pair consecutive rows
    let mut merged_rows = Vec::with_capacity(lde_len / 2);
    let mut iter = rows.into_iter();
    while let (Some(mut chunk0), Some(chunk1)) = (iter.next(), iter.next()) {
        chunk0.extend(chunk1);
        merged_rows.push(chunk0);
    }

    // CPU Merkle commit (Fp3)
    let (tree, root) = cpu_batch_commit(&merged_rows).ok_or(ProvingError::EmptyCommitment)?;

    Ok(crate::metal::phases::fp3_types::GpuRound2ResultFp3 {
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

    /// Differential test: GPU zerofier evaluations must match CPU zerofier evaluations.
    #[test]
    fn gpu_zerofier_matches_cpu_zerofier() {
        for trace_length in [16, 32, 64, 128] {
            let pub_inputs = FibonacciRAPPublicInputs {
                steps: trace_length - 1,
                a0: FpE::one(),
                a1: FpE::one(),
            };
            let proof_options = ProofOptions::default_test_options();
            let trace = fibonacci_rap_trace::<F>([FpE::one(), FpE::one()], trace_length);
            let air = FibonacciRAP::new(trace.num_rows(), &pub_inputs, &proof_options);
            let domain = Domain::new(&air);
            let state = StarkMetalState::new().unwrap();

            // CPU path
            let cpu_zerofier = air.transition_zerofier_evaluations(&domain);

            // GPU path
            let gpu_zerofier = gpu_transition_zerofier_evaluations(&air, &domain, &state);

            assert_eq!(
                cpu_zerofier.len(),
                gpu_zerofier.len(),
                "trace_length={trace_length}: number of zerofier vectors mismatch"
            );

            for (c_idx, (cpu_z, gpu_z)) in cpu_zerofier.iter().zip(gpu_zerofier.iter()).enumerate()
            {
                assert_eq!(
                    cpu_z.len(),
                    gpu_z.len(),
                    "trace_length={trace_length}, constraint {c_idx}: zerofier length mismatch"
                );
                for (i, (c, g)) in cpu_z.iter().zip(gpu_z.iter()).enumerate() {
                    assert_eq!(
                        c, g,
                        "trace_length={trace_length}, constraint {c_idx}, point {i}: value mismatch"
                    );
                }
            }
        }
    }

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
