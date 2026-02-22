//! GPU Phase 3: Out-of-Domain (OOD) Evaluations (CPU).
//!
//! This module mirrors the round 3 logic from the CPU STARK prover. It evaluates
//! trace polynomials and composition polynomial parts at an out-of-domain point `z`.
//! All operations here are small scalar evaluations, so no GPU acceleration is needed.

use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::{IsFFTField, IsField, IsSubFieldOf};
use lambdaworks_math::traits::AsBytes;

use lambdaworks_crypto::fiat_shamir::is_transcript::IsStarkTranscript;

use stark_platinum_prover::domain::Domain;
use stark_platinum_prover::prover::ProvingError;
use stark_platinum_prover::table::Table;
use stark_platinum_prover::trace::get_trace_evaluations;
use stark_platinum_prover::traits::AIR;

use crate::metal::phases::composition::GpuRound2Result;
use crate::metal::phases::rap::GpuRound1Result;

#[cfg(all(target_os = "macos", feature = "metal"))]
use lambdaworks_math::field::fields::u64_goldilocks_field::Goldilocks64Field;

/// Result of GPU Phase 3 (OOD evaluation round).
///
/// This is the GPU equivalent of `Round3<FieldExtension>` from the CPU prover.
/// Contains the out-of-domain evaluations of trace polynomials and composition
/// polynomial parts, as well as the sampled challenge point `z`.
pub struct GpuRound3Result<F: IsField> {
    /// OOD evaluations of trace polynomials.
    /// Table with rows = evaluation points (z*g^k), cols = trace columns (main + aux).
    pub trace_ood_evaluations: Table<F>,
    /// OOD evaluations of composition polynomial parts at z^N.
    pub composition_poly_parts_ood_evaluation: Vec<FieldElement<F>>,
    /// The out-of-domain challenge point z, needed by round 4.
    pub z: FieldElement<F>,
}

/// Executes GPU Phase 3 of the STARK prover: OOD evaluations.
///
/// This function also handles the transcript operations that happen between
/// round 2 and round 3 in the CPU prover:
///
/// 1. Append composition poly root to transcript
/// 2. Sample z from transcript using `sample_z_ood`
/// 3. Compute `z_power = z^N` where N = number of composition poly parts
/// 4. Evaluate each composition poly part at `z_power`
/// 5. Evaluate trace polynomials at z*g^k for all required frame offsets
/// 6. Append trace OOD evaluations to transcript (column by column)
/// 7. Append composition poly OOD evaluations to transcript
/// 8. Return result including z for use in round 4
///
/// # Type Parameters
///
/// - `F`: The base field (must equal the extension field for our GPU prover)
/// - `A`: An AIR whose `Field` and `FieldExtension` are both `F`
///
/// # Errors
///
/// Returns `ProvingError` if any operation fails (currently infallible but
/// returns Result for consistency with other phases).
pub fn gpu_round_3<F, A>(
    air: &A,
    domain: &Domain<F>,
    round_1_result: &GpuRound1Result<F>,
    round_2_result: &GpuRound2Result<F>,
    transcript: &mut impl IsStarkTranscript<F, F>,
) -> Result<GpuRound3Result<F>, ProvingError>
where
    F: IsFFTField + IsSubFieldOf<F> + Send + Sync,
    FieldElement<F>: AsBytes,
    A: AIR<Field = F, FieldExtension = F>,
{
    // Step 1: Append composition poly root to transcript.
    // This matches the CPU prover's `transcript.append_bytes(&round_2_result.composition_poly_root)`
    // which happens between round 2 and round 3 in prove().
    transcript.append_bytes(&round_2_result.composition_poly_root);

    // Step 2: Sample z from transcript.
    let z = transcript.sample_z_ood(
        &domain.lde_roots_of_unity_coset,
        &domain.trace_roots_of_unity,
    );

    // Step 3: Compute z^N where N is the number of composition poly parts.
    let z_power = z.pow(round_2_result.composition_poly_parts.len());

    // Step 4: Evaluate each composition poly part H_i at z^N.
    let composition_poly_parts_ood_evaluation: Vec<FieldElement<F>> = round_2_result
        .composition_poly_parts
        .iter()
        .map(|part| part.evaluate(&z_power))
        .collect();

    // Step 5: Evaluate trace polynomials at z*g^k for all required frame offsets.
    // This produces the Out-of-Domain Frame needed by the verifier.
    let trace_ood_evaluations = get_trace_evaluations::<F, F>(
        &round_1_result.main_trace_polys,
        &round_1_result.aux_trace_polys,
        &z,
        &air.context().transition_offsets,
        &domain.trace_primitive_root,
        air.step_size(),
    );

    // Step 6: Append trace OOD evaluations to transcript (column by column).
    let trace_ood_columns = trace_ood_evaluations.columns();
    for col in trace_ood_columns.iter() {
        for elem in col.iter() {
            transcript.append_field_element(elem);
        }
    }

    // Step 7: Append composition poly OOD evaluations to transcript.
    for element in composition_poly_parts_ood_evaluation.iter() {
        transcript.append_field_element(element);
    }

    // Step 8: Return result.
    Ok(GpuRound3Result {
        trace_ood_evaluations,
        composition_poly_parts_ood_evaluation,
        z,
    })
}

/// OOD evaluations for the Goldilocks optimized path using barycentric interpolation.
///
/// Uses barycentric interpolation on **original trace evaluations** (N points on
/// the roots-of-unity domain) to evaluate trace polynomials at OOD points. This
/// avoids reading GPU coefficient buffers entirely and eliminates the need for
/// coefficient storage after Phase 1.
///
/// Composition poly parts still use Horner on coefficient-form polynomials.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn gpu_round_3_goldilocks<A>(
    air: &A,
    domain: &Domain<Goldilocks64Field>,
    round_1_result: &GpuRound1Result<Goldilocks64Field>,
    round_2_result: &GpuRound2Result<Goldilocks64Field>,
    transcript: &mut impl IsStarkTranscript<Goldilocks64Field, Goldilocks64Field>,
) -> Result<GpuRound3Result<Goldilocks64Field>, ProvingError>
where
    A: AIR<Field = Goldilocks64Field, FieldExtension = Goldilocks64Field>,
{
    type F = Goldilocks64Field;
    type FpE = FieldElement<F>;

    // Step 1: Append composition poly root to transcript.
    transcript.append_bytes(&round_2_result.composition_poly_root);

    // Step 2: Sample z from transcript.
    let z = transcript.sample_z_ood(
        &domain.lde_roots_of_unity_coset,
        &domain.trace_roots_of_unity,
    );

    // Step 3: Compute z^N where N is the number of composition poly parts.
    let z_power = z.pow(round_2_result.composition_poly_parts.len());

    // Step 4: Evaluate each composition poly part H_i at z^N.
    let composition_poly_parts_ood_evaluation: Vec<FpE> = round_2_result
        .composition_poly_parts
        .iter()
        .map(|part| part.evaluate(&z_power))
        .collect();

    // Step 5: Evaluate trace polynomials at OOD points using barycentric interpolation
    // on the original trace evaluations (roots-of-unity domain, N points).
    let trace_ood_evaluations = get_trace_evaluations_barycentric(
        &round_1_result.main_trace_evals,
        &round_1_result.aux_trace_evals,
        &z,
        &air.context().transition_offsets,
        &domain.trace_primitive_root,
        air.step_size(),
    );

    // Step 6: Append trace OOD evaluations to transcript (column by column).
    let trace_ood_columns = trace_ood_evaluations.columns();
    for col in trace_ood_columns.iter() {
        for elem in col.iter() {
            transcript.append_field_element(elem);
        }
    }

    // Step 7: Append composition poly OOD evaluations to transcript.
    for element in composition_poly_parts_ood_evaluation.iter() {
        transcript.append_field_element(element);
    }

    Ok(GpuRound3Result {
        trace_ood_evaluations,
        composition_poly_parts_ood_evaluation,
        z,
    })
}

/// Compute trace OOD evaluations via barycentric interpolation on roots-of-unity domain.
///
/// Replaces `get_trace_evaluations` which requires coefficient-form polynomials.
/// Instead, this takes the original trace evaluations `P(omega^i)` and evaluates at
/// OOD points using the barycentric formula with coset_offset = 1.
///
/// Optimized to share denominator computation across columns: for each eval_point,
/// the denominators `(z - omega^i)` are batch-inverted once and reused for all columns,
/// avoiding redundant O(N) batch inversions per column.
///
/// Produces the same `Table<F>` layout as `get_trace_evaluations`.
#[cfg(all(target_os = "macos", feature = "metal"))]
fn get_trace_evaluations_barycentric(
    main_trace_evals: &[Vec<FieldElement<Goldilocks64Field>>],
    aux_trace_evals: &[Vec<FieldElement<Goldilocks64Field>>],
    z: &FieldElement<Goldilocks64Field>,
    frame_offsets: &[usize],
    primitive_root: &FieldElement<Goldilocks64Field>,
    step_size: usize,
) -> Table<Goldilocks64Field> {
    type F = Goldilocks64Field;
    type FpE = FieldElement<F>;

    // Compute evaluation points: z * g^(offset * step_size) for each offset
    let evaluation_points: Vec<FpE> = frame_offsets
        .iter()
        .flat_map(|offset| {
            let start = offset * step_size;
            let end = (offset + 1) * step_size;
            (start..end).collect::<Vec<_>>()
        })
        .map(|exponent| primitive_root.pow(exponent) * z)
        .collect();

    let n = if !main_trace_evals.is_empty() {
        main_trace_evals[0].len()
    } else if !aux_trace_evals.is_empty() {
        aux_trace_evals[0].len()
    } else {
        return Table::new(Vec::new(), 0);
    };

    // Primitive root of the trace domain (N-th root of unity).
    let trace_order = n.trailing_zeros() as u64;
    let omega = F::get_primitive_root_of_unity(trace_order)
        .expect("primitive root must exist for trace domain size");

    // Pre-compute omega powers {1, omega, omega^2, ..., omega^{N-1}} once.
    // These are the barycentric weights (numerator part) shared across all eval_points.
    let mut omega_powers: Vec<FpE> = Vec::with_capacity(n);
    {
        let mut w = FpE::one();
        for _ in 0..n {
            omega_powers.push(w);
            w *= omega;
        }
    }

    // For h=1 (standard domain): bf_scalar = 1/N, vanishing(z) = z^N - 1.
    let n_inv = FpE::from(n as u64)
        .inv()
        .expect("N is a power of two in an FFT field, always invertible");

    let main_width = main_trace_evals.len();
    let aux_width = aux_trace_evals.len();
    let total_width = main_width + aux_width;

    let mut table_data: Vec<FpE> = Vec::with_capacity(evaluation_points.len() * total_width);

    for eval_point in &evaluation_points {
        // Compute denominators (z_j - omega^i) and batch-invert ONCE per eval_point.
        let mut denoms: Vec<FpE> = Vec::with_capacity(n);
        for w_i in &omega_powers {
            denoms.push(eval_point - w_i);
        }
        FieldElement::inplace_batch_inverse(&mut denoms)
            .expect("z should not coincide with any domain point");

        // Pre-multiply: weighted_inv[i] = omega^i * denom_inv[i], shared across all columns.
        let weighted_inv: Vec<FpE> = omega_powers
            .iter()
            .zip(denoms.iter())
            .map(|(w, d)| *w * *d)
            .collect();

        // Prefactor: (z^N - 1) / N
        let vanishing = eval_point.pow(n) - FpE::one();
        let prefactor = n_inv * vanishing;

        // Accumulate for each column using the shared weighted inverses.
        for col_evals in main_trace_evals {
            let mut acc = FpE::zero();
            for (wd, yi) in weighted_inv.iter().zip(col_evals.iter()) {
                acc += *wd * *yi;
            }
            table_data.push(prefactor * acc);
        }
        for col_evals in aux_trace_evals {
            let mut acc = FpE::zero();
            for (wd, yi) in weighted_inv.iter().zip(col_evals.iter()) {
                acc += *wd * *yi;
            }
            table_data.push(prefactor * acc);
        }
    }

    Table::new(table_data, total_width)
}

/// CPU Phase 3 for Fp3 extension field proofs: OOD evaluations.
///
/// Evaluates trace polynomials and composition polynomial parts at an
/// out-of-domain point `z` (in Fp3). Main trace polys are in F, aux trace
/// and composition polys are in Fp3.
pub fn gpu_round_3_fp3<A>(
    air: &A,
    domain: &Domain<
        lambdaworks_math::field::fields::u64_goldilocks_field::Goldilocks64Field,
    >,
    round_1_result: &crate::metal::phases::fp3_types::GpuRound1ResultFp3,
    round_2_result: &crate::metal::phases::fp3_types::GpuRound2ResultFp3,
    transcript: &mut impl IsStarkTranscript<
        lambdaworks_math::field::fields::u64_goldilocks_field::Degree3GoldilocksExtensionField,
        lambdaworks_math::field::fields::u64_goldilocks_field::Goldilocks64Field,
    >,
) -> Result<crate::metal::phases::fp3_types::GpuRound3ResultFp3, ProvingError>
where
    A: AIR<
        Field = lambdaworks_math::field::fields::u64_goldilocks_field::Goldilocks64Field,
        FieldExtension = lambdaworks_math::field::fields::u64_goldilocks_field::Degree3GoldilocksExtensionField,
    >,
{
    use lambdaworks_math::field::fields::u64_goldilocks_field::{
        Degree3GoldilocksExtensionField, Goldilocks64Field,
    };
    type F = Goldilocks64Field;
    type Fp3 = Degree3GoldilocksExtensionField;
    type Fp3E = FieldElement<Fp3>;

    // Step 1: Append composition poly root to transcript.
    transcript.append_bytes(&round_2_result.composition_poly_root);

    // Step 2: Sample z from transcript (in Fp3).
    let z: Fp3E = transcript.sample_z_ood(
        &domain.lde_roots_of_unity_coset,
        &domain.trace_roots_of_unity,
    );

    // Step 3: Compute z^N where N is the number of composition poly parts.
    let z_power = z.pow(round_2_result.composition_poly_parts.len());

    // Step 4: Evaluate each composition poly part H_i at z^N.
    let composition_poly_parts_ood_evaluation: Vec<Fp3E> = round_2_result
        .composition_poly_parts
        .iter()
        .map(|part| part.evaluate(&z_power))
        .collect();

    // Step 5: Evaluate trace polynomials at z*g^k.
    // get_trace_evaluations::<F, Fp3> handles the mixed types:
    // - main trace polys are Polynomial<FieldElement<F>>, evaluated at z (Fp3) → Fp3
    // - aux trace polys are Polynomial<FieldElement<Fp3>>, evaluated at z (Fp3) → Fp3
    let trace_ood_evaluations = get_trace_evaluations::<F, Fp3>(
        &round_1_result.main_trace_polys,
        &round_1_result.aux_trace_polys,
        &z,
        &air.context().transition_offsets,
        &domain.trace_primitive_root,
        air.step_size(),
    );

    // Step 6: Append trace OOD evaluations to transcript (column by column).
    let trace_ood_columns = trace_ood_evaluations.columns();
    for col in trace_ood_columns.iter() {
        for elem in col.iter() {
            transcript.append_field_element(elem);
        }
    }

    // Step 7: Append composition poly OOD evaluations to transcript.
    for element in composition_poly_parts_ood_evaluation.iter() {
        transcript.append_field_element(element);
    }

    Ok(crate::metal::phases::fp3_types::GpuRound3ResultFp3 {
        trace_ood_evaluations,
        composition_poly_parts_ood_evaluation,
        z,
    })
}

#[cfg(all(test, target_os = "macos", feature = "metal"))]
mod tests {
    use super::*;
    use crate::metal::phases::composition::gpu_round_2;
    use crate::metal::phases::rap::gpu_round_1;
    use crate::metal::state::StarkMetalState;
    use lambdaworks_crypto::fiat_shamir::default_transcript::DefaultTranscript;
    use lambdaworks_math::field::fields::u64_goldilocks_field::Goldilocks64Field;
    use stark_platinum_prover::domain::Domain;
    use stark_platinum_prover::examples::fibonacci_rap::{
        fibonacci_rap_trace, FibonacciRAP, FibonacciRAPPublicInputs,
    };
    use stark_platinum_prover::proof::options::ProofOptions;

    type F = Goldilocks64Field;
    type FpE = FieldElement<F>;

    #[test]
    fn gpu_round_3_fibonacci_rap_goldilocks() {
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

        // z should be non-zero
        assert_ne!(round_3.z, FpE::zero());

        // Should have OOD evaluations for each composition poly part
        assert_eq!(
            round_3.composition_poly_parts_ood_evaluation.len(),
            round_2.composition_poly_parts.len()
        );

        // trace_ood_evaluations should have width = total trace columns (main + aux)
        let total_cols = round_1.main_trace_polys.len() + round_1.aux_trace_polys.len();
        assert_eq!(round_3.trace_ood_evaluations.width, total_cols);

        // trace_ood_evaluations should have rows = number of evaluation points
        // (transition_offsets.len() * step_size)
        let expected_rows = air.context().transition_offsets.len() * air.step_size();
        assert_eq!(round_3.trace_ood_evaluations.height, expected_rows);

        // All composition poly part OOD evaluations should be non-zero
        // (extremely unlikely to be zero for a random z)
        for eval in &round_3.composition_poly_parts_ood_evaluation {
            assert_ne!(*eval, FpE::zero());
        }
    }
}
