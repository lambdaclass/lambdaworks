//! GPU Phase 3: Out-of-Domain (OOD) Evaluations via Barycentric Interpolation.
//!
//! Evaluates trace polynomials and composition polynomial parts at out-of-domain
//! points using barycentric Lagrange interpolation on LDE evaluation data.
//! This avoids Horner evaluation on coefficient-form polynomials, enabling
//! elimination of the trace iFFT in Phase 1.

use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::{IsFFTField, IsField, IsSubFieldOf};
use lambdaworks_math::polynomial::barycentric::barycentric_evaluate_on_coset;
use lambdaworks_math::traits::AsBytes;

use lambdaworks_crypto::fiat_shamir::is_transcript::IsStarkTranscript;

use stark_platinum_prover::domain::Domain;
use stark_platinum_prover::prover::ProvingError;
use stark_platinum_prover::table::Table;
use stark_platinum_prover::traits::AIR;

use crate::metal::phases::composition::GpuRound2Result;
use crate::metal::phases::rap::GpuRound1Result;

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

/// Executes GPU Phase 3 of the STARK prover: OOD evaluations via barycentric interpolation.
///
/// Uses barycentric Lagrange interpolation on LDE evaluation data instead of
/// Horner evaluation on coefficient-form polynomials. This means the trace
/// polynomials in coefficient form (from Round 1) are not needed.
///
/// 1. Append composition poly root to transcript
/// 2. Sample z from transcript using `sample_z_ood`
/// 3. Evaluate each composition poly part at `z^N` via barycentric on its LDE
/// 4. Evaluate trace columns at z*g^k via barycentric on trace LDE
/// 5. Append evaluations to transcript
/// 6. Return result including z for use in round 4
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
    transcript.append_bytes(&round_2_result.composition_poly_root);

    // Step 2: Sample z from transcript.
    let z = transcript.sample_z_ood(
        &domain.lde_roots_of_unity_coset,
        &domain.trace_roots_of_unity,
    );

    // Step 3: Evaluate composition poly parts at z^N via barycentric.
    let num_parts = round_2_result.lde_composition_poly_evaluations.len();
    let z_power = z.pow(num_parts);

    let comp_lde_size = round_2_result.lde_composition_poly_evaluations[0].len();
    let comp_lde_order = comp_lde_size.trailing_zeros() as u64;
    let omega_comp = F::get_primitive_root_of_unity(comp_lde_order)
        .expect("composition LDE size must be a valid power of two");

    let composition_poly_parts_ood_evaluation: Vec<FieldElement<F>> = round_2_result
        .lde_composition_poly_evaluations
        .iter()
        .map(|evals| {
            barycentric_evaluate_on_coset::<F, F>(evals, &domain.coset_offset, &omega_comp, &z_power)
        })
        .collect();

    // Step 4: Evaluate trace columns at z*g^k via barycentric on LDE data.
    let lde_size = domain.lde_roots_of_unity_coset.len();
    let lde_order = lde_size.trailing_zeros() as u64;
    let omega_lde = F::get_primitive_root_of_unity(lde_order)
        .expect("LDE size must be a valid power of two");

    let frame_offsets = &air.context().transition_offsets;
    let step_size = air.step_size();

    let evaluation_points: Vec<FieldElement<F>> = frame_offsets
        .iter()
        .flat_map(|offset| {
            let start = offset * step_size;
            let end = (offset + 1) * step_size;
            (start..end).collect::<Vec<_>>()
        })
        .map(|exponent| &domain.trace_primitive_root.pow(exponent) * &z)
        .collect();

    let main_width = round_1_result.main_lde_evaluations.len();
    let aux_width = round_1_result.aux_lde_evaluations.len();
    let table_width = main_width + aux_width;

    let mut table_data = Vec::with_capacity(evaluation_points.len() * table_width);
    for eval_point in &evaluation_points {
        for col_evals in &round_1_result.main_lde_evaluations {
            table_data.push(barycentric_evaluate_on_coset::<F, F>(
                col_evals,
                &domain.coset_offset,
                &omega_lde,
                eval_point,
            ));
        }
        for col_evals in &round_1_result.aux_lde_evaluations {
            table_data.push(barycentric_evaluate_on_coset::<F, F>(
                col_evals,
                &domain.coset_offset,
                &omega_lde,
                eval_point,
            ));
        }
    }

    let trace_ood_evaluations = Table::new(table_data, table_width);

    // Step 5: Append trace OOD evaluations to transcript (column by column).
    let trace_ood_columns = trace_ood_evaluations.columns();
    for col in trace_ood_columns.iter() {
        for elem in col.iter() {
            transcript.append_field_element(elem);
        }
    }

    // Step 6: Append composition poly OOD evaluations to transcript.
    for element in composition_poly_parts_ood_evaluation.iter() {
        transcript.append_field_element(element);
    }

    Ok(GpuRound3Result {
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
    use stark_platinum_prover::trace::get_trace_evaluations;

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
            round_2.lde_composition_poly_evaluations.len()
        );

        // trace_ood_evaluations should have width = total trace columns (main + aux)
        let total_cols =
            round_1.main_lde_evaluations.len() + round_1.aux_lde_evaluations.len();
        assert_eq!(round_3.trace_ood_evaluations.width, total_cols);

        // trace_ood_evaluations should have rows = number of evaluation points
        let expected_rows = air.context().transition_offsets.len() * air.step_size();
        assert_eq!(round_3.trace_ood_evaluations.height, expected_rows);

        // All composition poly part OOD evaluations should be non-zero
        for eval in &round_3.composition_poly_parts_ood_evaluation {
            assert_ne!(*eval, FpE::zero());
        }
    }

    /// Differential test: barycentric OOD evaluations must match Horner evaluations.
    ///
    /// Runs barycentric (gpu_round_3) then compares its evaluation results against
    /// Horner evaluation using the same z, round_1 polys, and round_2 parts.
    #[test]
    fn barycentric_ood_matches_horner() {
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

        // Run barycentric OOD (consumes transcript state through round 3).
        let round_3 =
            gpu_round_3(&air, &domain, &round_1, &round_2, &mut transcript).unwrap();

        // Horner path: evaluate at the same z using coefficient-form polys.
        let z = &round_3.z;
        let num_parts = round_2.composition_poly_parts.len();
        let z_power = z.pow(num_parts);
        let horner_comp_evals: Vec<FpE> = round_2
            .composition_poly_parts
            .iter()
            .map(|part| part.evaluate(&z_power))
            .collect();
        let horner_trace_evals = get_trace_evaluations::<F, F>(
            &round_1.main_trace_polys,
            &round_1.aux_trace_polys,
            z,
            &air.context().transition_offsets,
            &domain.trace_primitive_root,
            air.step_size(),
        );

        // Verify composition poly OOD evaluations match.
        assert_eq!(
            round_3.composition_poly_parts_ood_evaluation, horner_comp_evals,
            "composition poly OOD evaluations mismatch"
        );

        // Verify trace OOD evaluations match.
        assert_eq!(
            round_3.trace_ood_evaluations.data, horner_trace_evals.data,
            "trace OOD evaluations mismatch"
        );
    }
}
