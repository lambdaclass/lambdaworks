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
        let expected_rows =
            air.context().transition_offsets.len() * air.step_size();
        assert_eq!(round_3.trace_ood_evaluations.height, expected_rows);

        // All composition poly part OOD evaluations should be non-zero
        // (extremely unlikely to be zero for a random z)
        for eval in &round_3.composition_poly_parts_ood_evaluation {
            assert_ne!(*eval, FpE::zero());
        }
    }
}
