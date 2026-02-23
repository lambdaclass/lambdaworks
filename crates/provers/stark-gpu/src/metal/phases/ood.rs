//! GPU Phase 3: Out-of-Domain (OOD) evaluations (CPU-side scalar work).

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
pub struct GpuRound3Result<F: IsField> {
    pub trace_ood_evaluations: Table<F>,
    pub composition_poly_parts_ood_evaluation: Vec<FieldElement<F>>,
    pub z: FieldElement<F>,
}

/// Appends OOD evaluations to the transcript (trace columns then composition parts).
fn append_ood_to_transcript<F: IsField>(
    transcript: &mut impl IsStarkTranscript<F, F>,
    trace_ood_evaluations: &Table<F>,
    composition_poly_parts_ood_evaluation: &[FieldElement<F>],
) {
    for col in trace_ood_evaluations.columns().iter() {
        for elem in col.iter() {
            transcript.append_field_element(elem);
        }
    }
    for element in composition_poly_parts_ood_evaluation.iter() {
        transcript.append_field_element(element);
    }
}

/// Executes GPU Phase 3: evaluates trace and composition polynomials at OOD point `z`.
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
    transcript.append_bytes(&round_2_result.composition_poly_root);

    let z = transcript.sample_z_ood(
        &domain.lde_roots_of_unity_coset,
        &domain.trace_roots_of_unity,
    );

    let z_power = crate::metal::exp_power_of_2(
        &z,
        round_2_result.composition_poly_parts.len().trailing_zeros(),
    );

    let composition_poly_parts_ood_evaluation: Vec<FieldElement<F>> = round_2_result
        .composition_poly_parts
        .iter()
        .map(|part| part.evaluate(&z_power))
        .collect();

    let trace_ood_evaluations = get_trace_evaluations::<F, F>(
        &round_1_result.main_trace_polys,
        &round_1_result.aux_trace_polys,
        &z,
        &air.context().transition_offsets,
        &domain.trace_primitive_root,
        air.step_size(),
    );

    append_ood_to_transcript(
        transcript,
        &trace_ood_evaluations,
        &composition_poly_parts_ood_evaluation,
    );

    Ok(GpuRound3Result {
        trace_ood_evaluations,
        composition_poly_parts_ood_evaluation,
        z,
    })
}

/// Goldilocks-optimized Phase 3 using barycentric interpolation on trace evaluations.
///
/// Avoids reading GPU coefficient buffers by interpolating directly from the
/// roots-of-unity domain evaluations.
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

    transcript.append_bytes(&round_2_result.composition_poly_root);

    let z = transcript.sample_z_ood(
        &domain.lde_roots_of_unity_coset,
        &domain.trace_roots_of_unity,
    );

    let z_power = crate::metal::exp_power_of_2(
        &z,
        round_2_result.composition_poly_parts.len().trailing_zeros(),
    );

    let composition_poly_parts_ood_evaluation: Vec<FpE> = round_2_result
        .composition_poly_parts
        .iter()
        .map(|part| part.evaluate(&z_power))
        .collect();

    let trace_ood_evaluations = get_trace_evaluations_barycentric(
        &round_1_result.main_trace_evals,
        &round_1_result.aux_trace_evals,
        &z,
        &air.context().transition_offsets,
        &domain.trace_primitive_root,
        air.step_size(),
    );

    append_ood_to_transcript(
        transcript,
        &trace_ood_evaluations,
        &composition_poly_parts_ood_evaluation,
    );

    Ok(GpuRound3Result {
        trace_ood_evaluations,
        composition_poly_parts_ood_evaluation,
        z,
    })
}

/// Evaluates trace polynomials at OOD points via barycentric interpolation on the
/// roots-of-unity domain. Denominators are batch-inverted once per evaluation point
/// and reused across all columns.
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

    // Build evaluation points: z * g^(offset * step_size) for each offset.
    let max_exponent = frame_offsets
        .iter()
        .map(|offset| (offset + 1) * step_size)
        .max()
        .unwrap_or(0);
    let mut g_powers = Vec::with_capacity(max_exponent);
    let mut g_pow = FpE::one();
    for _ in 0..max_exponent {
        g_powers.push(g_pow);
        g_pow *= primitive_root;
    }
    let evaluation_points: Vec<FpE> = frame_offsets
        .iter()
        .flat_map(|offset| {
            let start = offset * step_size;
            let end = (offset + 1) * step_size;
            (start..end).collect::<Vec<_>>()
        })
        .map(|exponent| g_powers[exponent] * z)
        .collect();

    let n = if !main_trace_evals.is_empty() {
        main_trace_evals[0].len()
    } else if !aux_trace_evals.is_empty() {
        aux_trace_evals[0].len()
    } else {
        return Table::new(Vec::new(), 0);
    };

    let trace_order = n.trailing_zeros() as u64;
    let omega = F::get_primitive_root_of_unity(trace_order)
        .expect("primitive root must exist for trace domain size");

    let mut omega_powers: Vec<FpE> = Vec::with_capacity(n);
    let mut w = FpE::one();
    for _ in 0..n {
        omega_powers.push(w);
        w *= omega;
    }

    // Barycentric scalar for standard domain (h=1): 1/N
    let n_inv = FpE::from(n as u64)
        .inv()
        .expect("N is a power of two in an FFT field, always invertible");

    let main_width = main_trace_evals.len();
    let aux_width = aux_trace_evals.len();
    let total_width = main_width + aux_width;

    use rayon::prelude::*;

    let table_data: Vec<FpE> = evaluation_points
        .par_iter()
        .flat_map(|eval_point| {
            // Batch-invert denominators (z_j - omega^i) once per eval_point.
            let mut denoms: Vec<FpE> = omega_powers.iter().map(|w_i| eval_point - w_i).collect();
            FieldElement::inplace_batch_inverse_parallel(&mut denoms)
                .expect("z should not coincide with any domain point");

            let weighted_inv: Vec<FpE> = omega_powers
                .iter()
                .zip(denoms.iter())
                .map(|(w, d)| *w * *d)
                .collect();

            // Prefactor: (z^N - 1) / N
            let vanishing =
                crate::metal::exp_power_of_2(eval_point, n.trailing_zeros()) - FpE::one();
            let prefactor = n_inv * vanishing;

            let mut row_data = Vec::with_capacity(total_width);
            for col_evals in main_trace_evals.iter().chain(aux_trace_evals.iter()) {
                let acc: FpE = weighted_inv
                    .iter()
                    .zip(col_evals.iter())
                    .map(|(wd, yi)| *wd * *yi)
                    .fold(FpE::zero(), |a, b| a + b);
                row_data.push(prefactor * acc);
            }
            row_data
        })
        .collect();

    Table::new(table_data, total_width)
}

/// Phase 3 for Fp3 extension field proofs.
///
/// Main trace polys are in F, aux trace and composition polys are in Fp3.
pub fn gpu_round_3_fp3<A>(
    air: &A,
    domain: &Domain<Goldilocks64Field>,
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

    transcript.append_bytes(&round_2_result.composition_poly_root);

    let z: Fp3E = transcript.sample_z_ood(
        &domain.lde_roots_of_unity_coset,
        &domain.trace_roots_of_unity,
    );

    let z_power = crate::metal::exp_power_of_2(
        &z,
        round_2_result.composition_poly_parts.len().trailing_zeros(),
    );

    let composition_poly_parts_ood_evaluation: Vec<Fp3E> = round_2_result
        .composition_poly_parts
        .iter()
        .map(|part| part.evaluate(&z_power))
        .collect();

    let trace_ood_evaluations = get_trace_evaluations::<F, Fp3>(
        &round_1_result.main_trace_polys,
        &round_1_result.aux_trace_polys,
        &z,
        &air.context().transition_offsets,
        &domain.trace_primitive_root,
        air.step_size(),
    );

    let trace_ood_columns = trace_ood_evaluations.columns();
    for col in trace_ood_columns.iter() {
        for elem in col.iter() {
            transcript.append_field_element(elem);
        }
    }
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

        assert_ne!(round_3.z, FpE::zero());
        assert_eq!(
            round_3.composition_poly_parts_ood_evaluation.len(),
            round_2.composition_poly_parts.len()
        );

        let total_cols = round_1.main_trace_polys.len() + round_1.aux_trace_polys.len();
        assert_eq!(round_3.trace_ood_evaluations.width, total_cols);

        let expected_rows = air.context().transition_offsets.len() * air.step_size();
        assert_eq!(round_3.trace_ood_evaluations.height, expected_rows);

        for eval in &round_3.composition_poly_parts_ood_evaluation {
            assert_ne!(*eval, FpE::zero());
        }
    }
}
