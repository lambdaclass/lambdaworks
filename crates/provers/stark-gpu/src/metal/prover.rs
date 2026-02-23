//! End-to-end GPU STARK prover.

use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::{IsFFTField, IsSubFieldOf};
use lambdaworks_math::traits::AsBytes;

use lambdaworks_crypto::fiat_shamir::is_transcript::IsStarkTranscript;

use stark_platinum_prover::domain::Domain;
use stark_platinum_prover::proof::stark::StarkProof;
use stark_platinum_prover::prover::ProvingError;
use stark_platinum_prover::trace::TraceTable;
use stark_platinum_prover::traits::AIR;

#[cfg(all(target_os = "macos", feature = "metal"))]
use crate::metal::merkle::GpuMerkleState;
#[cfg(all(target_os = "macos", feature = "metal"))]
use crate::metal::phases::composition::{gpu_round_2, gpu_round_2_goldilocks_merkle};
#[cfg(all(target_os = "macos", feature = "metal"))]
use crate::metal::phases::fri::{gpu_round_4, gpu_round_4_goldilocks};
#[cfg(all(target_os = "macos", feature = "metal"))]
use crate::metal::phases::ood::{gpu_round_3, gpu_round_3_goldilocks};
#[cfg(all(target_os = "macos", feature = "metal"))]
use crate::metal::phases::rap::{gpu_round_1, gpu_round_1_goldilocks};
#[cfg(all(target_os = "macos", feature = "metal"))]
use crate::metal::state::StarkMetalState;

/// Map a Metal/shader init error to `ProvingError`.
#[cfg(all(target_os = "macos", feature = "metal"))]
fn shader_err(label: &str, e: impl std::fmt::Display) -> ProvingError {
    ProvingError::FieldOperationError(format!("{label}: {e}"))
}

/// Generic GPU STARK prover (works with any FFT-friendly field).
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn prove_gpu<F, A>(
    air: &A,
    trace: &mut TraceTable<F, F>,
    transcript: &mut impl IsStarkTranscript<F, F>,
) -> Result<StarkProof<F, F>, ProvingError>
where
    F: IsFFTField + IsSubFieldOf<F> + Send + Sync,
    F::BaseType: Copy,
    FieldElement<F>: AsBytes + Sync + Send,
    A: AIR<Field = F, FieldExtension = F>,
{
    let state =
        StarkMetalState::new().map_err(|e| ProvingError::FieldOperationError(e.to_string()))?;
    let domain = Domain::new(air);

    let round_1 = gpu_round_1(air, trace, &domain, transcript, &state)?;
    let round_2 = gpu_round_2(air, &domain, &round_1, transcript, &state)?;
    let round_3 = gpu_round_3(air, &domain, &round_1, &round_2, transcript)?;
    let round_4 = gpu_round_4(air, &domain, &round_1, &round_2, &round_3, transcript)?;

    Ok(StarkProof {
        trace_length: air.trace_length(),
        lde_trace_main_merkle_root: round_1.main_merkle_root,
        lde_trace_aux_merkle_root: round_1.aux_merkle_root,
        trace_ood_evaluations: round_3.trace_ood_evaluations,
        composition_poly_root: round_2.composition_poly_root,
        composition_poly_parts_ood_evaluation: round_3.composition_poly_parts_ood_evaluation,
        fri_layers_merkle_roots: round_4.fri_layers_merkle_roots,
        fri_last_value: round_4.fri_last_value,
        query_list: round_4.query_list,
        deep_poly_openings: round_4.deep_poly_openings,
        nonce: round_4.nonce,
    })
}

/// Fully GPU-optimized prover for Goldilocks field (pre-compiled shaders).
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn prove_gpu_optimized<A>(
    air: &A,
    trace: &mut TraceTable<
        lambdaworks_math::field::fields::u64_goldilocks_field::Goldilocks64Field,
        lambdaworks_math::field::fields::u64_goldilocks_field::Goldilocks64Field,
    >,
    transcript: &mut impl IsStarkTranscript<
        lambdaworks_math::field::fields::u64_goldilocks_field::Goldilocks64Field,
        lambdaworks_math::field::fields::u64_goldilocks_field::Goldilocks64Field,
    >,
) -> Result<
    StarkProof<
        lambdaworks_math::field::fields::u64_goldilocks_field::Goldilocks64Field,
        lambdaworks_math::field::fields::u64_goldilocks_field::Goldilocks64Field,
    >,
    ProvingError,
>
where
    A: AIR<
        Field = lambdaworks_math::field::fields::u64_goldilocks_field::Goldilocks64Field,
        FieldExtension = lambdaworks_math::field::fields::u64_goldilocks_field::Goldilocks64Field,
    >,
{
    use crate::metal::constraint_eval::{FibRapConstraintState, FusedConstraintState};
    use crate::metal::deep_composition::{DeepCompositionState, DomainInversionState};
    use crate::metal::fft::CosetShiftState;
    use crate::metal::phases::fri::{FriDomainInvState, FriFoldEvalState, FriSquareInvState};

    let state =
        StarkMetalState::new().map_err(|e| ProvingError::FieldOperationError(e.to_string()))?;
    let domain = Domain::new(air);

    // Pre-compile GPU shaders once for the entire prove call.
    let (device, queue) = (&state.inner().device, &state.inner().queue);
    let constraint_state =
        FibRapConstraintState::new().map_err(|e| shader_err("Constraint shader", e))?;
    let fused_state =
        FusedConstraintState::new().map_err(|e| shader_err("Fused constraint shader", e))?;
    let deep_comp_state =
        DeepCompositionState::new().map_err(|e| shader_err("DEEP composition shader", e))?;
    let keccak_state =
        GpuMerkleState::new_keccak().map_err(|e| shader_err("Keccak256 shader", e))?;
    let coset_state = CosetShiftState::from_device_and_queue(device, queue)
        .map_err(|e| shader_err("Coset shift shader", e))?;
    let fold_eval_state = FriFoldEvalState::from_device_and_queue(device, queue)
        .map_err(|e| shader_err("FRI fold eval shader", e))?;
    let fri_domain_inv_state = FriDomainInvState::from_device_and_queue(device, queue)
        .map_err(|e| shader_err("FRI domain inv shader", e))?;
    let fri_square_inv_state = FriSquareInvState::from_device_and_queue(device, queue)
        .map_err(|e| shader_err("FRI square inv shader", e))?;
    let domain_inv_state =
        DomainInversionState::new().map_err(|e| shader_err("Domain inversions shader", e))?;

    let t = std::time::Instant::now();
    let round_1 = gpu_round_1_goldilocks(
        air,
        trace,
        &domain,
        transcript,
        &state,
        &keccak_state,
        &coset_state,
    )?;
    eprintln!("  Phase 1 (RAP):         {:>10.2?}", t.elapsed());

    let t = std::time::Instant::now();
    let round_2 = gpu_round_2_goldilocks_merkle(
        air,
        &domain,
        &round_1,
        transcript,
        &state,
        Some(&constraint_state),
        Some(&fused_state),
        &keccak_state,
        &coset_state,
    )?;
    eprintln!("  Phase 2 (Composition): {:>10.2?}", t.elapsed());

    let t = std::time::Instant::now();
    let round_3 = gpu_round_3_goldilocks(air, &domain, &round_1, &round_2, transcript)?;
    eprintln!("  Phase 3 (OOD):         {:>10.2?}", t.elapsed());

    let t = std::time::Instant::now();
    let round_4 = gpu_round_4_goldilocks(
        air,
        &domain,
        &round_1,
        &round_2,
        &round_3,
        transcript,
        &state,
        Some(&deep_comp_state),
        &keccak_state,
        &fold_eval_state,
        &fri_domain_inv_state,
        &fri_square_inv_state,
        Some(&domain_inv_state),
    )?;
    eprintln!("  Phase 4 (FRI):         {:>10.2?}", t.elapsed());

    Ok(StarkProof {
        trace_length: air.trace_length(),
        lde_trace_main_merkle_root: round_1.main_merkle_root,
        lde_trace_aux_merkle_root: round_1.aux_merkle_root,
        trace_ood_evaluations: round_3.trace_ood_evaluations,
        composition_poly_root: round_2.composition_poly_root,
        composition_poly_parts_ood_evaluation: round_3.composition_poly_parts_ood_evaluation,
        fri_layers_merkle_roots: round_4.fri_layers_merkle_roots,
        fri_last_value: round_4.fri_last_value,
        query_list: round_4.query_list,
        deep_poly_openings: round_4.deep_poly_openings,
        nonce: round_4.nonce,
    })
}

/// GPU-accelerated prover for Goldilocks + Fp3 extension field.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn prove_gpu_fp3<A>(
    air: &A,
    trace: &mut TraceTable<
        lambdaworks_math::field::fields::u64_goldilocks_field::Goldilocks64Field,
        lambdaworks_math::field::fields::u64_goldilocks_field::Degree3GoldilocksExtensionField,
    >,
    transcript: &mut impl IsStarkTranscript<
        lambdaworks_math::field::fields::u64_goldilocks_field::Degree3GoldilocksExtensionField,
        lambdaworks_math::field::fields::u64_goldilocks_field::Goldilocks64Field,
    >,
) -> Result<
    StarkProof<
        lambdaworks_math::field::fields::u64_goldilocks_field::Goldilocks64Field,
        lambdaworks_math::field::fields::u64_goldilocks_field::Degree3GoldilocksExtensionField,
    >,
    ProvingError,
>
where
    A: AIR<
        Field = lambdaworks_math::field::fields::u64_goldilocks_field::Goldilocks64Field,
        FieldExtension = lambdaworks_math::field::fields::u64_goldilocks_field::Degree3GoldilocksExtensionField,
    >,
{
    use crate::metal::phases::composition::gpu_round_2_fp3;
    use crate::metal::phases::fri::{gpu_round_4_fp3, FriFoldFp3State};
    use crate::metal::phases::ood::gpu_round_3_fp3;
    use crate::metal::phases::rap::gpu_round_1_fp3;

    let state =
        StarkMetalState::new().map_err(|e| ProvingError::FieldOperationError(e.to_string()))?;
    let domain = Domain::new(air);

    let keccak_state =
        GpuMerkleState::new_keccak().map_err(|e| shader_err("Keccak256 shader", e))?;
    let fri_fold_state =
        FriFoldFp3State::from_device_and_queue(&state.inner().device, &state.inner().queue)
            .map_err(|e| shader_err("FRI fold Fp3 shader", e))?;

    let t = std::time::Instant::now();
    let round_1 = gpu_round_1_fp3(air, trace, &domain, transcript, &state, &keccak_state)?;
    eprintln!("  Phase 1 (RAP):         {:>10.2?}", t.elapsed());

    let t = std::time::Instant::now();
    let round_2 = gpu_round_2_fp3(air, &domain, &round_1, transcript)?;
    eprintln!("  Phase 2 (Composition): {:>10.2?}", t.elapsed());

    let t = std::time::Instant::now();
    let round_3 = gpu_round_3_fp3(air, &domain, &round_1, &round_2, transcript)?;
    eprintln!("  Phase 3 (OOD):         {:>10.2?}", t.elapsed());

    let t = std::time::Instant::now();
    let round_4 = gpu_round_4_fp3(
        air,
        &domain,
        &round_1,
        &round_2,
        &round_3,
        transcript,
        &fri_fold_state,
    )?;
    eprintln!("  Phase 4 (FRI):         {:>10.2?}", t.elapsed());

    Ok(StarkProof {
        trace_length: air.trace_length(),
        lde_trace_main_merkle_root: round_1.main_merkle_root,
        lde_trace_aux_merkle_root: round_1.aux_merkle_root,
        trace_ood_evaluations: round_3.trace_ood_evaluations,
        composition_poly_root: round_2.composition_poly_root,
        composition_poly_parts_ood_evaluation: round_3.composition_poly_parts_ood_evaluation,
        fri_layers_merkle_roots: round_4.fri_layers_merkle_roots,
        fri_last_value: round_4.fri_last_value,
        query_list: round_4.query_list,
        deep_poly_openings: round_4.deep_poly_openings,
        nonce: round_4.nonce,
    })
}

#[cfg(all(test, target_os = "macos", feature = "metal"))]
mod tests {
    use std::ops::Div;

    use super::*;
    use lambdaworks_crypto::fiat_shamir::default_transcript::DefaultTranscript;
    use lambdaworks_math::field::fields::u64_goldilocks_field::{
        Degree3GoldilocksExtensionField, Goldilocks64Field,
    };
    use lambdaworks_math::helpers::resize_to_next_power_of_two;
    use stark_platinum_prover::constraints::boundary::{BoundaryConstraint, BoundaryConstraints};
    use stark_platinum_prover::constraints::transition::TransitionConstraint;
    use stark_platinum_prover::context::AirContext;
    use stark_platinum_prover::examples::fibonacci_rap::{
        fibonacci_rap_trace, FibonacciRAP, FibonacciRAPPublicInputs,
    };
    use stark_platinum_prover::proof::options::ProofOptions;
    use stark_platinum_prover::prover::{IsStarkProver, Prover};
    use stark_platinum_prover::traits::TransitionEvaluationContext;
    use stark_platinum_prover::verifier::{IsStarkVerifier, Verifier};

    type F = Goldilocks64Field;
    type FpE = FieldElement<F>;
    type Fp3 = Degree3GoldilocksExtensionField;
    type Fp3E = FieldElement<Fp3>;

    const TRACE_LEN: usize = 16;

    fn default_pub_inputs() -> FibonacciRAPPublicInputs<F> {
        FibonacciRAPPublicInputs {
            steps: TRACE_LEN,
            a0: FpE::one(),
            a1: FpE::one(),
        }
    }

    fn make_air_and_trace() -> (FibonacciRAP<F>, TraceTable<F, F>) {
        let pub_inputs = default_pub_inputs();
        let proof_options = ProofOptions::default_test_options();
        let trace = fibonacci_rap_trace::<F>([FpE::one(), FpE::one()], TRACE_LEN);
        let air = FibonacciRAP::new(trace.num_rows(), &pub_inputs, &proof_options);
        (air, trace)
    }

    #[test]
    fn gpu_proof_verifies_with_cpu_verifier() {
        let (air, mut trace) = make_air_and_trace();
        let mut prover_transcript = DefaultTranscript::<F>::new(&[]);
        let proof = prove_gpu(&air, &mut trace, &mut prover_transcript).unwrap();

        let mut verifier_transcript = DefaultTranscript::<F>::new(&[]);
        let result = Verifier::<F, F, _>::verify(&proof, &air, &mut verifier_transcript);
        assert!(result, "GPU proof must be verified by the CPU verifier");
    }

    #[test]
    fn gpu_proof_matches_cpu_proof() {
        let (air, mut cpu_trace) = make_air_and_trace();
        let mut cpu_transcript = DefaultTranscript::<F>::new(&[]);
        let cpu_proof =
            Prover::<F, F, _>::prove(&air, &mut cpu_trace, &mut cpu_transcript).unwrap();

        let (air, mut gpu_trace) = make_air_and_trace();
        let mut gpu_transcript = DefaultTranscript::<F>::new(&[]);
        let gpu_proof = prove_gpu(&air, &mut gpu_trace, &mut gpu_transcript).unwrap();

        assert_eq!(cpu_proof.trace_length, gpu_proof.trace_length);
        assert_eq!(
            cpu_proof.lde_trace_main_merkle_root,
            gpu_proof.lde_trace_main_merkle_root
        );
        assert_eq!(
            cpu_proof.lde_trace_aux_merkle_root,
            gpu_proof.lde_trace_aux_merkle_root
        );
        assert_eq!(
            cpu_proof.composition_poly_root,
            gpu_proof.composition_poly_root
        );
        assert_eq!(
            cpu_proof.fri_layers_merkle_roots,
            gpu_proof.fri_layers_merkle_roots
        );
        assert_eq!(cpu_proof.fri_last_value, gpu_proof.fri_last_value);
        assert_eq!(cpu_proof.nonce, gpu_proof.nonce);
        assert_eq!(
            cpu_proof.deep_poly_openings.len(),
            gpu_proof.deep_poly_openings.len()
        );
    }

    #[test]
    fn gpu_optimized_proof_verifies_with_cpu_verifier() {
        let (air, mut trace) = make_air_and_trace();
        let mut prover_transcript = DefaultTranscript::<F>::new(&[]);
        let proof = prove_gpu_optimized(&air, &mut trace, &mut prover_transcript).unwrap();

        let mut verifier_transcript = DefaultTranscript::<F>::new(&[]);
        let result = Verifier::<F, F, _>::verify(&proof, &air, &mut verifier_transcript);
        assert!(
            result,
            "GPU optimized proof must be verified by the CPU verifier"
        );
    }

    #[test]
    fn gpu_optimized_proof_matches_cpu_proof() {
        let (air, mut cpu_trace) = make_air_and_trace();
        let mut cpu_transcript = DefaultTranscript::<F>::new(&[]);
        let cpu_proof =
            Prover::<F, F, _>::prove(&air, &mut cpu_trace, &mut cpu_transcript).unwrap();

        let (air, mut gpu_trace) = make_air_and_trace();
        let mut gpu_transcript = DefaultTranscript::<F>::new(&[]);
        let gpu_proof = prove_gpu_optimized(&air, &mut gpu_trace, &mut gpu_transcript).unwrap();

        assert_eq!(cpu_proof.trace_length, gpu_proof.trace_length);
        assert_eq!(
            cpu_proof.lde_trace_main_merkle_root,
            gpu_proof.lde_trace_main_merkle_root
        );
        assert_eq!(
            cpu_proof.composition_poly_root,
            gpu_proof.composition_poly_root
        );
        assert_eq!(
            cpu_proof.fri_layers_merkle_roots,
            gpu_proof.fri_layers_merkle_roots
        );
        assert_eq!(cpu_proof.fri_last_value, gpu_proof.fri_last_value);
        assert_eq!(cpu_proof.nonce, gpu_proof.nonce);
    }

    // ---- Fp3 extension field test infrastructure ----

    #[derive(Clone)]
    struct FibConstraintFp3;

    impl TransitionConstraint<F, Fp3> for FibConstraintFp3 {
        fn degree(&self) -> usize {
            1
        }

        fn constraint_idx(&self) -> usize {
            0
        }

        fn end_exemptions(&self) -> usize {
            // Hardcoded for steps=16, padded trace_length=32
            3 + 32 - 16 - 1
        }

        fn evaluate(
            &self,
            evaluation_context: &TransitionEvaluationContext<F, Fp3>,
            transition_evaluations: &mut [Fp3E],
        ) {
            match evaluation_context {
                TransitionEvaluationContext::Prover { frame, .. } => {
                    let a0 = frame
                        .get_evaluation_step(0)
                        .get_main_evaluation_element(0, 0);
                    let a1 = frame
                        .get_evaluation_step(1)
                        .get_main_evaluation_element(0, 0);
                    let a2 = frame
                        .get_evaluation_step(2)
                        .get_main_evaluation_element(0, 0);
                    transition_evaluations[0] = (a2 - a1 - a0).to_extension::<Fp3>();
                }
                TransitionEvaluationContext::Verifier { frame, .. } => {
                    let a0 = frame
                        .get_evaluation_step(0)
                        .get_main_evaluation_element(0, 0);
                    let a1 = frame
                        .get_evaluation_step(1)
                        .get_main_evaluation_element(0, 0);
                    let a2 = frame
                        .get_evaluation_step(2)
                        .get_main_evaluation_element(0, 0);
                    transition_evaluations[0] = a2 - a1 - a0;
                }
            }
        }
    }

    #[derive(Clone)]
    struct PermutationConstraintFp3;

    impl TransitionConstraint<F, Fp3> for PermutationConstraintFp3 {
        fn degree(&self) -> usize {
            2
        }

        fn constraint_idx(&self) -> usize {
            1
        }

        fn end_exemptions(&self) -> usize {
            1
        }

        fn evaluate(
            &self,
            evaluation_context: &TransitionEvaluationContext<F, Fp3>,
            transition_evaluations: &mut [Fp3E],
        ) {
            match evaluation_context {
                TransitionEvaluationContext::Prover {
                    frame,
                    rap_challenges,
                    ..
                } => {
                    let s0 = frame.get_evaluation_step(0);
                    let s1 = frame.get_evaluation_step(1);
                    let z_i = s0.get_aux_evaluation_element(0, 0);
                    let z_next = s1.get_aux_evaluation_element(0, 0);
                    let gamma = &rap_challenges[0];
                    let a_i = s0.get_main_evaluation_element(0, 0);
                    let b_i = s0.get_main_evaluation_element(0, 1);
                    transition_evaluations[1] = z_next * (b_i + gamma) - z_i * (a_i + gamma);
                }
                TransitionEvaluationContext::Verifier {
                    frame,
                    rap_challenges,
                    ..
                } => {
                    let s0 = frame.get_evaluation_step(0);
                    let s1 = frame.get_evaluation_step(1);
                    let z_i = s0.get_aux_evaluation_element(0, 0);
                    let z_next = s1.get_aux_evaluation_element(0, 0);
                    let gamma = &rap_challenges[0];
                    let a_i = s0.get_main_evaluation_element(0, 0);
                    let b_i = s0.get_main_evaluation_element(0, 1);
                    transition_evaluations[1] = z_next * (b_i + gamma) - z_i * (a_i + gamma);
                }
            }
        }
    }

    struct FibonacciRAPFp3 {
        context: AirContext,
        trace_length: usize,
        pub_inputs: FibonacciRAPPublicInputs<F>,
        transition_constraints: Vec<Box<dyn TransitionConstraint<F, Fp3>>>,
    }

    impl AIR for FibonacciRAPFp3 {
        type Field = F;
        type FieldExtension = Fp3;
        type PublicInputs = FibonacciRAPPublicInputs<F>;

        fn step_size(&self) -> usize {
            1
        }

        fn new(
            trace_length: usize,
            pub_inputs: &Self::PublicInputs,
            proof_options: &ProofOptions,
        ) -> Self {
            let transition_constraints: Vec<Box<dyn TransitionConstraint<F, Fp3>>> = vec![
                Box::new(FibConstraintFp3),
                Box::new(PermutationConstraintFp3),
            ];
            let context = AirContext {
                proof_options: proof_options.clone(),
                trace_columns: 3,
                transition_offsets: vec![0, 1, 2],
                num_transition_constraints: transition_constraints.len(),
            };
            Self {
                context,
                trace_length,
                pub_inputs: pub_inputs.clone(),
                transition_constraints,
            }
        }

        fn build_auxiliary_trace(&self, trace: &mut TraceTable<F, Fp3>, challenges: &[Fp3E]) {
            let main_segment_cols = trace.columns_main();
            let not_perm = &main_segment_cols[0];
            let perm = &main_segment_cols[1];
            let gamma = &challenges[0];

            let mut aux_col: Vec<Fp3E> = Vec::new();
            for i in 0..trace.num_rows() {
                if i == 0 {
                    aux_col.push(Fp3E::one());
                } else {
                    let z_i = &aux_col[i - 1];
                    let n_p_term = not_perm[i - 1] + gamma;
                    let p_term = perm[i - 1] + gamma;
                    aux_col.push(z_i * n_p_term.div(p_term).unwrap());
                }
            }
            for (i, aux_elem) in aux_col.iter().enumerate().take(trace.num_rows()) {
                trace.set_aux(i, 0, *aux_elem);
            }
        }

        fn build_rap_challenges(
            &self,
            transcript: &mut dyn IsStarkTranscript<Fp3, F>,
        ) -> Vec<Fp3E> {
            vec![transcript.sample_field_element()]
        }

        fn trace_layout(&self) -> (usize, usize) {
            (2, 1)
        }

        fn boundary_constraints(&self, _rap_challenges: &[Fp3E]) -> BoundaryConstraints<Fp3> {
            let a0 = BoundaryConstraint::new_simple_main(0, Fp3E::one());
            let a1 = BoundaryConstraint::new_simple_main(1, Fp3E::one());
            let a0_aux = BoundaryConstraint::new_aux(0, 0, Fp3E::one());
            BoundaryConstraints::from_constraints(vec![a0, a1, a0_aux])
        }

        fn transition_constraints(&self) -> &Vec<Box<dyn TransitionConstraint<F, Fp3>>> {
            &self.transition_constraints
        }

        fn context(&self) -> &AirContext {
            &self.context
        }

        fn composition_poly_degree_bound(&self) -> usize {
            self.trace_length()
        }

        fn trace_length(&self) -> usize {
            self.trace_length
        }

        fn pub_inputs(&self) -> &Self::PublicInputs {
            &self.pub_inputs
        }
    }

    fn fibonacci_rap_trace_fp3(
        initial_values: [FpE; 2],
        trace_length: usize,
    ) -> TraceTable<F, Fp3> {
        let mut fib_seq: Vec<FpE> = vec![];
        fib_seq.push(initial_values[0]);
        fib_seq.push(initial_values[1]);
        for i in 2..trace_length {
            fib_seq.push(fib_seq[i - 1] + fib_seq[i - 2]);
        }

        let last_value = fib_seq[trace_length - 1];
        let mut fib_permuted = fib_seq.clone();
        fib_permuted[0] = last_value;
        fib_permuted[trace_length - 1] = initial_values[0];

        fib_seq.push(FpE::zero());
        fib_permuted.push(FpE::zero());
        let mut trace_cols = vec![fib_seq, fib_permuted];
        resize_to_next_power_of_two(&mut trace_cols);

        let aux_columns = vec![vec![Fp3E::zero(); trace_cols[0].len()]];
        TraceTable::from_columns(trace_cols, aux_columns, 1)
    }

    #[test]
    fn gpu_fp3_proof_verifies() {
        let pub_inputs = default_pub_inputs();
        let proof_options = ProofOptions::default_test_options();
        let mut trace = fibonacci_rap_trace_fp3([FpE::one(), FpE::one()], TRACE_LEN);
        let air = FibonacciRAPFp3::new(trace.num_rows(), &pub_inputs, &proof_options);

        let mut prover_transcript = DefaultTranscript::<Fp3>::new(&[]);
        let proof = prove_gpu_fp3(&air, &mut trace, &mut prover_transcript).unwrap();

        let mut verifier_transcript = DefaultTranscript::<Fp3>::new(&[]);
        let result = Verifier::<F, Fp3, _>::verify(&proof, &air, &mut verifier_transcript);
        assert!(result, "Fp3 proof must be verified by the CPU verifier");
    }
}
