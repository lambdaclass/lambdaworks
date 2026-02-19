//! End-to-end GPU STARK prover.
//!
//! This module provides `prove_gpu()`, which orchestrates all 4 GPU prover phases
//! and assembles a `StarkProof` that is verifiable by the standard CPU verifier.

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
use crate::metal::merkle::GpuKeccakMerkleState;
#[cfg(all(target_os = "macos", feature = "metal"))]
use crate::metal::phases::composition::gpu_round_2;
#[cfg(all(target_os = "macos", feature = "metal"))]
use crate::metal::phases::composition::gpu_round_2_goldilocks_merkle;
#[cfg(all(target_os = "macos", feature = "metal"))]
use crate::metal::phases::fri::gpu_round_4;
#[cfg(all(target_os = "macos", feature = "metal"))]
use crate::metal::phases::fri::gpu_round_4_goldilocks;
#[cfg(all(target_os = "macos", feature = "metal"))]
use crate::metal::phases::ood::gpu_round_3;
#[cfg(all(target_os = "macos", feature = "metal"))]
use crate::metal::phases::rap::{gpu_round_1, gpu_round_1_goldilocks};
#[cfg(all(target_os = "macos", feature = "metal"))]
use crate::metal::state::StarkMetalState;

/// Prove a STARK using Metal GPU acceleration.
///
/// Produces the same `StarkProof` as the CPU prover. The verifier is unchanged.
///
/// This function orchestrates all 4 GPU prover phases:
/// 1. RAP: trace interpolation + LDE + Merkle commit (GPU FFT)
/// 2. Composition: constraint evaluation + IFFT + LDE + commit (CPU constraints, GPU FFT)
/// 3. OOD: polynomial evaluations at out-of-domain point (CPU)
/// 4. FRI: DEEP composition + iterative folding + queries (CPU)
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

    // Phase 1: RAP (trace interpolation + LDE + Merkle commit)
    let round_1 = gpu_round_1(air, trace, &domain, transcript, &state)?;

    // Phase 2: Composition polynomial (constraint evaluation + GPU FFT LDE + commit)
    let round_2 = gpu_round_2(air, &domain, &round_1, transcript, &state)?;

    // Phase 3: OOD evaluations (polynomial evaluations at out-of-domain point)
    let round_3 = gpu_round_3(air, &domain, &round_1, &round_2, transcript)?;

    // Phase 4: DEEP composition + FRI + queries
    let round_4 = gpu_round_4(air, &domain, &round_1, &round_2, &round_3, transcript)?;

    // Assemble proof (same structure as the CPU prover)
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

/// Prove a STARK using fully GPU-optimized pipeline for Goldilocks field.
///
/// This is a concrete version of [`prove_gpu`] that uses GPU Metal shaders for:
/// - FFT (interpolation + LDE) via `MetalState`
/// - Constraint evaluation via `fibonacci_rap_constraints.metal`
/// - DEEP composition polynomial via `deep_composition.metal`
///
/// Metal shaders are compiled once at the start and reused across all phases,
/// avoiding per-phase recompilation overhead.
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
    use crate::metal::constraint_eval::FibRapConstraintState;
    use crate::metal::deep_composition::DeepCompositionState;
    use crate::metal::fft::CosetShiftState;
    use crate::metal::phases::fri::FriFoldState;

    let state =
        StarkMetalState::new().map_err(|e| ProvingError::FieldOperationError(e.to_string()))?;
    let domain = Domain::new(air);

    // Pre-compile GPU shaders once for the entire prove call.
    let constraint_state = FibRapConstraintState::new()
        .map_err(|e| ProvingError::FieldOperationError(format!("Constraint shader: {e}")))?;
    let deep_comp_state = DeepCompositionState::new()
        .map_err(|e| ProvingError::FieldOperationError(format!("DEEP composition shader: {e}")))?;
    let keccak_state = GpuKeccakMerkleState::new()
        .map_err(|e| ProvingError::FieldOperationError(format!("Keccak256 shader: {e}")))?;
    // Coset shift and FRI fold states share the device/queue from StarkMetalState
    // to avoid exceeding Metal resource limits (command queues).
    let coset_state = CosetShiftState::from_device_and_queue(
        &state.inner().device,
        &state.inner().queue,
    )
    .map_err(|e| ProvingError::FieldOperationError(format!("Coset shift shader: {e}")))?;
    let fri_fold_state = FriFoldState::from_device_and_queue(
        &state.inner().device,
        &state.inner().queue,
    )
    .map_err(|e| ProvingError::FieldOperationError(format!("FRI fold shader: {e}")))?;

    // Phase 1: RAP (trace interpolation + LDE + GPU Merkle commit)
    let t = std::time::Instant::now();
    let round_1 = gpu_round_1_goldilocks(air, trace, &domain, transcript, &state, &keccak_state)?;
    eprintln!("  Phase 1 (RAP):         {:>10.2?}", t.elapsed());

    // Phase 2: Composition polynomial - GPU constraint eval + GPU IFFT + GPU FFT LDE + GPU Merkle commit
    let t = std::time::Instant::now();
    let round_2 = gpu_round_2_goldilocks_merkle(
        air,
        &domain,
        &round_1,
        transcript,
        &state,
        Some(&constraint_state),
        &keccak_state,
        &coset_state,
    )?;
    eprintln!("  Phase 2 (Composition): {:>10.2?}", t.elapsed());

    // Phase 3: OOD evaluations (CPU)
    let t = std::time::Instant::now();
    let round_3 = gpu_round_3(air, &domain, &round_1, &round_2, transcript)?;
    eprintln!("  Phase 3 (OOD):         {:>10.2?}", t.elapsed());

    // Phase 4: DEEP composition (GPU) + FRI (GPU FFT + GPU Merkle) + queries
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
        &coset_state,
        &fri_fold_state,
    )?;
    eprintln!("  Phase 4 (FRI):         {:>10.2?}", t.elapsed());

    // Assemble proof
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

/// Prove a STARK using CPU prover with Goldilocks base field and Fp3 extension.
///
/// This validates the Fp3 extension field infrastructure end-to-end.
/// All phases use the CPU prover's generic F != E path.
/// Future work: replace individual phases with GPU-accelerated versions
/// using the Fp3 Metal shaders (DEEP composition, FRI fold).
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
    use stark_platinum_prover::prover::{IsStarkProver, Prover};
    type GF = lambdaworks_math::field::fields::u64_goldilocks_field::Goldilocks64Field;
    type Fp3 =
        lambdaworks_math::field::fields::u64_goldilocks_field::Degree3GoldilocksExtensionField;

    Prover::<GF, Fp3, _>::prove(air, trace, transcript)
}

#[cfg(all(test, target_os = "macos", feature = "metal"))]
mod tests {
    use super::*;
    use lambdaworks_crypto::fiat_shamir::default_transcript::DefaultTranscript;
    use lambdaworks_math::field::fields::u64_goldilocks_field::Goldilocks64Field;
    use stark_platinum_prover::examples::fibonacci_rap::{
        fibonacci_rap_trace, FibonacciRAP, FibonacciRAPPublicInputs,
    };
    use stark_platinum_prover::proof::options::ProofOptions;
    use stark_platinum_prover::prover::{IsStarkProver, Prover};
    use stark_platinum_prover::verifier::{IsStarkVerifier, Verifier};

    type F = Goldilocks64Field;
    type FpE = FieldElement<F>;

    /// Test that a GPU-generated proof is accepted by the CPU verifier.
    #[test]
    fn gpu_proof_verifies_with_cpu_verifier() {
        let trace_length = 16;
        let pub_inputs = FibonacciRAPPublicInputs {
            steps: trace_length,
            a0: FpE::one(),
            a1: FpE::one(),
        };
        let proof_options = ProofOptions::default_test_options();
        let mut trace = fibonacci_rap_trace::<F>([FpE::one(), FpE::one()], trace_length);
        let air = FibonacciRAP::new(trace.num_rows(), &pub_inputs, &proof_options);

        // GPU prover
        let mut prover_transcript = DefaultTranscript::<F>::new(&[]);
        let proof = prove_gpu(&air, &mut trace, &mut prover_transcript).unwrap();

        // CPU verifier (uses fresh transcript with same seed)
        let mut verifier_transcript = DefaultTranscript::<F>::new(&[]);
        let result = Verifier::<F, F, _>::verify(&proof, &air, &mut verifier_transcript);
        assert!(result, "GPU proof must be verified by the CPU verifier");
    }

    /// Test that GPU proof matches CPU proof byte-for-byte.
    #[test]
    fn gpu_proof_matches_cpu_proof() {
        let trace_length = 16;
        let pub_inputs = FibonacciRAPPublicInputs {
            steps: trace_length,
            a0: FpE::one(),
            a1: FpE::one(),
        };
        let proof_options = ProofOptions::default_test_options();

        // CPU proof
        let mut cpu_trace = fibonacci_rap_trace::<F>([FpE::one(), FpE::one()], trace_length);
        let air = FibonacciRAP::new(cpu_trace.num_rows(), &pub_inputs, &proof_options);
        let mut cpu_transcript = DefaultTranscript::<F>::new(&[]);
        let cpu_proof =
            Prover::<F, F, _>::prove(&air, &mut cpu_trace, &mut cpu_transcript).unwrap();

        // GPU proof
        let mut gpu_trace = fibonacci_rap_trace::<F>([FpE::one(), FpE::one()], trace_length);
        let air = FibonacciRAP::new(gpu_trace.num_rows(), &pub_inputs, &proof_options);
        let mut gpu_transcript = DefaultTranscript::<F>::new(&[]);
        let gpu_proof = prove_gpu(&air, &mut gpu_trace, &mut gpu_transcript).unwrap();

        // Compare proofs
        assert_eq!(
            cpu_proof.trace_length, gpu_proof.trace_length,
            "trace_length mismatch"
        );
        assert_eq!(
            cpu_proof.lde_trace_main_merkle_root, gpu_proof.lde_trace_main_merkle_root,
            "main merkle root mismatch"
        );
        assert_eq!(
            cpu_proof.lde_trace_aux_merkle_root, gpu_proof.lde_trace_aux_merkle_root,
            "aux merkle root mismatch"
        );
        assert_eq!(
            cpu_proof.composition_poly_root, gpu_proof.composition_poly_root,
            "composition poly root mismatch"
        );
        assert_eq!(
            cpu_proof.fri_layers_merkle_roots, gpu_proof.fri_layers_merkle_roots,
            "FRI merkle roots mismatch"
        );
        assert_eq!(
            cpu_proof.fri_last_value, gpu_proof.fri_last_value,
            "FRI last value mismatch"
        );
        assert_eq!(cpu_proof.nonce, gpu_proof.nonce, "nonce mismatch");

        // Deep poly openings count
        assert_eq!(
            cpu_proof.deep_poly_openings.len(),
            gpu_proof.deep_poly_openings.len(),
            "deep poly openings count mismatch"
        );
    }

    /// Test that the optimized GPU prover produces a valid proof.
    #[test]
    fn gpu_optimized_proof_verifies_with_cpu_verifier() {
        let trace_length = 16;
        let pub_inputs = FibonacciRAPPublicInputs {
            steps: trace_length,
            a0: FpE::one(),
            a1: FpE::one(),
        };
        let proof_options = ProofOptions::default_test_options();
        let mut trace = fibonacci_rap_trace::<F>([FpE::one(), FpE::one()], trace_length);
        let air = FibonacciRAP::new(trace.num_rows(), &pub_inputs, &proof_options);

        let mut prover_transcript = DefaultTranscript::<F>::new(&[]);
        let proof = prove_gpu_optimized(&air, &mut trace, &mut prover_transcript).unwrap();

        let mut verifier_transcript = DefaultTranscript::<F>::new(&[]);
        let result = Verifier::<F, F, _>::verify(&proof, &air, &mut verifier_transcript);
        assert!(
            result,
            "GPU optimized proof must be verified by the CPU verifier"
        );
    }

    /// Test that the optimized GPU prover matches the CPU prover.
    #[test]
    fn gpu_optimized_proof_matches_cpu_proof() {
        let trace_length = 16;
        let pub_inputs = FibonacciRAPPublicInputs {
            steps: trace_length,
            a0: FpE::one(),
            a1: FpE::one(),
        };
        let proof_options = ProofOptions::default_test_options();

        // CPU proof
        let mut cpu_trace = fibonacci_rap_trace::<F>([FpE::one(), FpE::one()], trace_length);
        let air = FibonacciRAP::new(cpu_trace.num_rows(), &pub_inputs, &proof_options);
        let mut cpu_transcript = DefaultTranscript::<F>::new(&[]);
        let cpu_proof =
            Prover::<F, F, _>::prove(&air, &mut cpu_trace, &mut cpu_transcript).unwrap();

        // GPU optimized proof
        let mut gpu_trace = fibonacci_rap_trace::<F>([FpE::one(), FpE::one()], trace_length);
        let air = FibonacciRAP::new(gpu_trace.num_rows(), &pub_inputs, &proof_options);
        let mut gpu_transcript = DefaultTranscript::<F>::new(&[]);
        let gpu_proof = prove_gpu_optimized(&air, &mut gpu_trace, &mut gpu_transcript).unwrap();

        assert_eq!(
            cpu_proof.trace_length, gpu_proof.trace_length,
            "trace_length mismatch"
        );
        assert_eq!(
            cpu_proof.lde_trace_main_merkle_root, gpu_proof.lde_trace_main_merkle_root,
            "main merkle root mismatch"
        );
        assert_eq!(
            cpu_proof.composition_poly_root, gpu_proof.composition_poly_root,
            "composition poly root mismatch"
        );
        assert_eq!(
            cpu_proof.fri_layers_merkle_roots, gpu_proof.fri_layers_merkle_roots,
            "FRI merkle roots mismatch"
        );
        assert_eq!(
            cpu_proof.fri_last_value, gpu_proof.fri_last_value,
            "FRI last value mismatch"
        );
        assert_eq!(cpu_proof.nonce, gpu_proof.nonce, "nonce mismatch");
    }

    // =========================================================================
    // Fp3 extension field test infrastructure
    // =========================================================================

    use std::ops::Div;

    use lambdaworks_math::field::fields::u64_goldilocks_field::Degree3GoldilocksExtensionField;
    use lambdaworks_math::helpers::resize_to_next_power_of_two;
    use stark_platinum_prover::constraints::boundary::{BoundaryConstraint, BoundaryConstraints};
    use stark_platinum_prover::constraints::transition::TransitionConstraint;
    use stark_platinum_prover::context::AirContext;
    use stark_platinum_prover::traits::TransitionEvaluationContext;

    type Fp3 = Degree3GoldilocksExtensionField;
    type Fp3E = FieldElement<Fp3>;

    /// Fibonacci transition constraint for Fp3 extension: a[i+2] - a[i+1] - a[i] = 0.
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
            // Hardcoded for steps=16 → padded trace_length=32
            3 + 32 - 16 - 1
        }

        fn evaluate(
            &self,
            evaluation_context: &TransitionEvaluationContext<F, Fp3>,
            transition_evaluations: &mut [Fp3E],
        ) {
            match evaluation_context {
                TransitionEvaluationContext::Prover { frame, .. } => {
                    let s0 = frame.get_evaluation_step(0);
                    let s1 = frame.get_evaluation_step(1);
                    let s2 = frame.get_evaluation_step(2);
                    // Main elements are &FieldElement<F>; compute in F, embed to Fp3
                    let a0 = s0.get_main_evaluation_element(0, 0);
                    let a1 = s1.get_main_evaluation_element(0, 0);
                    let a2 = s2.get_main_evaluation_element(0, 0);
                    transition_evaluations[0] = (a2 - a1 - a0).to_extension::<Fp3>();
                }
                TransitionEvaluationContext::Verifier { frame, .. } => {
                    let s0 = frame.get_evaluation_step(0);
                    let s1 = frame.get_evaluation_step(1);
                    let s2 = frame.get_evaluation_step(2);
                    // Main elements are &FieldElement<Fp3>
                    let a0 = s0.get_main_evaluation_element(0, 0);
                    let a1 = s1.get_main_evaluation_element(0, 0);
                    let a2 = s2.get_main_evaluation_element(0, 0);
                    transition_evaluations[0] = a2 - a1 - a0;
                }
            }
        }
    }

    /// Permutation constraint for Fp3 extension:
    /// z[i+1] * (b[i] + gamma) - z[i] * (a[i] + gamma) = 0.
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
                    let z_i_plus_one = s1.get_aux_evaluation_element(0, 0);
                    let gamma = &rap_challenges[0];
                    // Main elements are &FieldElement<F>; mixed arithmetic F + Fp3 → Fp3
                    let a_i = s0.get_main_evaluation_element(0, 0);
                    let b_i = s0.get_main_evaluation_element(0, 1);
                    transition_evaluations[1] =
                        z_i_plus_one * (b_i + gamma) - z_i * (a_i + gamma);
                }
                TransitionEvaluationContext::Verifier {
                    frame,
                    rap_challenges,
                    ..
                } => {
                    let s0 = frame.get_evaluation_step(0);
                    let s1 = frame.get_evaluation_step(1);
                    let z_i = s0.get_aux_evaluation_element(0, 0);
                    let z_i_plus_one = s1.get_aux_evaluation_element(0, 0);
                    let gamma = &rap_challenges[0];
                    let a_i = s0.get_main_evaluation_element(0, 0);
                    let b_i = s0.get_main_evaluation_element(0, 1);
                    transition_evaluations[1] =
                        z_i_plus_one * (b_i + gamma) - z_i * (a_i + gamma);
                }
            }
        }
    }

    /// FibonacciRAP AIR with Fp3 extension field.
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

        fn build_auxiliary_trace(
            &self,
            trace: &mut TraceTable<F, Fp3>,
            challenges: &[Fp3E],
        ) {
            let main_segment_cols = trace.columns_main();
            let not_perm = &main_segment_cols[0];
            let perm = &main_segment_cols[1];
            let gamma = &challenges[0];
            let trace_len = trace.num_rows();

            let mut aux_col: Vec<Fp3E> = Vec::new();
            for i in 0..trace_len {
                if i == 0 {
                    aux_col.push(Fp3E::one());
                } else {
                    let z_i = &aux_col[i - 1];
                    // Mixed arithmetic: FieldElement<F> + &FieldElement<Fp3> → FieldElement<Fp3>
                    let n_p_term = not_perm[i - 1] + gamma;
                    let p_term = &perm[i - 1] + gamma;
                    aux_col.push(z_i * n_p_term.div(p_term).unwrap());
                }
            }
            for (i, aux_elem) in aux_col.iter().enumerate().take(trace.num_rows()) {
                trace.set_aux(i, 0, aux_elem.clone());
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

        fn boundary_constraints(
            &self,
            _rap_challenges: &[Fp3E],
        ) -> BoundaryConstraints<Fp3> {
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

    /// Generate a Fibonacci RAP trace with Fp3 extension field.
    ///
    /// Main columns are in base field (same computation as base-field version).
    /// Aux column is initialized to Fp3 zeros (filled by build_auxiliary_trace).
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

    /// Test that an Fp3 proof generated via CPU prover verifies correctly.
    #[test]
    fn gpu_fp3_proof_verifies() {
        let trace_length = 16;
        let pub_inputs = FibonacciRAPPublicInputs {
            steps: trace_length,
            a0: FpE::one(),
            a1: FpE::one(),
        };
        let proof_options = ProofOptions::default_test_options();
        let mut trace = fibonacci_rap_trace_fp3([FpE::one(), FpE::one()], trace_length);
        let air = FibonacciRAPFp3::new(trace.num_rows(), &pub_inputs, &proof_options);

        // Prove with Fp3 extension
        let mut prover_transcript = DefaultTranscript::<Fp3>::new(&[]);
        let proof = prove_gpu_fp3(&air, &mut trace, &mut prover_transcript).unwrap();

        // Verify with CPU verifier
        let mut verifier_transcript = DefaultTranscript::<Fp3>::new(&[]);
        let result = Verifier::<F, Fp3, _>::verify(&proof, &air, &mut verifier_transcript);
        assert!(result, "Fp3 proof must be verified by the CPU verifier");
    }
}
