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
}
