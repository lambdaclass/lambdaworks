//! GPU Phase 1: RAP (trace interpolation + LDE + commit).
//!
//! This module mirrors `round_1_randomized_air_with_preprocessing` from the CPU
//! STARK prover but uses Metal GPU FFT for interpolation and LDE evaluation.

use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::{IsFFTField, IsField, IsSubFieldOf};
use lambdaworks_math::polynomial::Polynomial;
use lambdaworks_math::traits::AsBytes;

use lambdaworks_crypto::fiat_shamir::is_transcript::IsStarkTranscript;

use stark_platinum_prover::config::{BatchedMerkleTree, Commitment};
use stark_platinum_prover::domain::Domain;
use stark_platinum_prover::prover::ProvingError;
use stark_platinum_prover::trace::{columns2rows_bit_reversed, TraceTable};
use stark_platinum_prover::traits::AIR;

#[cfg(all(target_os = "macos", feature = "metal"))]
use crate::metal::fft::{gpu_evaluate_offset_fft, gpu_interpolate_fft};
use crate::metal::merkle::cpu_batch_commit;
#[cfg(all(target_os = "macos", feature = "metal"))]
use crate::metal::state::StarkMetalState;

#[cfg(all(target_os = "macos", feature = "metal"))]
use crate::metal::merkle::{gpu_batch_commit_goldilocks, GpuKeccakMerkleState};
#[cfg(all(target_os = "macos", feature = "metal"))]
use lambdaworks_math::field::fields::u64_goldilocks_field::Goldilocks64Field;

/// Result of GPU Phase 1 (RAP round).
///
/// This is the GPU equivalent of `Round1<Field, FieldExtension>` from the CPU
/// prover. We define our own struct because `Round1`'s fields are `pub(crate)`
/// in the STARK crate and cannot be constructed from an external crate.
///
/// All data needed by subsequent prover phases (constraint evaluation, FRI) is
/// stored here.
pub struct GpuRound1Result<F: IsField>
where
    FieldElement<F>: AsBytes,
{
    /// Polynomial coefficients from interpolating main trace columns.
    pub main_trace_polys: Vec<Polynomial<FieldElement<F>>>,
    /// LDE evaluations of main trace (column-major: one Vec per column).
    pub main_lde_evaluations: Vec<Vec<FieldElement<F>>>,
    /// Merkle tree for main trace LDE.
    pub main_merkle_tree: BatchedMerkleTree<F>,
    /// Merkle root for main trace.
    pub main_merkle_root: Commitment,
    /// Polynomial coefficients from interpolating auxiliary trace columns (if RAP).
    pub aux_trace_polys: Vec<Polynomial<FieldElement<F>>>,
    /// LDE evaluations of auxiliary trace (column-major).
    pub aux_lde_evaluations: Vec<Vec<FieldElement<F>>>,
    /// Merkle tree for auxiliary trace LDE (if RAP).
    pub aux_merkle_tree: Option<BatchedMerkleTree<F>>,
    /// Merkle root for auxiliary trace.
    pub aux_merkle_root: Option<Commitment>,
    /// RAP challenges sampled from the transcript.
    pub rap_challenges: Vec<FieldElement<F>>,
}

/// Executes GPU Phase 1 of the STARK prover: RAP (trace interpolation + LDE + commit).
///
/// This mirrors `round_1_randomized_air_with_preprocessing` from the CPU prover:
///
/// 1. Extract columns from the main trace table
/// 2. Interpolate each column via GPU FFT to get polynomial coefficients
/// 3. Evaluate each polynomial on the LDE coset domain via GPU FFT
/// 4. Bit-reverse + transpose evaluations (CPU) for Merkle commitment layout
/// 5. Build Merkle tree and commit (CPU, Keccak256)
/// 6. Append commitment root to the transcript
/// 7. Sample RAP challenges from the transcript
/// 8. If the AIR has trace interaction (auxiliary trace): build aux trace, repeat 1-6
///
/// # Type Parameters
///
/// - `F`: The base field (must equal the extension field for our GPU prover)
/// - `A`: An AIR whose `Field` and `FieldExtension` are both `F`
///
/// # Errors
///
/// Returns `ProvingError` if FFT, Merkle commitment, or transcript operations fail.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn gpu_round_1<F, A>(
    air: &A,
    trace: &mut TraceTable<F, F>,
    _domain: &Domain<F>,
    transcript: &mut impl IsStarkTranscript<F, F>,
    state: &StarkMetalState,
) -> Result<GpuRound1Result<F>, ProvingError>
where
    F: IsFFTField + IsSubFieldOf<F> + Send + Sync,
    F::BaseType: Copy,
    FieldElement<F>: AsBytes + Sync + Send,
    A: AIR<Field = F, FieldExtension = F>,
{
    // --- Main trace ---

    // Step 1: Extract main trace columns
    let main_columns = trace.columns_main();

    // Step 2: Interpolate each column on GPU to get polynomial coefficients
    let main_trace_polys = interpolate_columns_gpu(&main_columns, state)?;

    // Step 3: Evaluate each polynomial on the LDE coset domain using GPU
    let blowup_factor = air.blowup_factor() as usize;
    let coset_offset = air.coset_offset();
    let main_lde_evaluations =
        evaluate_polys_on_lde_gpu(&main_trace_polys, blowup_factor, &coset_offset, state)?;

    // Step 4: Bit-reverse + transpose for Merkle commitment layout (CPU)
    let main_permuted_rows = columns2rows_bit_reversed(&main_lde_evaluations);

    // Step 5: Build Merkle tree and commit (CPU)
    let (main_merkle_tree, main_merkle_root) =
        cpu_batch_commit(&main_permuted_rows).ok_or(ProvingError::EmptyCommitment)?;

    // Step 6: Append root to transcript
    transcript.append_bytes(&main_merkle_root);

    // Step 7: Sample RAP challenges
    let rap_challenges = air.build_rap_challenges(transcript);

    // --- Auxiliary trace (if RAP) ---
    let (aux_trace_polys, aux_lde_evaluations, aux_merkle_tree, aux_merkle_root) =
        if air.has_trace_interaction() {
            // Build auxiliary trace columns based on RAP challenges
            air.build_auxiliary_trace(trace, &rap_challenges);

            // Extract auxiliary trace columns
            let aux_columns = trace.columns_aux();

            // Interpolate auxiliary columns on GPU
            let aux_polys = interpolate_columns_gpu(&aux_columns, state)?;

            // Evaluate auxiliary polynomials on LDE domain using GPU
            let aux_lde_evals =
                evaluate_polys_on_lde_gpu(&aux_polys, blowup_factor, &coset_offset, state)?;

            // Bit-reverse + transpose for Merkle commitment
            let aux_permuted_rows = columns2rows_bit_reversed(&aux_lde_evals);

            // Build Merkle tree and commit
            let (aux_tree, aux_root) =
                cpu_batch_commit(&aux_permuted_rows).ok_or(ProvingError::EmptyCommitment)?;

            // Append auxiliary root to transcript
            transcript.append_bytes(&aux_root);

            (aux_polys, aux_lde_evals, Some(aux_tree), Some(aux_root))
        } else {
            (Vec::new(), Vec::new(), None, None)
        };

    Ok(GpuRound1Result {
        main_trace_polys,
        main_lde_evaluations,
        main_merkle_tree,
        main_merkle_root,
        aux_trace_polys,
        aux_lde_evaluations,
        aux_merkle_tree,
        aux_merkle_root,
        rap_challenges,
    })
}

/// GPU-optimized Phase 1 for Goldilocks field with GPU Merkle commit.
///
/// This is a concrete version of [`gpu_round_1`] that uses the GPU Keccak256
/// shader for Merkle tree construction instead of CPU hashing.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn gpu_round_1_goldilocks<A>(
    air: &A,
    trace: &mut TraceTable<Goldilocks64Field, Goldilocks64Field>,
    _domain: &Domain<Goldilocks64Field>,
    transcript: &mut impl IsStarkTranscript<Goldilocks64Field, Goldilocks64Field>,
    state: &StarkMetalState,
    keccak_state: &GpuKeccakMerkleState,
) -> Result<GpuRound1Result<Goldilocks64Field>, ProvingError>
where
    A: AIR<Field = Goldilocks64Field, FieldExtension = Goldilocks64Field>,
{
    // --- Main trace ---

    let main_columns = trace.columns_main();
    let main_trace_polys = interpolate_columns_gpu(&main_columns, state)?;

    let blowup_factor = air.blowup_factor() as usize;
    let coset_offset = air.coset_offset();
    let main_lde_evaluations =
        evaluate_polys_on_lde_gpu(&main_trace_polys, blowup_factor, &coset_offset, state)?;

    // GPU Merkle commit: transpose + bit-reverse + hash all on GPU
    let (main_merkle_tree, main_merkle_root) =
        gpu_batch_commit_goldilocks(&main_lde_evaluations, keccak_state)
            .ok_or(ProvingError::EmptyCommitment)?;

    transcript.append_bytes(&main_merkle_root);
    let rap_challenges = air.build_rap_challenges(transcript);

    // --- Auxiliary trace (if RAP) ---
    let (aux_trace_polys, aux_lde_evaluations, aux_merkle_tree, aux_merkle_root) =
        if air.has_trace_interaction() {
            air.build_auxiliary_trace(trace, &rap_challenges);
            let aux_columns = trace.columns_aux();
            let aux_polys = interpolate_columns_gpu(&aux_columns, state)?;
            let aux_lde_evals =
                evaluate_polys_on_lde_gpu(&aux_polys, blowup_factor, &coset_offset, state)?;

            // GPU Merkle commit for auxiliary trace
            let (aux_tree, aux_root) =
                gpu_batch_commit_goldilocks(&aux_lde_evals, keccak_state)
                    .ok_or(ProvingError::EmptyCommitment)?;

            transcript.append_bytes(&aux_root);
            (aux_polys, aux_lde_evals, Some(aux_tree), Some(aux_root))
        } else {
            (Vec::new(), Vec::new(), None, None)
        };

    Ok(GpuRound1Result {
        main_trace_polys,
        main_lde_evaluations,
        main_merkle_tree,
        main_merkle_root,
        aux_trace_polys,
        aux_lde_evaluations,
        aux_merkle_tree,
        aux_merkle_root,
        rap_challenges,
    })
}

/// Interpolates a set of evaluation columns into polynomials using GPU FFT.
///
/// Each column is a vector of field element evaluations at roots of unity.
/// Returns a vector of polynomials (one per column) whose coefficients are
/// computed via inverse FFT on the Metal GPU.
#[cfg(all(target_os = "macos", feature = "metal"))]
fn interpolate_columns_gpu<F>(
    columns: &[Vec<FieldElement<F>>],
    state: &StarkMetalState,
) -> Result<Vec<Polynomial<FieldElement<F>>>, ProvingError>
where
    F: IsFFTField + IsSubFieldOf<F>,
    F::BaseType: Copy,
{
    columns
        .iter()
        .map(|col| {
            let coeffs = gpu_interpolate_fft::<F>(col, state.inner())
                .map_err(|e| ProvingError::FieldOperationError(format!("GPU FFT error: {e}")))?;
            Ok(Polynomial::new(&coeffs))
        })
        .collect()
}

/// Evaluates polynomials on the LDE coset domain using GPU FFT.
///
/// For each polynomial, this computes evaluations at `{offset * w^i}` where
/// `w` is a primitive root of unity of order `len * blowup_factor`.
/// Returns a vector of evaluation vectors (one per polynomial, column-major).
#[cfg(all(target_os = "macos", feature = "metal"))]
fn evaluate_polys_on_lde_gpu<F>(
    polys: &[Polynomial<FieldElement<F>>],
    blowup_factor: usize,
    offset: &FieldElement<F>,
    state: &StarkMetalState,
) -> Result<Vec<Vec<FieldElement<F>>>, ProvingError>
where
    F: IsFFTField + IsSubFieldOf<F>,
    F::BaseType: Copy,
{
    polys
        .iter()
        .map(|poly| {
            gpu_evaluate_offset_fft::<F>(poly.coefficients(), blowup_factor, offset, state.inner())
                .map_err(|e| ProvingError::FieldOperationError(format!("GPU LDE FFT error: {e}")))
        })
        .collect()
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

    type F = Goldilocks64Field;
    type FpE = FieldElement<F>;

    #[test]
    fn gpu_round_1_fibonacci_rap_goldilocks() {
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
        let result = gpu_round_1(&air, &mut trace, &domain, &mut transcript, &state).unwrap();

        // Fibonacci RAP has 2 main columns
        assert_eq!(result.main_trace_polys.len(), 2);
        // LDE should have blowup_factor * trace_length evaluations per column
        let expected_lde_len = trace.num_rows() * proof_options.blowup_factor as usize;
        assert_eq!(result.main_lde_evaluations[0].len(), expected_lde_len);
        // Merkle root should be non-zero
        assert_ne!(result.main_merkle_root, [0u8; 32]);
        // Fibonacci RAP has auxiliary trace (permutation argument): 1 aux column
        assert!(result.aux_merkle_root.is_some());
        assert_eq!(result.aux_trace_polys.len(), 1);
        assert!(!result.rap_challenges.is_empty());
        // Aux LDE evaluations should match expected size
        assert_eq!(result.aux_lde_evaluations[0].len(), expected_lde_len);
    }

    /// Verify that GPU interpolation + LDE produces the same results as the CPU path.
    #[test]
    fn gpu_round_1_matches_cpu_main_trace() {
        let trace_length = 16;
        let pub_inputs = FibonacciRAPPublicInputs {
            steps: trace_length - 1,
            a0: FpE::one(),
            a1: FpE::one(),
        };
        let proof_options = ProofOptions::default_test_options();
        let trace = fibonacci_rap_trace::<F>([FpE::one(), FpE::one()], trace_length);
        let air = FibonacciRAP::new(trace.num_rows(), &pub_inputs, &proof_options);
        let state = StarkMetalState::new().unwrap();

        // --- CPU path ---
        let main_columns = trace.columns_main();
        let blowup_factor = air.blowup_factor() as usize;
        let coset_offset = air.coset_offset();

        let cpu_polys: Vec<Polynomial<FieldElement<F>>> = main_columns
            .iter()
            .map(|col| Polynomial::interpolate_fft::<F>(col).unwrap())
            .collect();

        let cpu_lde_evals: Vec<Vec<FieldElement<F>>> = cpu_polys
            .iter()
            .map(|poly| {
                Polynomial::evaluate_offset_fft::<F>(poly, blowup_factor, None, &coset_offset)
                    .unwrap()
            })
            .collect();

        // --- GPU path ---
        let gpu_polys = interpolate_columns_gpu(&main_columns, &state).unwrap();
        let gpu_lde_evals =
            evaluate_polys_on_lde_gpu(&gpu_polys, blowup_factor, &coset_offset, &state).unwrap();

        // Compare polynomials
        assert_eq!(cpu_polys.len(), gpu_polys.len());
        for (i, (cpu_poly, gpu_poly)) in cpu_polys.iter().zip(&gpu_polys).enumerate() {
            assert_eq!(
                cpu_poly.coefficients().len(),
                gpu_poly.coefficients().len(),
                "Column {i}: coefficient count mismatch"
            );
            for (j, (c, g)) in cpu_poly
                .coefficients()
                .iter()
                .zip(gpu_poly.coefficients())
                .enumerate()
            {
                assert_eq!(c, g, "Column {i}, coeff {j}: value mismatch");
            }
        }

        // Compare LDE evaluations
        assert_eq!(cpu_lde_evals.len(), gpu_lde_evals.len());
        for (i, (cpu_col, gpu_col)) in cpu_lde_evals.iter().zip(&gpu_lde_evals).enumerate() {
            assert_eq!(
                cpu_col.len(),
                gpu_col.len(),
                "Column {i}: LDE eval count mismatch"
            );
            for (j, (c, g)) in cpu_col.iter().zip(gpu_col).enumerate() {
                assert_eq!(c, g, "Column {i}, eval {j}: LDE value mismatch");
            }
        }
    }
}
