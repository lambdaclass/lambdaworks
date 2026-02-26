//! GPU Phase 1: RAP (trace interpolation + LDE + commit).

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
use crate::metal::fft::{
    gpu_evaluate_offset_fft, gpu_evaluate_offset_fft_to_buffers_batch, gpu_interpolate_fft,
    gpu_lde_from_evaluations, CosetShiftState,
};
use crate::metal::merkle::cpu_batch_commit;
#[cfg(all(target_os = "macos", feature = "metal"))]
use crate::metal::state::StarkMetalState;
#[cfg(all(target_os = "macos", feature = "metal"))]
use crate::metal::to_raw_u64;

#[cfg(all(target_os = "macos", feature = "metal"))]
use crate::metal::merkle::{gpu_batch_commit_from_column_buffers, GpuMerkleState};
#[cfg(all(target_os = "macos", feature = "metal"))]
use lambdaworks_math::field::fields::u64_goldilocks_field::Goldilocks64Field;

/// Result of GPU Phase 1 (RAP round).
///
/// GPU equivalent of `Round1<Field, FieldExtension>` from the CPU prover.
/// We define our own struct because `Round1`'s fields are `pub(crate)`.
pub struct GpuRound1Result<F: IsField>
where
    FieldElement<F>: AsBytes,
{
    pub main_trace_polys: Vec<Polynomial<FieldElement<F>>>,
    pub main_lde_evaluations: Vec<Vec<FieldElement<F>>>,
    pub main_merkle_tree: BatchedMerkleTree<F>,
    pub main_merkle_root: Commitment,
    pub aux_trace_polys: Vec<Polynomial<FieldElement<F>>>,
    pub aux_lde_evaluations: Vec<Vec<FieldElement<F>>>,
    pub aux_merkle_tree: Option<BatchedMerkleTree<F>>,
    pub aux_merkle_root: Option<Commitment>,
    pub rap_challenges: Vec<FieldElement<F>>,
    /// Original main trace evaluations on roots-of-unity domain (for barycentric OOD).
    pub main_trace_evals: Vec<Vec<FieldElement<F>>>,
    /// Original aux trace evaluations on roots-of-unity domain (for barycentric OOD).
    pub aux_trace_evals: Vec<Vec<FieldElement<F>>>,
    /// Retained GPU buffers for main trace LDE (avoids re-upload in DEEP composition).
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub main_lde_gpu_buffers: Option<Vec<metal::Buffer>>,
    /// Retained GPU buffers for auxiliary trace LDE.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub aux_lde_gpu_buffers: Option<Vec<metal::Buffer>>,
    /// Retained GPU buffer for LDE coset points (canonical u64 values).
    /// Computed once and reused in Phase 2 and Phase 4.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub lde_coset_gpu_buffer: Option<metal::Buffer>,
}

/// Generic GPU Phase 1: interpolate + LDE + Merkle commit via GPU FFT.
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
    let main_columns = trace.columns_main();
    let main_trace_polys = interpolate_columns_gpu(&main_columns, state)?;

    let blowup_factor = air.blowup_factor() as usize;
    let coset_offset = air.coset_offset();
    let main_lde_evaluations =
        evaluate_polys_on_lde_gpu(&main_trace_polys, blowup_factor, &coset_offset, state)?;

    let main_permuted_rows = columns2rows_bit_reversed(&main_lde_evaluations);
    let (main_merkle_tree, main_merkle_root) =
        cpu_batch_commit(&main_permuted_rows).ok_or(ProvingError::EmptyCommitment)?;

    transcript.append_bytes(&main_merkle_root);
    let rap_challenges = air.build_rap_challenges(transcript);

    let (aux_trace_polys, aux_lde_evaluations, aux_merkle_tree, aux_merkle_root) =
        if air.has_aux_trace() {
            air.build_auxiliary_trace(trace, &rap_challenges)?;
            let aux_columns = trace.columns_aux();
            let aux_polys = interpolate_columns_gpu(&aux_columns, state)?;
            let aux_lde_evals =
                evaluate_polys_on_lde_gpu(&aux_polys, blowup_factor, &coset_offset, state)?;
            let aux_permuted_rows = columns2rows_bit_reversed(&aux_lde_evals);
            let (aux_tree, aux_root) =
                cpu_batch_commit(&aux_permuted_rows).ok_or(ProvingError::EmptyCommitment)?;
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
        main_trace_evals: Vec::new(),
        aux_trace_evals: Vec::new(),
        main_lde_gpu_buffers: None,
        aux_lde_gpu_buffers: None,
        lde_coset_gpu_buffer: None,
    })
}

/// Goldilocks-optimized Phase 1 with fused LDE pipeline and GPU Merkle commit.
///
/// Uses buffer pipeline: IFFT + div-N + coset shift + FFT stay on GPU.
/// No CPU readback until Phase 3 (barycentric OOD).
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn gpu_round_1_goldilocks<A>(
    air: &A,
    trace: &mut TraceTable<Goldilocks64Field, Goldilocks64Field>,
    _domain: &Domain<Goldilocks64Field>,
    transcript: &mut impl IsStarkTranscript<Goldilocks64Field, Goldilocks64Field>,
    state: &StarkMetalState,
    keccak_state: &GpuMerkleState,
    coset_state: &CosetShiftState,
) -> Result<GpuRound1Result<Goldilocks64Field>, ProvingError>
where
    A: AIR<Field = Goldilocks64Field, FieldExtension = Goldilocks64Field>,
{
    let main_columns = trace.columns_main();
    let main_trace_evals: Vec<Vec<FieldElement<Goldilocks64Field>>> = main_columns.clone();

    let blowup_factor = air.blowup_factor() as usize;
    let coset_offset = air.coset_offset();

    // Fused IFFT + div-N + coset shift + FFT per column (no intermediate coefficient storage)
    let (main_lde_buffers, main_lde_domain_size) = gpu_lde_from_evaluations(
        &main_columns,
        blowup_factor,
        &coset_offset,
        coset_state,
        state.inner(),
    )
    .map_err(|e| ProvingError::FieldOperationError(format!("GPU main LDE error: {e}")))?;

    // GPU Merkle commit directly from FFT output buffers
    let buffer_refs: Vec<&metal::Buffer> = main_lde_buffers.iter().collect();
    let (main_merkle_tree, main_merkle_root) =
        gpu_batch_commit_from_column_buffers(&buffer_refs, main_lde_domain_size, keccak_state)
            .ok_or(ProvingError::EmptyCommitment)?;

    transcript.append_bytes(&main_merkle_root);
    let rap_challenges = air.build_rap_challenges(transcript);

    let (aux_merkle_tree, aux_merkle_root, aux_gpu_bufs, aux_trace_evals) = if air.has_aux_trace() {
        air.build_auxiliary_trace(trace, &rap_challenges)?;
        let aux_columns = trace.columns_aux();
        let aux_evals: Vec<Vec<FieldElement<Goldilocks64Field>>> = aux_columns.clone();

        let (aux_lde_buffers, aux_lde_domain_size) = gpu_lde_from_evaluations(
            &aux_columns,
            blowup_factor,
            &coset_offset,
            coset_state,
            state.inner(),
        )
        .map_err(|e| ProvingError::FieldOperationError(format!("GPU aux LDE error: {e}")))?;

        let buf_refs: Vec<&metal::Buffer> = aux_lde_buffers.iter().collect();
        let (aux_tree, aux_root) =
            gpu_batch_commit_from_column_buffers(&buf_refs, aux_lde_domain_size, keccak_state)
                .ok_or(ProvingError::EmptyCommitment)?;

        transcript.append_bytes(&aux_root);
        (
            Some(aux_tree),
            Some(aux_root),
            Some(aux_lde_buffers),
            aux_evals,
        )
    } else {
        (None, None, None, Vec::new())
    };

    // Pre-compute LDE coset points as a GPU buffer once (reused by Phase 2 and Phase 4).
    let coset_raw: Vec<u64> = to_raw_u64(&_domain.lde_roots_of_unity_coset);
    let lde_coset_gpu_buffer = state.inner().device.new_buffer_with_data(
        coset_raw.as_ptr().cast(),
        (coset_raw.len() * std::mem::size_of::<u64>()) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    Ok(GpuRound1Result {
        main_trace_polys: Vec::new(),
        main_lde_evaluations: Vec::new(),
        main_merkle_tree,
        main_merkle_root,
        aux_trace_polys: Vec::new(),
        aux_lde_evaluations: Vec::new(),
        aux_merkle_tree,
        aux_merkle_root,
        rap_challenges,
        main_trace_evals,
        aux_trace_evals,
        main_lde_gpu_buffers: Some(main_lde_buffers),
        aux_lde_gpu_buffers: aux_gpu_bufs,
        lde_coset_gpu_buffer: Some(lde_coset_gpu_buffer),
    })
}

/// GPU-accelerated Phase 1 for Fp3 extension field proofs.
///
/// Main trace (base field): GPU FFT + GPU Keccak Merkle.
/// Aux trace (Fp3): CPU FFT + CPU Merkle commit.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn gpu_round_1_fp3<A>(
    air: &A,
    trace: &mut TraceTable<
        Goldilocks64Field,
        lambdaworks_math::field::fields::u64_goldilocks_field::Degree3GoldilocksExtensionField,
    >,
    domain: &Domain<Goldilocks64Field>,
    transcript: &mut impl IsStarkTranscript<
        lambdaworks_math::field::fields::u64_goldilocks_field::Degree3GoldilocksExtensionField,
        Goldilocks64Field,
    >,
    state: &StarkMetalState,
    keccak_state: &GpuMerkleState,
) -> Result<crate::metal::phases::fp3_types::GpuRound1ResultFp3, ProvingError>
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

    let main_columns = trace.columns_main();
    let main_trace_polys = interpolate_columns_gpu(&main_columns, state)?;

    let blowup_factor = air.blowup_factor() as usize;
    let coset_offset = air.coset_offset();

    let (main_lde_buffers, main_lde_domain_size) = evaluate_polys_on_lde_gpu_to_buffers(
        &main_trace_polys,
        blowup_factor,
        &coset_offset,
        state,
    )?;

    // Reconstruct CPU vecs from GPU buffers via UMA pointer (no intermediate Vec<u64>)
    let main_lde_evaluations: Vec<Vec<FieldElement<Goldilocks64Field>>> = main_lde_buffers
        .iter()
        .map(|buf| {
            (0..main_lde_domain_size)
                .map(|i| read_element_from_buffer(buf, i))
                .collect()
        })
        .collect();

    let buffer_refs: Vec<&metal::Buffer> = main_lde_buffers.iter().collect();
    let (main_merkle_tree, main_merkle_root) =
        gpu_batch_commit_from_column_buffers(&buffer_refs, main_lde_domain_size, keccak_state)
            .ok_or(ProvingError::EmptyCommitment)?;

    transcript.append_bytes(&main_merkle_root);
    let rap_challenges = air.build_rap_challenges(transcript);

    let (aux_trace_polys, aux_lde_evaluations, aux_merkle_tree, aux_merkle_root) =
        if air.has_aux_trace() {
            air.build_auxiliary_trace(trace, &rap_challenges)?;

            let aux_columns = trace.columns_aux();
            let aux_polys: Vec<Polynomial<Fp3E>> = aux_columns
                .iter()
                .map(|col| Polynomial::interpolate_fft::<F>(col))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| ProvingError::FieldOperationError(format!("CPU aux FFT: {e}")))?;

            let aux_lde_evals: Vec<Vec<Fp3E>> = aux_polys
                .iter()
                .map(|poly| {
                    Polynomial::evaluate_offset_fft::<F>(
                        poly,
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
                .map_err(|e| ProvingError::FieldOperationError(format!("CPU aux LDE FFT: {e}")))?;

            let aux_permuted_rows = columns2rows_bit_reversed(&aux_lde_evals);
            let (aux_tree, aux_root) =
                cpu_batch_commit(&aux_permuted_rows).ok_or(ProvingError::EmptyCommitment)?;
            transcript.append_bytes(&aux_root);
            (aux_polys, aux_lde_evals, Some(aux_tree), Some(aux_root))
        } else {
            (Vec::new(), Vec::new(), None, None)
        };

    Ok(crate::metal::phases::fp3_types::GpuRound1ResultFp3 {
        main_trace_polys,
        aux_trace_polys,
        main_lde_evaluations,
        aux_lde_evaluations,
        main_merkle_tree,
        main_merkle_root,
        aux_merkle_tree,
        aux_merkle_root,
        rap_challenges,
    })
}

/// Interpolates evaluation columns into polynomials using GPU FFT.
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

/// Evaluates polynomials on the LDE coset domain, returning GPU Buffers.
///
/// Keeps FFT results as Metal Buffers without CPU readback. Twiddle factors
/// are generated once and reused for all columns (batch FFT).
#[cfg(all(target_os = "macos", feature = "metal"))]
fn evaluate_polys_on_lde_gpu_to_buffers(
    polys: &[Polynomial<FieldElement<Goldilocks64Field>>],
    blowup_factor: usize,
    offset: &FieldElement<Goldilocks64Field>,
    state: &StarkMetalState,
) -> Result<(Vec<metal::Buffer>, usize), ProvingError> {
    if polys.is_empty() {
        return Ok((Vec::new(), 0));
    }

    let coeff_slices: Vec<&[FieldElement<Goldilocks64Field>]> =
        polys.iter().map(|p| p.coefficients()).collect();

    let buffer_results = gpu_evaluate_offset_fft_to_buffers_batch(
        &coeff_slices,
        blowup_factor,
        offset,
        state.inner(),
    )
    .map_err(|e| ProvingError::FieldOperationError(format!("GPU LDE FFT error: {e}")))?;

    let mut gpu_buffers = Vec::with_capacity(polys.len());
    let mut domain_size = 0;

    for (buffer, ds) in buffer_results {
        domain_size = ds;
        gpu_buffers.push(buffer);
    }

    Ok((gpu_buffers, domain_size))
}

/// Read a single Goldilocks field element from a Metal buffer at a given index.
///
/// On Apple Silicon UMA, `StorageModeShared` buffers share physical memory --
/// this is a direct pointer dereference with no DMA transfer.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn read_element_from_buffer(
    buffer: &metal::Buffer,
    index: usize,
) -> FieldElement<Goldilocks64Field> {
    let ptr = buffer.contents() as *const u64;
    let len = buffer.length() as usize / std::mem::size_of::<u64>();
    assert!(
        index < len,
        "buffer index {index} out of bounds (len {len})"
    );
    let raw = unsafe { *ptr.add(index) };
    FieldElement::from_raw(raw)
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

        assert_eq!(result.main_trace_polys.len(), 2);
        let expected_lde_len = trace.num_rows() * proof_options.blowup_factor as usize;
        assert_eq!(result.main_lde_evaluations[0].len(), expected_lde_len);
        assert_ne!(result.main_merkle_root, [0u8; 32]);
        assert!(result.aux_merkle_root.is_some());
        assert_eq!(result.aux_trace_polys.len(), 1);
        assert!(!result.rap_challenges.is_empty());
        assert_eq!(result.aux_lde_evaluations[0].len(), expected_lde_len);
    }

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

        let gpu_polys = interpolate_columns_gpu(&main_columns, &state).unwrap();
        let gpu_lde_evals =
            evaluate_polys_on_lde_gpu(&gpu_polys, blowup_factor, &coset_offset, &state).unwrap();

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
