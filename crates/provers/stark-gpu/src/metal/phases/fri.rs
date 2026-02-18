//! GPU Phase 4: DEEP Composition Polynomial + FRI + Queries.
//!
//! This module mirrors `round_4_compute_and_run_fri_on_the_deep_composition_polynomial`
//! from the CPU STARK prover. It computes the DEEP composition polynomial, runs FRI
//! commit and query phases, performs grinding, and extracts Merkle opening proofs.
//!
//! All operations currently run on CPU. GPU acceleration of FRI folding and DEEP
//! composition comes in later tasks.

use lambdaworks_math::fft::cpu::bit_reversing::reverse_index;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::{IsFFTField, IsField, IsSubFieldOf};
use lambdaworks_math::polynomial::Polynomial;
use lambdaworks_math::traits::AsBytes;

use lambdaworks_crypto::fiat_shamir::is_transcript::IsStarkTranscript;

use stark_platinum_prover::config::{BatchedMerkleTree, Commitment};
use stark_platinum_prover::domain::Domain;
use stark_platinum_prover::fri;
use stark_platinum_prover::fri::fri_decommit::FriDecommitment;
use stark_platinum_prover::grinding;
use stark_platinum_prover::proof::stark::{DeepPolynomialOpening, PolynomialOpenings};
use stark_platinum_prover::prover::ProvingError;
use stark_platinum_prover::table::Table;
use stark_platinum_prover::traits::AIR;

use crate::metal::phases::composition::GpuRound2Result;
use crate::metal::phases::ood::GpuRound3Result;
use crate::metal::phases::rap::GpuRound1Result;

#[cfg(all(target_os = "macos", feature = "metal"))]
use crate::metal::deep_composition::{gpu_compute_deep_composition_poly, DeepCompositionState};
#[cfg(all(target_os = "macos", feature = "metal"))]
use crate::metal::state::StarkMetalState;
#[cfg(all(target_os = "macos", feature = "metal"))]
use lambdaworks_math::field::fields::u64_goldilocks_field::Goldilocks64Field;

/// Type alias matching the CPU prover's deep polynomial openings type.
pub type DeepPolynomialOpenings<F, E> = Vec<DeepPolynomialOpening<F, E>>;

/// Result of GPU Phase 4 (FRI round).
///
/// This is the GPU equivalent of `Round4<Field, FieldExtension>` from the CPU prover.
/// Contains the FRI last value, layer commitments, query decommitments, deep polynomial
/// openings (Merkle proofs for trace and composition poly evaluations), and the grinding
/// nonce.
pub struct GpuRound4Result<F: IsField>
where
    FieldElement<F>: AsBytes,
{
    /// The final constant value from FRI folding.
    pub fri_last_value: FieldElement<F>,
    /// Merkle roots of each FRI inner layer.
    pub fri_layers_merkle_roots: Vec<Commitment>,
    /// Merkle opening proofs for trace and composition polynomial evaluations at query points.
    pub deep_poly_openings: DeepPolynomialOpenings<F, F>,
    /// FRI query decommitments (symmetric evaluations + auth paths for each layer).
    pub query_list: Vec<FriDecommitment<F>>,
    /// Grinding nonce (None if grinding_factor == 0).
    pub nonce: Option<u64>,
}

/// Executes GPU Phase 4 of the STARK prover: DEEP composition + FRI + queries.
///
/// This mirrors `round_4_compute_and_run_fri_on_the_deep_composition_polynomial` from
/// the CPU prover:
///
/// 1. Sample gamma from transcript and compute deep composition coefficients
/// 2. Compute DEEP composition polynomial (CPU)
/// 3. Run FRI commit phase (iterative folding + Merkle commits)
/// 4. Grinding: find nonce if security_bits > 0
/// 5. Sample query indexes
/// 6. Run FRI query phase
/// 7. Extract FRI Merkle roots from layers
/// 8. Open deep composition poly (get Merkle proofs for trace and composition poly)
/// 9. Assemble and return GpuRound4Result
///
/// # Type Parameters
///
/// - `F`: The base field (must equal the extension field for our GPU prover)
/// - `A`: An AIR whose `Field` and `FieldExtension` are both `F`
///
/// # Errors
///
/// Returns `ProvingError` if FRI, grinding, or Merkle operations fail.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn gpu_round_4<F, A>(
    air: &A,
    domain: &Domain<F>,
    round_1_result: &GpuRound1Result<F>,
    round_2_result: &GpuRound2Result<F>,
    round_3_result: &GpuRound3Result<F>,
    transcript: &mut impl IsStarkTranscript<F, F>,
) -> Result<GpuRound4Result<F>, ProvingError>
where
    F: IsFFTField + IsSubFieldOf<F> + Send + Sync,
    F::BaseType: Copy,
    FieldElement<F>: AsBytes + Sync + Send,
    A: AIR<Field = F, FieldExtension = F>,
{
    // Step 1: Sample gamma and compute deep composition coefficients.
    // These are powers of gamma used to linearly combine the trace and composition
    // polynomial terms into the DEEP composition polynomial.
    let gamma = transcript.sample_field_element();

    let n_terms_composition_poly = round_2_result.lde_composition_poly_evaluations.len();
    let num_terms_trace =
        air.context().transition_offsets.len() * air.step_size() * air.context().trace_columns;

    let mut deep_composition_coefficients: Vec<_> =
        core::iter::successors(Some(FieldElement::one()), |x| Some(x * &gamma))
            .take(n_terms_composition_poly + num_terms_trace)
            .collect();

    // Split: first `num_terms_trace` coefficients are for trace terms,
    // remainder are for composition poly terms.
    let trace_term_coeffs: Vec<_> = deep_composition_coefficients
        .drain(..num_terms_trace)
        .collect::<Vec<_>>()
        .chunks(air.context().transition_offsets.len() * air.step_size())
        .map(|chunk| chunk.to_vec())
        .collect();

    let composition_gammas = deep_composition_coefficients;

    // Step 2: Compute DEEP composition polynomial (CPU).
    // Combine main + aux trace polys (since F == FieldExtension, no conversion needed).
    let mut all_trace_polys = round_1_result.main_trace_polys.clone();
    all_trace_polys.extend(round_1_result.aux_trace_polys.iter().cloned());

    let deep_composition_poly = compute_deep_composition_poly(
        &all_trace_polys,
        &round_2_result.composition_poly_parts,
        &round_3_result.trace_ood_evaluations,
        &round_3_result.composition_poly_parts_ood_evaluation,
        &round_3_result.z,
        &domain.trace_primitive_root,
        &composition_gammas,
        &trace_term_coeffs,
    );

    // Step 3: Run FRI commit phase.
    let domain_size = domain.lde_roots_of_unity_coset.len();
    let (fri_last_value, fri_layers) = fri::commit_phase::<F, F>(
        domain.root_order as usize,
        deep_composition_poly,
        transcript,
        &domain.coset_offset,
        domain_size,
    )?;

    // Step 4: Grinding.
    let security_bits = air.context().proof_options.grinding_factor;
    let mut nonce = None;
    if security_bits > 0 {
        let nonce_value = grinding::generate_nonce(&transcript.state(), security_bits)
            .ok_or(ProvingError::NonceNotFound(security_bits))?;
        transcript.append_bytes(&nonce_value.to_be_bytes());
        nonce = Some(nonce_value);
    }

    // Step 5: Sample query indexes.
    let number_of_queries = air.options().fri_number_of_queries;
    let domain_size_u64 = domain_size as u64;
    let iotas: Vec<usize> = (0..number_of_queries)
        .map(|_| transcript.sample_u64(domain_size_u64 >> 1) as usize)
        .collect();

    // Step 6: Run FRI query phase.
    let query_list = fri::query_phase(&fri_layers, &iotas)?;

    // Step 7: Extract FRI Merkle roots from layers.
    let fri_layers_merkle_roots: Vec<_> = fri_layers
        .iter()
        .map(|layer| layer.merkle_tree.root)
        .collect();

    // Step 8: Open deep composition poly (Merkle proofs for trace + composition poly).
    let deep_poly_openings =
        open_deep_composition_poly(domain, round_1_result, round_2_result, &iotas)?;

    Ok(GpuRound4Result {
        fri_last_value,
        fri_layers_merkle_roots,
        deep_poly_openings,
        query_list,
        nonce,
    })
}

/// Computes the DEEP composition polynomial.
///
/// The DEEP composition polynomial is a linear combination of:
/// - H terms: `sum_i gamma_i * (H_i(X) - H_i(z^N)) / (X - z^N)`
/// - Trace terms: `sum_jk gamma_jk * (t_j(X) - t_j(z*g^k)) / (X - z*g^k)`
///
/// where each division is performed via Ruffini division (synthetic division by a
/// linear factor).
#[allow(clippy::too_many_arguments)]
fn compute_deep_composition_poly<F>(
    trace_polys: &[Polynomial<FieldElement<F>>],
    composition_poly_parts: &[Polynomial<FieldElement<F>>],
    trace_ood_evaluations: &Table<F>,
    composition_poly_ood_evaluations: &[FieldElement<F>],
    z: &FieldElement<F>,
    primitive_root: &FieldElement<F>,
    composition_gammas: &[FieldElement<F>],
    trace_term_coeffs: &[Vec<FieldElement<F>>],
) -> Polynomial<FieldElement<F>>
where
    F: IsFFTField + IsSubFieldOf<F>,
{
    let z_power = z.pow(composition_poly_parts.len());

    // H terms: sum_i gamma_i * (H_i(X) - H_i(z^N)) / (X - z^N)
    let mut h_terms = Polynomial::zero();
    for (i, part) in composition_poly_parts.iter().enumerate() {
        let h_i_eval = &composition_poly_ood_evaluations[i];
        let h_i_term = &composition_gammas[i] * (part - h_i_eval);
        h_terms += h_i_term;
    }
    debug_assert_eq!(h_terms.evaluate(&z_power), FieldElement::zero());
    h_terms.ruffini_division_inplace(&z_power);

    // Trace terms: sum_jk gamma_jk * (t_j(X) - t_j(z*g^k)) / (X - z*g^k)
    let trace_evaluations_columns = trace_ood_evaluations.columns();

    // Pre-compute z_shifted values: z * g^k for each frame offset
    let num_offsets = trace_ood_evaluations.height;
    let z_shifted_values: Vec<FieldElement<F>> = (0..num_offsets)
        .map(|offset| primitive_root.pow(offset) * z)
        .collect();

    let trace_terms =
        trace_polys
            .iter()
            .enumerate()
            .fold(Polynomial::zero(), |accumulator, (i, t_j)| {
                let gammas_i = &trace_term_coeffs[i];
                let trace_evaluations_i = &trace_evaluations_columns[i];

                let trace_int = trace_evaluations_i
                    .iter()
                    .zip(&z_shifted_values)
                    .zip(gammas_i)
                    .fold(
                        Polynomial::zero(),
                        |trace_agg, ((trace_term_poly_evaluation, z_shifted), trace_gamma)| {
                            let mut poly = t_j - trace_term_poly_evaluation;
                            poly.ruffini_division_inplace(z_shifted);
                            trace_agg + poly * trace_gamma
                        },
                    );
                accumulator + trace_int
            });

    h_terms + trace_terms
}

/// Opens trace polynomials at a given query index.
///
/// For a given challenge index, computes `index = challenge * 2` and `index_sym = challenge * 2 + 1`
/// (the symmetric pair), retrieves Merkle proofs from the tree, and extracts the
/// evaluations from the column-major LDE data using bit-reversed indexing.
fn open_trace_polys<F>(
    domain_size: usize,
    tree: &BatchedMerkleTree<F>,
    lde_evaluations: &[Vec<FieldElement<F>>],
    challenge: usize,
) -> Result<PolynomialOpenings<F>, ProvingError>
where
    F: IsField,
    FieldElement<F>: AsBytes + Sync + Send,
{
    let index = challenge * 2;
    let index_sym = challenge * 2 + 1;

    let proof = tree.get_proof_by_pos(index).ok_or_else(|| {
        ProvingError::MerkleTreeError(format!("Failed to get proof at position {}", index))
    })?;
    let proof_sym = tree.get_proof_by_pos(index_sym).ok_or_else(|| {
        ProvingError::MerkleTreeError(format!("Failed to get proof at position {}", index_sym))
    })?;

    let actual_index = reverse_index(index, domain_size as u64);
    let actual_index_sym = reverse_index(index_sym, domain_size as u64);

    let evaluations: Vec<_> = lde_evaluations
        .iter()
        .map(|col| col[actual_index].clone())
        .collect();
    let evaluations_sym: Vec<_> = lde_evaluations
        .iter()
        .map(|col| col[actual_index_sym].clone())
        .collect();

    Ok(PolynomialOpenings {
        proof,
        proof_sym,
        evaluations,
        evaluations_sym,
    })
}

/// Opens composition polynomial at a given query index.
///
/// The composition polynomial Merkle tree uses a special paired-row layout
/// where consecutive bit-reversed evaluations are merged. This function retrieves
/// the Merkle proof and extracts evaluations accordingly.
fn open_composition_poly<F>(
    composition_poly_merkle_tree: &BatchedMerkleTree<F>,
    lde_composition_poly_evaluations: &[Vec<FieldElement<F>>],
    index: usize,
) -> Result<PolynomialOpenings<F>, ProvingError>
where
    F: IsField,
    FieldElement<F>: AsBytes + Sync + Send,
{
    let proof = composition_poly_merkle_tree
        .get_proof_by_pos(index)
        .ok_or_else(|| {
            ProvingError::MerkleTreeError(format!("Failed to get proof at position {}", index))
        })?;

    let lde_composition_poly_parts_evaluation: Vec<_> = lde_composition_poly_evaluations
        .iter()
        .flat_map(|part| {
            vec![
                part[reverse_index(index * 2, part.len() as u64)].clone(),
                part[reverse_index(index * 2 + 1, part.len() as u64)].clone(),
            ]
        })
        .collect();

    Ok(PolynomialOpenings {
        proof: proof.clone(),
        proof_sym: proof,
        evaluations: lde_composition_poly_parts_evaluation
            .clone()
            .into_iter()
            .step_by(2)
            .collect(),
        evaluations_sym: lde_composition_poly_parts_evaluation
            .into_iter()
            .skip(1)
            .step_by(2)
            .collect(),
    })
}

/// Opens the deep composition polynomial at a set of query indexes.
///
/// For each query index, this produces Merkle opening proofs for:
/// - Main trace polynomial evaluations
/// - Auxiliary trace polynomial evaluations (if present)
/// - Composition polynomial part evaluations
fn open_deep_composition_poly<F>(
    domain: &Domain<F>,
    round_1_result: &GpuRound1Result<F>,
    round_2_result: &GpuRound2Result<F>,
    indexes_to_open: &[usize],
) -> Result<DeepPolynomialOpenings<F, F>, ProvingError>
where
    F: IsFFTField,
    FieldElement<F>: AsBytes + Sync + Send,
{
    let domain_size = domain.lde_roots_of_unity_coset.len();
    let mut openings = Vec::new();

    for index in indexes_to_open.iter() {
        // Open main trace
        let main_trace_opening = open_trace_polys(
            domain_size,
            &round_1_result.main_merkle_tree,
            &round_1_result.main_lde_evaluations,
            *index,
        )?;

        // Open composition polynomial
        let composition_openings = open_composition_poly(
            &round_2_result.composition_poly_merkle_tree,
            &round_2_result.lde_composition_poly_evaluations,
            *index,
        )?;

        // Open auxiliary trace (if present)
        let aux_trace_polys = match (
            round_1_result.aux_merkle_tree.as_ref(),
            round_1_result.aux_lde_evaluations.is_empty(),
        ) {
            (Some(aux_tree), false) => Some(open_trace_polys(
                domain_size,
                aux_tree,
                &round_1_result.aux_lde_evaluations,
                *index,
            )?),
            _ => None,
        };

        openings.push(DeepPolynomialOpening {
            composition_poly: composition_openings,
            main_trace_polys: main_trace_opening,
            aux_trace_polys,
        });
    }

    Ok(openings)
}

/// GPU-optimized Phase 4 for Goldilocks field with GPU DEEP composition.
///
/// This is a concrete version of [`gpu_round_4`] that uses the Metal GPU shader
/// for DEEP composition polynomial computation instead of CPU Ruffini division.
///
/// If `precompiled_deep` is `Some`, uses the pre-compiled shader state.
#[cfg(all(target_os = "macos", feature = "metal"))]
#[allow(clippy::too_many_arguments)]
pub fn gpu_round_4_goldilocks<A>(
    air: &A,
    domain: &Domain<Goldilocks64Field>,
    round_1_result: &GpuRound1Result<Goldilocks64Field>,
    round_2_result: &GpuRound2Result<Goldilocks64Field>,
    round_3_result: &GpuRound3Result<Goldilocks64Field>,
    transcript: &mut impl IsStarkTranscript<Goldilocks64Field, Goldilocks64Field>,
    gpu_state: &StarkMetalState,
    precompiled_deep: Option<&DeepCompositionState>,
) -> Result<GpuRound4Result<Goldilocks64Field>, ProvingError>
where
    A: AIR<Field = Goldilocks64Field, FieldExtension = Goldilocks64Field>,
{
    type F = Goldilocks64Field;

    // Step 1: Sample gamma and compute deep composition coefficients.
    let gamma = transcript.sample_field_element();

    let n_terms_composition_poly = round_2_result.lde_composition_poly_evaluations.len();
    let num_terms_trace =
        air.context().transition_offsets.len() * air.step_size() * air.context().trace_columns;

    let mut deep_composition_coefficients: Vec<_> =
        core::iter::successors(Some(FieldElement::one()), |x| Some(x * gamma))
            .take(n_terms_composition_poly + num_terms_trace)
            .collect();

    let trace_term_coeffs: Vec<_> = deep_composition_coefficients
        .drain(..num_terms_trace)
        .collect::<Vec<_>>()
        .chunks(air.context().transition_offsets.len() * air.step_size())
        .map(|chunk| chunk.to_vec())
        .collect();

    let composition_gammas = deep_composition_coefficients;

    // Step 2: Compute DEEP composition polynomial on GPU.
    let deep_composition_poly = gpu_compute_deep_composition_poly(
        round_1_result,
        round_2_result,
        round_3_result,
        domain,
        &composition_gammas,
        &trace_term_coeffs,
        gpu_state,
        precompiled_deep,
    )?;

    // Step 3: Run FRI commit phase.
    let domain_size = domain.lde_roots_of_unity_coset.len();
    let (fri_last_value, fri_layers) = fri::commit_phase::<F, F>(
        domain.root_order as usize,
        deep_composition_poly,
        transcript,
        &domain.coset_offset,
        domain_size,
    )?;

    // Step 4: Grinding.
    let security_bits = air.context().proof_options.grinding_factor;
    let mut nonce = None;
    if security_bits > 0 {
        let nonce_value = grinding::generate_nonce(&transcript.state(), security_bits)
            .ok_or(ProvingError::NonceNotFound(security_bits))?;
        transcript.append_bytes(&nonce_value.to_be_bytes());
        nonce = Some(nonce_value);
    }

    // Step 5: Sample query indexes.
    let number_of_queries = air.options().fri_number_of_queries;
    let domain_size_u64 = domain_size as u64;
    let iotas: Vec<usize> = (0..number_of_queries)
        .map(|_| transcript.sample_u64(domain_size_u64 >> 1) as usize)
        .collect();

    // Step 6: Run FRI query phase.
    let query_list = fri::query_phase(&fri_layers, &iotas)?;

    // Step 7: Extract FRI Merkle roots from layers.
    let fri_layers_merkle_roots: Vec<_> = fri_layers
        .iter()
        .map(|layer| layer.merkle_tree.root)
        .collect();

    // Step 8: Open deep composition poly (Merkle proofs for trace + composition poly).
    let deep_poly_openings =
        open_deep_composition_poly(domain, round_1_result, round_2_result, &iotas)?;

    Ok(GpuRound4Result {
        fri_last_value,
        fri_layers_merkle_roots,
        deep_poly_openings,
        query_list,
        nonce,
    })
}

#[cfg(all(test, target_os = "macos", feature = "metal"))]
mod tests {
    use super::*;
    use crate::metal::phases::composition::gpu_round_2;
    use crate::metal::phases::ood::gpu_round_3;
    use crate::metal::phases::rap::gpu_round_1;
    use crate::metal::state::StarkMetalState;
    use lambdaworks_crypto::fiat_shamir::default_transcript::DefaultTranscript;
    use lambdaworks_crypto::fiat_shamir::is_transcript::IsTranscript;
    use lambdaworks_math::field::fields::u64_goldilocks_field::Goldilocks64Field;
    use stark_platinum_prover::proof::options::ProofOptions;

    use stark_platinum_prover::examples::fibonacci_rap::{
        fibonacci_rap_trace, FibonacciRAP, FibonacciRAPPublicInputs,
    };

    type F = Goldilocks64Field;
    type FpE = FieldElement<F>;

    #[test]
    fn gpu_round_4_fibonacci_rap_goldilocks() {
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
        let round_4 =
            gpu_round_4(&air, &domain, &round_1, &round_2, &round_3, &mut transcript).unwrap();

        // FRI last value should be non-zero (extremely unlikely to be zero)
        assert_ne!(round_4.fri_last_value, FpE::zero());

        // Should have FRI layer merkle roots (number depends on domain size)
        assert!(
            !round_4.fri_layers_merkle_roots.is_empty(),
            "FRI layers merkle roots should not be empty"
        );

        // Query list should have `fri_number_of_queries` entries
        assert_eq!(
            round_4.query_list.len(),
            proof_options.fri_number_of_queries
        );

        // Deep poly openings should have `fri_number_of_queries` entries
        assert_eq!(
            round_4.deep_poly_openings.len(),
            proof_options.fri_number_of_queries
        );

        // Each deep poly opening should have main trace, composition, and aux trace openings
        for opening in &round_4.deep_poly_openings {
            assert!(
                !opening.main_trace_polys.evaluations.is_empty(),
                "main trace evaluations should not be empty"
            );
            assert!(
                !opening.composition_poly.evaluations.is_empty(),
                "composition poly evaluations should not be empty"
            );
            // FibonacciRAP has auxiliary trace
            assert!(
                opening.aux_trace_polys.is_some(),
                "aux trace openings should be present for FibonacciRAP"
            );
        }

        // Nonce should be present since grinding_factor > 0 in test options
        assert!(
            round_4.nonce.is_some(),
            "nonce should be present when grinding_factor > 0"
        );
    }

    /// Verify that the deep composition polynomial computation produces correct results
    /// by checking that h_terms evaluates to zero at z^N (a necessary mathematical property).
    #[test]
    fn deep_composition_poly_h_terms_vanish_at_z_power() {
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

        let mut transcript = DefaultTranscript::<F>::new(&[]);
        let round_1 = gpu_round_1(&air, &mut trace, &domain, &mut transcript, &state).unwrap();
        let round_2 = gpu_round_2(&air, &domain, &round_1, &mut transcript, &state).unwrap();
        let round_3 = gpu_round_3(&air, &domain, &round_1, &round_2, &mut transcript).unwrap();

        // Sample gamma the same way as gpu_round_4
        let gamma: FieldElement<F> = transcript.sample_field_element();
        let n_terms_composition_poly = round_2.lde_composition_poly_evaluations.len();

        let composition_gammas: Vec<_> =
            core::iter::successors(Some(FieldElement::one()), |x| Some(x * gamma))
                .take(n_terms_composition_poly)
                .collect();

        // Verify that sum_i gamma_i * (H_i(X) - H_i(z^N)) vanishes at z^N
        let z_power = round_3.z.pow(round_2.composition_poly_parts.len());
        let mut h_terms = Polynomial::zero();
        for (i, part) in round_2.composition_poly_parts.iter().enumerate() {
            let h_i_eval = &round_3.composition_poly_parts_ood_evaluation[i];
            let h_i_term = &composition_gammas[i] * (part - h_i_eval);
            h_terms += h_i_term;
        }
        assert_eq!(
            h_terms.evaluate(&z_power),
            FieldElement::zero(),
            "H terms must vanish at z^N before Ruffini division"
        );
    }
}
