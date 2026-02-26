//! FRI-based PCS adapter: wraps FRI with DEEP technique for opening at arbitrary points.
//!
//! DEEP opening at z:
//! 1. Given committed f(X), claimed f(z) = v
//! 2. Quotient: q(X) = (f(X) - v) / (X - z) via ruffini_division
//! 3. Run FRI on q(X) — if f(z) = v, q has degree < deg(f), FRI accepts
//!
//! Batch opening (k polynomials at same z):
//! 1. Sample alpha from transcript
//! 2. Combined quotient: q(X) = sum_j alpha^j * (f_j(X) - v_j) / (X - z)
//! 3. FRI commit + sample queries on q
//! 4. For each query, decommit original polynomial evaluations + Merkle proofs
//! 5. Verifier reconstructs expected quotient from original poly evals and checks
//!    against FRI layer-0 decommitments

use lambdaworks_crypto::fiat_shamir::is_transcript::IsTranscript;
use lambdaworks_crypto::merkle_tree::backends::types::Keccak256Backend;
use lambdaworks_crypto::merkle_tree::merkle::MerkleTree;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::IsFFTField;
use lambdaworks_math::polynomial::Polynomial;
use lambdaworks_math::traits::{AsBytes, ByteConversion};

use super::query::fri_query_all;
use super::types::{FriConfig, FriProof, OriginalPolyDecommitment};
use super::{fri_commit_and_sample, fri_verify};
use crate::univariate::pcs::{CommitmentSchemeError, IsUnivariateCommitmentScheme};

/// FRI-based polynomial commitment scheme.
///
/// Commits by evaluating on an extended domain and Merkle-hashing the LDE.
/// Opens at arbitrary points via the DEEP quotient technique.
pub struct FriCommitmentScheme {
    pub config: FriConfig,
}

/// FRI commitment: Merkle root of the LDE evaluations.
#[derive(Clone, Debug)]
pub struct FriCommitment {
    pub merkle_root: [u8; 32],
}

/// Prover state: polynomial coefficients + LDE evaluations + Merkle tree.
pub struct FriProverState<F: IsFFTField>
where
    FieldElement<F>: AsBytes,
    F::BaseType: Send + Sync,
{
    /// Polynomial in coefficient form (interpolated from evals on H).
    pub coefficients: Polynomial<FieldElement<F>>,
    /// LDE evaluations on the extended domain.
    pub lde_evals: Vec<FieldElement<F>>,
    /// Merkle tree of LDE evaluations.
    pub merkle_tree: MerkleTree<Keccak256Backend<F>>,
    /// Degree bound (= domain size N).
    pub degree_bound: usize,
}

/// Batch opening proof: a FRI proof of the combined DEEP quotient, plus
/// decommitments from the original polynomial LDEs that bind the quotient
/// to the committed polynomials.
#[derive(Clone, Debug)]
pub struct FriBatchOpeningProof<F: IsFFTField> {
    /// The FRI proof for the combined quotient polynomial.
    pub fri_proof: FriProof<F>,
    /// Degree bound of the original polynomials (= N, the domain size).
    pub degree_bound: usize,
    /// Per-query, per-polynomial decommitments from the original LDEs.
    /// `original_decommitments[q][j]` = decommitment for query `q`, polynomial `j`.
    pub original_decommitments: Vec<Vec<OriginalPolyDecommitment<F>>>,
}

impl FriCommitmentScheme {
    pub fn new(config: FriConfig) -> Self {
        Self { config }
    }
}

impl<F> IsUnivariateCommitmentScheme<F> for FriCommitmentScheme
where
    F: IsFFTField,
    F::BaseType: Send + Sync,
    FieldElement<F>: AsBytes + ByteConversion + Clone + Send + Sync,
{
    type Commitment = FriCommitment;
    type ProverState = FriProverState<F>;
    type BatchOpeningProof = FriBatchOpeningProof<F>;

    fn commit<T: IsTranscript<F>>(
        &self,
        evals_on_h: &[FieldElement<F>],
        transcript: &mut T,
    ) -> Result<(Self::Commitment, Self::ProverState), CommitmentSchemeError>
    where
        FieldElement<F>: ByteConversion,
    {
        let n = evals_on_h.len();
        if !n.is_power_of_two() || n == 0 {
            return Err(CommitmentSchemeError::InvalidInput(
                "evaluations length must be a nonzero power of 2".into(),
            ));
        }

        let log_n = n.trailing_zeros() as usize;
        if log_n > FriConfig::MAX_LOG_DEGREE {
            return Err(CommitmentSchemeError::InvalidInput(format!(
                "degree bound 2^{log_n} exceeds maximum 2^{}",
                FriConfig::MAX_LOG_DEGREE
            )));
        }

        // Interpolate to get coefficient form
        let poly = Polynomial::interpolate_fft::<F>(evals_on_h).map_err(|e| {
            CommitmentSchemeError::InternalError(format!("FFT interpolation failed: {e}"))
        })?;

        // Compute LDE (with blowup)
        let blowup = self.config.blowup_factor();
        let lde_evals = Polynomial::evaluate_fft::<F>(&poly, blowup, Some(n)).map_err(|e| {
            CommitmentSchemeError::InternalError(format!("FFT evaluation failed: {e}"))
        })?;

        let lde_size = n * blowup;
        let lde_evals: Vec<FieldElement<F>> = lde_evals.into_iter().take(lde_size).collect();

        // Build Merkle tree
        let merkle_tree =
            MerkleTree::<Keccak256Backend<F>>::build(&lde_evals).ok_or_else(|| {
                CommitmentSchemeError::InternalError("failed to build Merkle tree".into())
            })?;

        // Append Merkle root to transcript
        transcript.append_bytes(&merkle_tree.root);

        let commitment = FriCommitment {
            merkle_root: merkle_tree.root,
        };

        let prover_state = FriProverState {
            coefficients: poly,
            lde_evals,
            merkle_tree,
            degree_bound: n,
        };

        Ok((commitment, prover_state))
    }

    fn open(
        &self,
        state: &Self::ProverState,
        z: &FieldElement<F>,
    ) -> Result<FieldElement<F>, CommitmentSchemeError> {
        Ok(state.coefficients.evaluate(z))
    }

    fn batch_open<T: IsTranscript<F>>(
        &self,
        states: &[&Self::ProverState],
        z: &FieldElement<F>,
        transcript: &mut T,
    ) -> Result<(Vec<FieldElement<F>>, Self::BatchOpeningProof), CommitmentSchemeError>
    where
        FieldElement<F>: ByteConversion,
    {
        if states.is_empty() {
            return Err(CommitmentSchemeError::InvalidInput(
                "no polynomials to open".into(),
            ));
        }

        let degree_bound = states[0].degree_bound;

        // All polynomials must share the same degree bound (same domain size)
        if states.iter().any(|s| s.degree_bound != degree_bound) {
            return Err(CommitmentSchemeError::InvalidInput(
                "all polynomials must have the same degree bound".into(),
            ));
        }

        // degree_bound must be a power of two (comes from cyclic domain H of size 2^n)
        // so that the FRI LDE domain size = degree_bound * blowup is also a power of two
        if !degree_bound.is_power_of_two() || degree_bound == 0 {
            return Err(CommitmentSchemeError::InvalidInput(
                "degree_bound must be a nonzero power of 2".into(),
            ));
        }

        let config = &self.config;

        // Bind FRI config and degree_bound to transcript (prevents parameter confusion)
        transcript.append_bytes(b"fri-config");
        transcript.append_bytes(&(config.log_blowup as u64).to_le_bytes());
        transcript.append_bytes(&(config.num_queries as u64).to_le_bytes());
        transcript.append_bytes(&(degree_bound as u64).to_le_bytes());

        // Evaluate each polynomial at z
        let values: Vec<FieldElement<F>> =
            states.iter().map(|s| s.coefficients.evaluate(z)).collect();

        // Append values to transcript
        for v in &values {
            transcript.append_field_element(v);
        }

        // Sample alpha for random linear combination
        let alpha: FieldElement<F> = transcript.sample_field_element();

        // Build combined DEEP quotient: q(X) = sum_j alpha^j * (f_j(X) - v_j) / (X - z)
        let mut combined_quotient = Polynomial::zero();
        let mut alpha_power = FieldElement::<F>::one();

        for (state, value) in states.iter().zip(values.iter()) {
            let shifted = &state.coefficients - &Polynomial::new(core::slice::from_ref(value));
            let quotient = shifted.ruffini_division(z);
            let scaled = quotient.scale_coeffs(&alpha_power);
            combined_quotient += scaled;
            alpha_power = &alpha_power * &alpha;
        }

        // FRI commit + sample query indices (instead of full fri_prove)
        let (commit_result, query_indices) =
            fri_commit_and_sample(&combined_quotient, config, transcript).map_err(|e| {
                CommitmentSchemeError::InternalError(format!("FRI commit failed: {e}"))
            })?;

        // FRI query phase for the quotient
        let query_rounds = if commit_result.layers.is_empty() {
            vec![vec![]; config.num_queries]
        } else {
            fri_query_all(
                &query_indices,
                &commit_result.layers,
                &commit_result.merkle_trees,
            )
            .map_err(|e| CommitmentSchemeError::InternalError(format!("FRI query failed: {e}")))?
        };

        let fri_proof = FriProof {
            layer_merkle_roots: commit_result.layers.iter().map(|l| l.merkle_root).collect(),
            query_rounds,
            final_value: commit_result.final_value,
        };

        // Extract decommitments from original polynomial LDEs at each query index.
        // The FRI layer-0 domain has size degree_bound * blowup, which is the same
        // as the original LDE domain size. So query indices map directly.
        let blowup = config.blowup_factor();
        let lde_domain_size = degree_bound * blowup;
        let half = lde_domain_size / 2;

        let mut original_decommitments = Vec::with_capacity(query_indices.len());

        // Verify all LDE eval vectors have the expected length before indexing
        for (j, state) in states.iter().enumerate() {
            if state.lde_evals.len() != lde_domain_size {
                return Err(CommitmentSchemeError::InternalError(format!(
                    "polynomial {j}: LDE length {} != expected {lde_domain_size}",
                    state.lde_evals.len()
                )));
            }
        }

        for &qi in &query_indices {
            let idx = qi % half;
            let idx_sym = idx + half;

            let mut per_poly = Vec::with_capacity(states.len());
            for state in states.iter() {
                let eval = state.lde_evals[idx].clone();
                let eval_sym = state.lde_evals[idx_sym].clone();

                let auth_path = state.merkle_tree.get_proof_by_pos(idx).ok_or_else(|| {
                    CommitmentSchemeError::InternalError(format!("no Merkle proof at index {idx}"))
                })?;
                let auth_path_sym =
                    state.merkle_tree.get_proof_by_pos(idx_sym).ok_or_else(|| {
                        CommitmentSchemeError::InternalError(format!(
                            "no Merkle proof at symmetric index {idx_sym}"
                        ))
                    })?;

                per_poly.push(OriginalPolyDecommitment {
                    eval,
                    eval_sym,
                    auth_path,
                    auth_path_sym,
                });
            }
            original_decommitments.push(per_poly);
        }

        let proof = FriBatchOpeningProof {
            fri_proof,
            degree_bound,
            original_decommitments,
        };

        Ok((values, proof))
    }

    fn absorb_commitment<T: IsTranscript<F>>(commitment: &Self::Commitment, transcript: &mut T) {
        transcript.append_bytes(&commitment.merkle_root);
    }

    fn degree_bound_from_proof(proof: &Self::BatchOpeningProof) -> Option<usize> {
        Some(proof.degree_bound)
    }

    fn verify_batch_opening<T: IsTranscript<F>>(
        &self,
        commitments: &[&Self::Commitment],
        z: &FieldElement<F>,
        values: &[FieldElement<F>],
        proof: &Self::BatchOpeningProof,
        transcript: &mut T,
    ) -> Result<(), CommitmentSchemeError>
    where
        FieldElement<F>: ByteConversion,
    {
        if commitments.len() != values.len() {
            return Err(CommitmentSchemeError::InvalidInput(
                "commitments and values length mismatch".into(),
            ));
        }

        let num_polys = commitments.len();
        let config = &self.config;

        // Validate degree_bound from proof
        if !proof.degree_bound.is_power_of_two() || proof.degree_bound == 0 {
            return Err(CommitmentSchemeError::InvalidInput(
                "proof degree_bound must be a nonzero power of 2".into(),
            ));
        }
        let log_deg = proof.degree_bound.trailing_zeros() as usize;
        if log_deg > FriConfig::MAX_LOG_DEGREE {
            return Err(CommitmentSchemeError::InvalidInput(format!(
                "proof degree_bound 2^{log_deg} exceeds maximum 2^{}",
                FriConfig::MAX_LOG_DEGREE
            )));
        }

        // Bind FRI config and degree_bound to transcript (must match prover)
        transcript.append_bytes(b"fri-config");
        transcript.append_bytes(&(config.log_blowup as u64).to_le_bytes());
        transcript.append_bytes(&(config.num_queries as u64).to_le_bytes());
        transcript.append_bytes(&(proof.degree_bound as u64).to_le_bytes());

        // Append values to transcript (must match prover)
        for v in values {
            transcript.append_field_element(v);
        }

        // Sample alpha (must match prover)
        let alpha: FieldElement<F> = transcript.sample_field_element();

        // Verify FRI proof for the combined quotient — returns query indices
        let query_indices = fri_verify(&proof.fri_proof, proof.degree_bound, config, transcript)
            .map_err(|e| CommitmentSchemeError::InternalError(format!("FRI verify failed: {e}")))?;

        // Verify original polynomial decommitments and quotient consistency
        if proof.original_decommitments.len() != query_indices.len() {
            return Err(CommitmentSchemeError::InvalidInput(
                "original decommitments count doesn't match query count".into(),
            ));
        }

        let blowup = config.blowup_factor();
        let lde_domain_size = proof.degree_bound * blowup;
        let half = lde_domain_size / 2;

        // Compute the LDE domain generator
        let log_lde = lde_domain_size.trailing_zeros() as u64;
        let omega_lde = F::get_primitive_root_of_unity(log_lde).map_err(|_| {
            CommitmentSchemeError::InternalError("no root of unity for LDE domain".into())
        })?;

        // Precompute alpha powers
        let mut alpha_powers = Vec::with_capacity(num_polys);
        let mut ap = FieldElement::<F>::one();
        for _ in 0..num_polys {
            alpha_powers.push(ap.clone());
            ap = &ap * &alpha;
        }

        for (q, (qi, decommitments)) in query_indices
            .iter()
            .zip(proof.original_decommitments.iter())
            .enumerate()
        {
            if decommitments.len() != num_polys {
                return Err(CommitmentSchemeError::InvalidInput(format!(
                    "query {q}: expected {num_polys} decommitments, got {}",
                    decommitments.len()
                )));
            }

            let idx = qi % half;
            let idx_sym = idx + half;

            // Domain point x = omega_lde^idx
            let x = omega_lde.pow(idx);

            // For each original polynomial, verify Merkle proofs against committed roots
            for (j, decomm) in decommitments.iter().enumerate() {
                let root = &commitments[j].merkle_root;

                if !decomm
                    .auth_path
                    .verify::<Keccak256Backend<F>>(root, idx, &decomm.eval)
                {
                    return Err(CommitmentSchemeError::VerificationFailed);
                }
                if !decomm.auth_path_sym.verify::<Keccak256Backend<F>>(
                    root,
                    idx_sym,
                    &decomm.eval_sym,
                ) {
                    return Err(CommitmentSchemeError::VerificationFailed);
                }
            }

            // Compute expected quotient at x:
            // q(x) = sum_j alpha^j * (f_j(x) - v_j) / (x - z)
            let x_minus_z_inv = (&x - z)
                .inv()
                .map_err(|_| CommitmentSchemeError::InternalError("x - z is zero".into()))?;

            let mut expected_q_x = FieldElement::<F>::zero();
            for (j, decomm) in decommitments.iter().enumerate() {
                let diff = &decomm.eval - &values[j];
                expected_q_x += &alpha_powers[j] * &diff * &x_minus_z_inv;
            }

            // Check against FRI layer-0 eval at this query
            let fri_layer0 = &proof.fri_proof.query_rounds[q];
            if fri_layer0.is_empty() {
                // Constant quotient — check against final value
                if expected_q_x != proof.fri_proof.final_value {
                    return Err(CommitmentSchemeError::VerificationFailed);
                }
            } else if expected_q_x != fri_layer0[0].eval {
                return Err(CommitmentSchemeError::VerificationFailed);
            }

            // Same check for symmetric point -x
            let neg_x = -&x;
            let neg_x_minus_z_inv = (&neg_x - z)
                .inv()
                .map_err(|_| CommitmentSchemeError::InternalError("-x - z is zero".into()))?;

            let mut expected_q_neg_x = FieldElement::<F>::zero();
            for (j, decomm) in decommitments.iter().enumerate() {
                let diff = &decomm.eval_sym - &values[j];
                expected_q_neg_x += &alpha_powers[j] * &diff * &neg_x_minus_z_inv;
            }

            if fri_layer0.is_empty() {
                if expected_q_neg_x != proof.fri_proof.final_value {
                    return Err(CommitmentSchemeError::VerificationFailed);
                }
            } else if expected_q_neg_x != fri_layer0[0].eval_sym {
                return Err(CommitmentSchemeError::VerificationFailed);
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::univariate::pcs::IsUnivariateCommitmentScheme;
    use lambdaworks_crypto::fiat_shamir::default_transcript::DefaultTranscript;
    use lambdaworks_math::field::fields::fft_friendly::quartic_babybear::Degree4BabyBearExtensionField;

    type F = Degree4BabyBearExtensionField;
    type FE = FieldElement<F>;

    fn default_fri_pcs() -> FriCommitmentScheme {
        FriCommitmentScheme::new(FriConfig::default())
    }

    #[test]
    fn fri_pcs_single_open_verify() {
        let pcs = default_fri_pcs();
        let evals: Vec<FE> = (1..=8).map(|i| FE::from(i as u64)).collect();

        let mut prover_transcript = DefaultTranscript::<F>::new(b"fri_pcs_single");
        let (commitment, state) = pcs
            .commit::<DefaultTranscript<F>>(&evals, &mut prover_transcript)
            .unwrap();

        let z = FE::from(42u64);
        let value = pcs.open(&state, &z).unwrap();

        // Batch open with a single polynomial
        let (values, proof) = pcs
            .batch_open::<DefaultTranscript<F>>(&[&state], &z, &mut prover_transcript)
            .unwrap();
        assert_eq!(values[0], value);

        // Verify
        let mut verifier_transcript = DefaultTranscript::<F>::new(b"fri_pcs_single");
        // Replay commitment
        verifier_transcript.append_bytes(&commitment.merkle_root);

        pcs.verify_batch_opening::<DefaultTranscript<F>>(
            &[&commitment],
            &z,
            &values,
            &proof,
            &mut verifier_transcript,
        )
        .unwrap();
    }

    #[test]
    fn fri_pcs_batch_open_verify() {
        let pcs = default_fri_pcs();
        let evals1: Vec<FE> = (1..=8).map(|i| FE::from(i as u64)).collect();
        let evals2: Vec<FE> = (10..=17).map(|i| FE::from(i as u64)).collect();

        let mut prover_transcript = DefaultTranscript::<F>::new(b"fri_pcs_batch");
        let (c1, s1) = pcs
            .commit::<DefaultTranscript<F>>(&evals1, &mut prover_transcript)
            .unwrap();
        let (c2, s2) = pcs
            .commit::<DefaultTranscript<F>>(&evals2, &mut prover_transcript)
            .unwrap();

        let z = FE::from(99u64);
        let (values, proof) = pcs
            .batch_open::<DefaultTranscript<F>>(&[&s1, &s2], &z, &mut prover_transcript)
            .unwrap();

        assert_eq!(values.len(), 2);
        assert_eq!(values[0], s1.coefficients.evaluate(&z));
        assert_eq!(values[1], s2.coefficients.evaluate(&z));

        // Verify
        let mut verifier_transcript = DefaultTranscript::<F>::new(b"fri_pcs_batch");
        // Replay commitments
        verifier_transcript.append_bytes(&c1.merkle_root);
        verifier_transcript.append_bytes(&c2.merkle_root);

        pcs.verify_batch_opening::<DefaultTranscript<F>>(
            &[&c1, &c2],
            &z,
            &values,
            &proof,
            &mut verifier_transcript,
        )
        .unwrap();
    }

    #[test]
    fn fri_pcs_tampered_value_rejected() {
        let pcs = default_fri_pcs();
        let evals: Vec<FE> = (1..=8).map(|i| FE::from(i as u64)).collect();

        let mut prover_transcript = DefaultTranscript::<F>::new(b"fri_pcs_tamper");
        let (commitment, state) = pcs
            .commit::<DefaultTranscript<F>>(&evals, &mut prover_transcript)
            .unwrap();

        let z = FE::from(42u64);
        let (mut values, proof) = pcs
            .batch_open::<DefaultTranscript<F>>(&[&state], &z, &mut prover_transcript)
            .unwrap();

        // Tamper with the value
        values[0] += FE::one();

        let mut verifier_transcript = DefaultTranscript::<F>::new(b"fri_pcs_tamper");
        verifier_transcript.append_bytes(&commitment.merkle_root);

        let result = pcs.verify_batch_opening::<DefaultTranscript<F>>(
            &[&commitment],
            &z,
            &values,
            &proof,
            &mut verifier_transcript,
        );

        // Tampering should cause quotient consistency check or FRI failure
        assert!(result.is_err());
    }

    #[test]
    fn fri_pcs_commitment_binding() {
        let pcs = default_fri_pcs();
        // Verify that a forged quotient (unrelated to committed polys) is rejected.
        // Honest commit + open, then swap in a different polynomial's proof.
        let evals1: Vec<FE> = (1..=8).map(|i| FE::from(i as u64)).collect();
        let evals2: Vec<FE> = (100..=107).map(|i| FE::from(i as u64)).collect();

        // Commit poly1
        let mut t1 = DefaultTranscript::<F>::new(b"binding");
        let (c1, _s1) = pcs
            .commit::<DefaultTranscript<F>>(&evals1, &mut t1)
            .unwrap();

        // Commit poly2 (different transcript to isolate)
        let mut t2 = DefaultTranscript::<F>::new(b"binding2");
        let (_c2, s2) = pcs
            .commit::<DefaultTranscript<F>>(&evals2, &mut t2)
            .unwrap();

        let z = FE::from(42u64);

        // Open poly2 (honest proof for poly2)
        let mut t_open = DefaultTranscript::<F>::new(b"binding");
        t_open.append_bytes(&c1.merkle_root);
        let (values2, proof2) = pcs
            .batch_open::<DefaultTranscript<F>>(&[&s2], &z, &mut t_open)
            .unwrap();

        // Now try to verify: commitment to poly1, but values and proof from poly2.
        // The verifier should reject because the original poly decommitments in proof2
        // won't match commitment c1's Merkle root.
        let mut t_verify = DefaultTranscript::<F>::new(b"binding");
        t_verify.append_bytes(&c1.merkle_root);

        let result = pcs.verify_batch_opening::<DefaultTranscript<F>>(
            &[&c1],
            &z,
            &values2,
            &proof2,
            &mut t_verify,
        );

        assert!(result.is_err(), "forged quotient should be rejected");
    }

    #[test]
    fn fri_pcs_wrong_degree_bound_rejected() {
        let pcs = default_fri_pcs();
        let evals: Vec<FE> = (1..=8).map(|i| FE::from(i as u64)).collect();

        let mut prover_transcript = DefaultTranscript::<F>::new(b"deg_bound");
        let (commitment, state) = pcs
            .commit::<DefaultTranscript<F>>(&evals, &mut prover_transcript)
            .unwrap();

        let z = FE::from(42u64);
        let (values, mut proof) = pcs
            .batch_open::<DefaultTranscript<F>>(&[&state], &z, &mut prover_transcript)
            .unwrap();

        // Tamper with degree_bound
        proof.degree_bound = 4; // was 8

        let mut verifier_transcript = DefaultTranscript::<F>::new(b"deg_bound");
        verifier_transcript.append_bytes(&commitment.merkle_root);

        let result = pcs.verify_batch_opening::<DefaultTranscript<F>>(
            &[&commitment],
            &z,
            &values,
            &proof,
            &mut verifier_transcript,
        );

        // Changing degree_bound changes the FRI domain size, causing query index mismatch
        assert!(result.is_err(), "wrong degree bound should be rejected");
    }
}
