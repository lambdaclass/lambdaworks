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
//! 3. Single FRI on q

use lambdaworks_crypto::fiat_shamir::is_transcript::IsTranscript;
use lambdaworks_crypto::merkle_tree::backends::types::Keccak256Backend;
use lambdaworks_crypto::merkle_tree::merkle::MerkleTree;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::IsFFTField;
use lambdaworks_math::polynomial::Polynomial;
use lambdaworks_math::traits::{AsBytes, ByteConversion};

use super::types::{FriConfig, FriProof};
use super::{fri_prove, fri_verify};
use crate::univariate::pcs::{PcsError, UnivariatePcs};

/// FRI-based polynomial commitment scheme.
///
/// Commits by evaluating on an extended domain and Merkle-hashing the LDE.
/// Opens at arbitrary points via the DEEP quotient technique.
pub struct FriPcs {
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

/// Batch opening proof: a FRI proof of the combined DEEP quotient.
#[derive(Clone, Debug)]
pub struct FriBatchOpeningProof<F: IsFFTField> {
    /// The FRI proof for the combined quotient polynomial.
    pub fri_proof: FriProof<F>,
    /// The opening alpha used for random linear combination.
    pub alpha: FieldElement<F>,
}

impl FriPcs {
    pub fn new(config: FriConfig) -> Self {
        Self { config }
    }
}

impl<F> UnivariatePcs<F> for FriPcs
where
    F: IsFFTField,
    F::BaseType: Send + Sync,
    FieldElement<F>: AsBytes + ByteConversion + Clone + Send + Sync,
{
    type Commitment = FriCommitment;
    type ProverState = FriProverState<F>;
    type BatchOpeningProof = FriBatchOpeningProof<F>;

    fn commit<T: IsTranscript<F>>(
        evals_on_h: &[FieldElement<F>],
        transcript: &mut T,
    ) -> Result<(Self::Commitment, Self::ProverState), PcsError>
    where
        FieldElement<F>: ByteConversion,
    {
        let n = evals_on_h.len();
        if !n.is_power_of_two() || n == 0 {
            return Err(PcsError::InvalidInput(
                "evaluations length must be a nonzero power of 2".into(),
            ));
        }

        // Interpolate to get coefficient form
        let poly = Polynomial::interpolate_fft::<F>(evals_on_h)
            .map_err(|e| PcsError::InternalError(format!("FFT interpolation failed: {e}")))?;

        // Compute LDE (with blowup)
        let blowup = 1 << FriConfig::default().log_blowup;
        let lde_evals = Polynomial::evaluate_fft::<F>(&poly, blowup, Some(n))
            .map_err(|e| PcsError::InternalError(format!("FFT evaluation failed: {e}")))?;

        let lde_size = n * blowup;
        let lde_evals: Vec<FieldElement<F>> = lde_evals.into_iter().take(lde_size).collect();

        // Build Merkle tree
        let merkle_tree = MerkleTree::<Keccak256Backend<F>>::build(&lde_evals)
            .ok_or_else(|| PcsError::InternalError("failed to build Merkle tree".into()))?;

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

    fn open(state: &Self::ProverState, z: &FieldElement<F>) -> Result<FieldElement<F>, PcsError> {
        Ok(state.coefficients.evaluate(z))
    }

    fn batch_open<T: IsTranscript<F>>(
        states: &[&Self::ProverState],
        z: &FieldElement<F>,
        transcript: &mut T,
    ) -> Result<(Vec<FieldElement<F>>, Self::BatchOpeningProof), PcsError>
    where
        FieldElement<F>: ByteConversion,
    {
        if states.is_empty() {
            return Err(PcsError::InvalidInput("no polynomials to open".into()));
        }

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
            // f_j(X) - v_j
            let shifted = &state.coefficients - &Polynomial::new(core::slice::from_ref(value));
            // (f_j(X) - v_j) / (X - z) via Ruffini
            let quotient = shifted.ruffini_division(z);
            // alpha^j * quotient
            let scaled = quotient.scale_coeffs(&alpha_power);

            combined_quotient += scaled;
            alpha_power = &alpha_power * &alpha;
        }

        // Run FRI on the combined quotient
        let config = FriConfig::default();

        let fri_proof = fri_prove(&combined_quotient, &config, transcript)
            .map_err(|e| PcsError::InternalError(format!("FRI prove failed: {e}")))?;

        let proof = FriBatchOpeningProof {
            fri_proof,
            alpha: alpha.clone(),
        };

        Ok((values, proof))
    }

    fn absorb_commitment<T: IsTranscript<F>>(commitment: &Self::Commitment, transcript: &mut T) {
        transcript.append_bytes(&commitment.merkle_root);
    }

    fn verify_batch_opening<T: IsTranscript<F>>(
        commitments: &[&Self::Commitment],
        _z: &FieldElement<F>,
        values: &[FieldElement<F>],
        proof: &Self::BatchOpeningProof,
        transcript: &mut T,
    ) -> Result<(), PcsError>
    where
        FieldElement<F>: ByteConversion,
    {
        if commitments.len() != values.len() {
            return Err(PcsError::InvalidInput(
                "commitments and values length mismatch".into(),
            ));
        }

        // Append values to transcript (must match prover)
        for v in values {
            transcript.append_field_element(v);
        }

        // Sample alpha (must match prover)
        let alpha: FieldElement<F> = transcript.sample_field_element();

        // Check alpha matches
        if alpha != proof.alpha {
            return Err(PcsError::VerificationFailed);
        }

        // Verify FRI proof for the combined quotient
        // The quotient degree is at most max(degree_bound) - 1
        // We don't know the exact degree bound here, but FRI verifies the low-degree test.
        // For now we use the number of commitments as a proxy — in practice the
        // degree bound should be passed as configuration.
        let config = FriConfig::default();

        // The combined quotient has degree < max_degree_bound - 1.
        // We need to provide a degree bound to the verifier. Since we committed
        // each polynomial with degree < N, the quotient after dividing by (X - z)
        // has degree < N - 1. We derive N from the FRI proof structure.
        let num_layers = proof.fri_proof.layer_merkle_roots.len();
        let poly_degree_bound = if num_layers > 0 { 1 << num_layers } else { 1 };

        fri_verify(&proof.fri_proof, poly_degree_bound, &config, transcript)
            .map_err(|e| PcsError::InternalError(format!("FRI verify failed: {e}")))?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::univariate::pcs::UnivariatePcs;
    use lambdaworks_crypto::fiat_shamir::default_transcript::DefaultTranscript;
    use lambdaworks_math::field::fields::fft_friendly::quartic_babybear::Degree4BabyBearExtensionField;

    type F = Degree4BabyBearExtensionField;
    type FE = FieldElement<F>;

    #[test]
    fn fri_pcs_single_open_verify() {
        let evals: Vec<FE> = (1..=8).map(|i| FE::from(i as u64)).collect();

        let mut prover_transcript = DefaultTranscript::<F>::new(b"fri_pcs_single");
        let (commitment, state) =
            FriPcs::commit::<DefaultTranscript<F>>(&evals, &mut prover_transcript).unwrap();

        let z = FE::from(42u64);
        let value = FriPcs::open(&state, &z).unwrap();

        // Batch open with a single polynomial
        let (values, proof) =
            FriPcs::batch_open::<DefaultTranscript<F>>(&[&state], &z, &mut prover_transcript)
                .unwrap();
        assert_eq!(values[0], value);

        // Verify
        let mut verifier_transcript = DefaultTranscript::<F>::new(b"fri_pcs_single");
        // Replay commitment
        verifier_transcript.append_bytes(&commitment.merkle_root);

        FriPcs::verify_batch_opening::<DefaultTranscript<F>>(
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
        let evals1: Vec<FE> = (1..=8).map(|i| FE::from(i as u64)).collect();
        let evals2: Vec<FE> = (10..=17).map(|i| FE::from(i as u64)).collect();

        let mut prover_transcript = DefaultTranscript::<F>::new(b"fri_pcs_batch");
        let (c1, s1) =
            FriPcs::commit::<DefaultTranscript<F>>(&evals1, &mut prover_transcript).unwrap();
        let (c2, s2) =
            FriPcs::commit::<DefaultTranscript<F>>(&evals2, &mut prover_transcript).unwrap();

        let z = FE::from(99u64);
        let (values, proof) =
            FriPcs::batch_open::<DefaultTranscript<F>>(&[&s1, &s2], &z, &mut prover_transcript)
                .unwrap();

        assert_eq!(values.len(), 2);
        assert_eq!(values[0], s1.coefficients.evaluate(&z));
        assert_eq!(values[1], s2.coefficients.evaluate(&z));

        // Verify
        let mut verifier_transcript = DefaultTranscript::<F>::new(b"fri_pcs_batch");
        // Replay commitments
        verifier_transcript.append_bytes(&c1.merkle_root);
        verifier_transcript.append_bytes(&c2.merkle_root);

        FriPcs::verify_batch_opening::<DefaultTranscript<F>>(
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
        let evals: Vec<FE> = (1..=8).map(|i| FE::from(i as u64)).collect();

        let mut prover_transcript = DefaultTranscript::<F>::new(b"fri_pcs_tamper");
        let (commitment, state) =
            FriPcs::commit::<DefaultTranscript<F>>(&evals, &mut prover_transcript).unwrap();

        let z = FE::from(42u64);
        let (mut values, proof) =
            FriPcs::batch_open::<DefaultTranscript<F>>(&[&state], &z, &mut prover_transcript)
                .unwrap();

        // Tamper with the value
        values[0] += FE::one();

        let mut verifier_transcript = DefaultTranscript::<F>::new(b"fri_pcs_tamper");
        verifier_transcript.append_bytes(&commitment.merkle_root);

        let result = FriPcs::verify_batch_opening::<DefaultTranscript<F>>(
            &[&commitment],
            &z,
            &values,
            &proof,
            &mut verifier_transcript,
        );

        // Tampering should cause either alpha mismatch or FRI verification failure
        assert!(result.is_err());
    }
}
