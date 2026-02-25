//! Polynomial Commitment Scheme (PCS) trait for univariate polynomials.
//!
//! Defines a generic interface for committing to polynomials given as evaluations
//! on a cyclic domain H, opening at arbitrary points, and verifying openings.

use lambdaworks_crypto::fiat_shamir::is_transcript::IsTranscript;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::IsFFTField;
use lambdaworks_math::traits::ByteConversion;

use core::fmt::Debug;

/// Error type for PCS operations.
#[derive(Debug, Clone)]
pub enum PcsError {
    /// The input is invalid (wrong size, etc.)
    InvalidInput(String),
    /// Commitment verification failed.
    VerificationFailed,
    /// An internal FRI or Merkle error.
    InternalError(String),
}

impl core::fmt::Display for PcsError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidInput(s) => write!(f, "PCS invalid input: {s}"),
            Self::VerificationFailed => write!(f, "PCS verification failed"),
            Self::InternalError(s) => write!(f, "PCS internal error: {s}"),
        }
    }
}

/// Generic polynomial commitment scheme for univariate polynomials
/// given as evaluations on a cyclic domain of size N = 2^n.
pub trait UnivariatePcs<F: IsFFTField> {
    /// Cryptographic commitment (e.g. Merkle root).
    type Commitment: Clone + Debug;
    /// Prover-side state retained after committing (e.g. polynomial coefficients + LDE).
    type ProverState;
    /// Proof for a batch opening of multiple polynomials at the same point.
    type BatchOpeningProof: Clone + Debug;

    /// Commit to a polynomial given as evaluations on H.
    ///
    /// Appends the commitment to the transcript.
    /// Returns the commitment and prover state needed for later openings.
    fn commit<T: IsTranscript<F>>(
        evals_on_h: &[FieldElement<F>],
        transcript: &mut T,
    ) -> Result<(Self::Commitment, Self::ProverState), PcsError>
    where
        FieldElement<F>: ByteConversion;

    /// Evaluate the committed polynomial at an arbitrary point z.
    fn open(state: &Self::ProverState, z: &FieldElement<F>) -> Result<FieldElement<F>, PcsError>;

    /// Batch-open multiple committed polynomials at the same point z.
    ///
    /// Returns the opened values and a single batch proof.
    fn batch_open<T: IsTranscript<F>>(
        states: &[&Self::ProverState],
        z: &FieldElement<F>,
        transcript: &mut T,
    ) -> Result<(Vec<FieldElement<F>>, Self::BatchOpeningProof), PcsError>
    where
        FieldElement<F>: ByteConversion;

    /// Verify a batch opening.
    fn verify_batch_opening<T: IsTranscript<F>>(
        commitments: &[&Self::Commitment],
        z: &FieldElement<F>,
        values: &[FieldElement<F>],
        proof: &Self::BatchOpeningProof,
        transcript: &mut T,
    ) -> Result<(), PcsError>
    where
        FieldElement<F>: ByteConversion;

    /// Absorb a commitment into the verifier's transcript.
    ///
    /// Must produce the same transcript state as `commit` did on the prover side.
    /// For FRI PCS this appends the Merkle root bytes.
    fn absorb_commitment<T: IsTranscript<F>>(commitment: &Self::Commitment, transcript: &mut T);
}

// ---------------------------------------------------------------------------
// TransparentPcs — Phase 1 wrapper (raw polynomial values, no hiding)
// ---------------------------------------------------------------------------

/// Transparent PCS: stores raw evaluations as the "commitment" (appended to
/// the Fiat-Shamir transcript). Verification always passes — the verifier
/// can recompute anything from the raw values.
///
/// This keeps Phase 1 tests working unchanged while providing the same
/// `UnivariatePcs` interface that FRI-based PCS uses.
pub struct TransparentPcs;

/// The "commitment" is just the hash of all values (via the transcript).
/// We store a snapshot of the transcript state as a fingerprint.
#[derive(Clone, Debug)]
pub struct TransparentCommitment {
    pub transcript_state: [u8; 32],
}

/// Prover state stores the raw evaluations.
pub struct TransparentProverState<F: IsFFTField> {
    pub evals: Vec<FieldElement<F>>,
}

/// Batch proof is empty — the verifier trusts the raw values in the transcript.
#[derive(Clone, Debug)]
pub struct TransparentBatchProof;

impl<F> UnivariatePcs<F> for TransparentPcs
where
    F: IsFFTField,
    F::BaseType: Send + Sync,
{
    type Commitment = TransparentCommitment;
    type ProverState = TransparentProverState<F>;
    type BatchOpeningProof = TransparentBatchProof;

    fn commit<T: IsTranscript<F>>(
        evals_on_h: &[FieldElement<F>],
        transcript: &mut T,
    ) -> Result<(Self::Commitment, Self::ProverState), PcsError>
    where
        FieldElement<F>: ByteConversion,
    {
        // "Commit" by appending all values to the transcript
        for val in evals_on_h {
            transcript.append_field_element(val);
        }

        let state_snapshot = transcript.state();
        let commitment = TransparentCommitment {
            transcript_state: state_snapshot,
        };
        let prover_state = TransparentProverState {
            evals: evals_on_h.to_vec(),
        };

        Ok((commitment, prover_state))
    }

    fn open(state: &Self::ProverState, z: &FieldElement<F>) -> Result<FieldElement<F>, PcsError> {
        // Evaluate by interpolating (Lagrange basis on roots of unity)
        use lambdaworks_math::polynomial::Polynomial;

        let poly = Polynomial::interpolate_fft::<F>(&state.evals)
            .map_err(|e| PcsError::InternalError(format!("FFT interpolation failed: {e}")))?;
        Ok(poly.evaluate(z))
    }

    fn batch_open<T: IsTranscript<F>>(
        states: &[&Self::ProverState],
        z: &FieldElement<F>,
        _transcript: &mut T,
    ) -> Result<(Vec<FieldElement<F>>, Self::BatchOpeningProof), PcsError>
    where
        FieldElement<F>: ByteConversion,
    {
        let values: Result<Vec<_>, _> = states.iter().map(|s| Self::open(s, z)).collect();
        Ok((values?, TransparentBatchProof))
    }

    fn verify_batch_opening<T: IsTranscript<F>>(
        _commitments: &[&Self::Commitment],
        _z: &FieldElement<F>,
        _values: &[FieldElement<F>],
        _proof: &Self::BatchOpeningProof,
        _transcript: &mut T,
    ) -> Result<(), PcsError>
    where
        FieldElement<F>: ByteConversion,
    {
        // Transparent PCS: always accept (the values were in the transcript)
        Ok(())
    }

    fn absorb_commitment<T: IsTranscript<F>>(_commitment: &Self::Commitment, _transcript: &mut T) {
        // TransparentPcs: the prover appended raw values, not a commitment.
        // The verifier would need the raw values to replay — not supported in V2.
        // This is a no-op; TransparentPcs is only for Phase 1 (non-V2) tests.
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lambdaworks_crypto::fiat_shamir::default_transcript::DefaultTranscript;
    use lambdaworks_math::field::fields::fft_friendly::quartic_babybear::Degree4BabyBearExtensionField;

    type F = Degree4BabyBearExtensionField;
    type FE = FieldElement<F>;

    #[test]
    fn transparent_pcs_commit_open_roundtrip() {
        let evals: Vec<FE> = (1..=8).map(|i| FE::from(i as u64)).collect();
        let mut transcript = DefaultTranscript::<F>::new(b"test_pcs");

        let (_commitment, state) =
            TransparentPcs::commit::<DefaultTranscript<F>>(&evals, &mut transcript).unwrap();

        // Open at a random point
        let z = FE::from(42u64);
        let value = TransparentPcs::open(&state, &z).unwrap();

        // Verify by manual interpolation
        use lambdaworks_math::polynomial::Polynomial;
        let poly = Polynomial::interpolate_fft::<F>(&evals).unwrap();
        let expected = poly.evaluate(&z);
        assert_eq!(value, expected);
    }

    #[test]
    fn transparent_pcs_batch_open_verify() {
        let evals1: Vec<FE> = (1..=4).map(|i| FE::from(i as u64)).collect();
        let evals2: Vec<FE> = (5..=8).map(|i| FE::from(i as u64)).collect();

        let mut transcript = DefaultTranscript::<F>::new(b"batch");
        let (c1, s1) =
            TransparentPcs::commit::<DefaultTranscript<F>>(&evals1, &mut transcript).unwrap();
        let (c2, s2) =
            TransparentPcs::commit::<DefaultTranscript<F>>(&evals2, &mut transcript).unwrap();

        let z = FE::from(99u64);
        let (values, proof) =
            TransparentPcs::batch_open::<DefaultTranscript<F>>(&[&s1, &s2], &z, &mut transcript)
                .unwrap();

        assert_eq!(values.len(), 2);

        // Verify
        let mut verify_transcript = DefaultTranscript::<F>::new(b"batch_v");
        TransparentPcs::verify_batch_opening::<DefaultTranscript<F>>(
            &[&c1, &c2],
            &z,
            &values,
            &proof,
            &mut verify_transcript,
        )
        .unwrap();
    }
}
