//! Core traits for Polynomial Commitment Schemes.
//!
//! This module defines the trait hierarchy for PCS implementations:
//!
//! - [`PolynomialCommitmentScheme`]: Base trait with setup, commit, open, verify
//! - [`BatchPCS`]: Extension for batch operations
//! - [`SerializablePCS`]: Extension for serialization support
//!
//! # Design
//!
//! Inspired by [arkworks poly-commit](https://github.com/arkworks-rs/poly-commit)
//! and [Plonky3](https://github.com/Plonky3/Plonky3), this design separates:
//!
//! - **PublicParameters**: Global parameters from setup (e.g., SRS)
//! - **CommitterKey**: Prover-side key for committing and opening
//! - **VerifierKey**: Verifier-side key for verification
//! - **Commitment**: The polynomial commitment
//! - **CommitmentState**: Auxiliary state from commitment (e.g., randomness)
//! - **Proof**: Opening proof

use crate::pcs::error::PCSError;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::IsField;
use lambdaworks_math::polynomial::Polynomial;

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

/// Base trait for Polynomial Commitment Schemes.
///
/// A polynomial commitment scheme allows a prover to commit to a polynomial
/// and later prove evaluations at specific points.
///
/// # Type Parameters
///
/// - `F`: The field over which polynomials are defined.
///
/// # Example
///
/// ```ignore
/// // Setup phase (typically done once)
/// let pp = MyPCS::setup(max_degree, &mut rng)?;
/// let (ck, vk) = MyPCS::trim(&pp, supported_degree)?;
///
/// // Commit phase
/// let (commitment, state) = MyPCS::commit(&ck, &polynomial)?;
///
/// // Open phase
/// let proof = MyPCS::open(&ck, &polynomial, &state, &point)?;
///
/// // Verify phase
/// let evaluation = polynomial.evaluate(&point);
/// let valid = MyPCS::verify(&vk, &commitment, &point, &evaluation, &proof)?;
/// ```
pub trait PolynomialCommitmentScheme<F: IsField> {
    /// Public parameters generated during setup.
    ///
    /// For KZG, this is the Structured Reference String (SRS).
    /// For FRI, this contains the configuration and domain information.
    type PublicParameters;

    /// Key used by the prover to commit and create opening proofs.
    ///
    /// This is typically a subset of the public parameters optimized
    /// for the prover's operations.
    type CommitterKey;

    /// Key used by the verifier to verify opening proofs.
    ///
    /// This is typically a minimal subset of the public parameters
    /// needed for verification.
    type VerifierKey;

    /// The commitment to a polynomial.
    ///
    /// For KZG, this is an elliptic curve point.
    /// For FRI, this is a Merkle root.
    type Commitment: Clone + PartialEq + Eq;

    /// Auxiliary state stored after commitment.
    ///
    /// This may include randomness used for hiding commitments,
    /// or intermediate values needed for efficient opening.
    type CommitmentState;

    /// Proof of correct evaluation.
    ///
    /// For KZG, this is an elliptic curve point (quotient commitment).
    /// For FRI, this contains Merkle proofs and layer evaluations.
    type Proof;

    /// Returns the maximum polynomial degree supported by this PCS instance.
    fn max_degree(&self) -> usize;

    /// Generate public parameters for the scheme.
    ///
    /// # Arguments
    ///
    /// * `max_degree` - Maximum polynomial degree to support.
    /// * `rng` - Random number generator for setup.
    ///
    /// # Errors
    ///
    /// Returns [`PCSError::SetupError`] if setup fails.
    #[cfg(feature = "std")]
    fn setup<R: rand::RngCore>(
        max_degree: usize,
        rng: &mut R,
    ) -> Result<Self::PublicParameters, PCSError>;

    /// Specialize public parameters for a specific degree bound.
    ///
    /// This produces separate keys for the prover and verifier,
    /// potentially reducing their size for the specific use case.
    ///
    /// # Arguments
    ///
    /// * `pp` - Public parameters from setup.
    /// * `supported_degree` - Maximum degree to support (must be <= setup degree).
    ///
    /// # Errors
    ///
    /// Returns [`PCSError::DegreeTooLarge`] if `supported_degree` exceeds
    /// the maximum degree in `pp`.
    fn trim(
        pp: &Self::PublicParameters,
        supported_degree: usize,
    ) -> Result<(Self::CommitterKey, Self::VerifierKey), PCSError>;

    /// Commit to a polynomial.
    ///
    /// # Arguments
    ///
    /// * `ck` - Committer key.
    /// * `polynomial` - The polynomial to commit to.
    ///
    /// # Returns
    ///
    /// A tuple of (commitment, commitment_state).
    ///
    /// # Errors
    ///
    /// Returns [`PCSError::DegreeTooLarge`] if the polynomial degree exceeds
    /// the supported degree, or [`PCSError::CommitmentError`] for other failures.
    fn commit(
        ck: &Self::CommitterKey,
        polynomial: &Polynomial<FieldElement<F>>,
    ) -> Result<(Self::Commitment, Self::CommitmentState), PCSError>;

    /// Create an opening proof for a polynomial at a point.
    ///
    /// # Arguments
    ///
    /// * `ck` - Committer key.
    /// * `polynomial` - The committed polynomial.
    /// * `commitment_state` - State from the commit phase.
    /// * `point` - The evaluation point.
    ///
    /// # Errors
    ///
    /// Returns [`PCSError::OpeningError`] if proof generation fails.
    fn open(
        ck: &Self::CommitterKey,
        polynomial: &Polynomial<FieldElement<F>>,
        commitment_state: &Self::CommitmentState,
        point: &FieldElement<F>,
    ) -> Result<Self::Proof, PCSError>;

    /// Verify an opening proof.
    ///
    /// # Arguments
    ///
    /// * `vk` - Verifier key.
    /// * `commitment` - The polynomial commitment.
    /// * `point` - The evaluation point.
    /// * `evaluation` - The claimed evaluation p(point).
    /// * `proof` - The opening proof.
    ///
    /// # Returns
    ///
    /// `Ok(true)` if the proof is valid, `Ok(false)` if invalid,
    /// or an error if verification cannot be performed.
    fn verify(
        vk: &Self::VerifierKey,
        commitment: &Self::Commitment,
        point: &FieldElement<F>,
        evaluation: &FieldElement<F>,
        proof: &Self::Proof,
    ) -> Result<bool, PCSError>;

    /// Commit to multiple polynomials.
    ///
    /// Default implementation commits to each polynomial individually.
    /// Implementations may override for better efficiency.
    #[cfg(feature = "alloc")]
    fn commit_batch(
        ck: &Self::CommitterKey,
        polynomials: &[Polynomial<FieldElement<F>>],
    ) -> Result<(Vec<Self::Commitment>, Vec<Self::CommitmentState>), PCSError> {
        let results: Result<Vec<_>, _> = polynomials.iter().map(|p| Self::commit(ck, p)).collect();

        results.map(|v| v.into_iter().unzip())
    }
}

/// Extension trait for batch opening and verification.
///
/// This trait provides efficient batch operations for PCS implementations
/// that support them natively (e.g., via random linear combinations).
#[cfg(feature = "alloc")]
pub trait BatchPCS<F: IsField>: PolynomialCommitmentScheme<F> {
    /// Batch proof type.
    ///
    /// May be the same as `Self::Proof` or a more efficient representation
    /// for multiple openings.
    type BatchProof;

    /// Open multiple polynomials at a single point.
    ///
    /// # Arguments
    ///
    /// * `ck` - Committer key.
    /// * `polynomials` - The committed polynomials.
    /// * `commitment_states` - States from the commit phase.
    /// * `point` - The evaluation point.
    /// * `challenge` - Random challenge for combining polynomials.
    ///
    /// # Errors
    ///
    /// Returns [`PCSError::LengthMismatch`] if polynomials and states
    /// have different lengths.
    fn batch_open_single_point(
        ck: &Self::CommitterKey,
        polynomials: &[Polynomial<FieldElement<F>>],
        commitment_states: &[Self::CommitmentState],
        point: &FieldElement<F>,
        challenge: &FieldElement<F>,
    ) -> Result<Self::BatchProof, PCSError>;

    /// Verify a batch opening proof at a single point.
    ///
    /// # Arguments
    ///
    /// * `vk` - Verifier key.
    /// * `commitments` - The polynomial commitments.
    /// * `point` - The evaluation point.
    /// * `evaluations` - The claimed evaluations.
    /// * `proof` - The batch opening proof.
    /// * `challenge` - Random challenge (must match the one used in opening).
    fn batch_verify_single_point(
        vk: &Self::VerifierKey,
        commitments: &[Self::Commitment],
        point: &FieldElement<F>,
        evaluations: &[FieldElement<F>],
        proof: &Self::BatchProof,
        challenge: &FieldElement<F>,
    ) -> Result<bool, PCSError>;

    /// Open multiple polynomials at multiple points.
    ///
    /// # Arguments
    ///
    /// * `ck` - Committer key.
    /// * `polynomials` - The committed polynomials.
    /// * `commitment_states` - States from the commit phase.
    /// * `points` - The evaluation points.
    /// * `challenge` - Random challenge for combining.
    fn batch_open_multi_point(
        ck: &Self::CommitterKey,
        polynomials: &[Polynomial<FieldElement<F>>],
        commitment_states: &[Self::CommitmentState],
        points: &[FieldElement<F>],
        challenge: &FieldElement<F>,
    ) -> Result<Self::BatchProof, PCSError>;

    /// Verify a batch opening proof at multiple points.
    ///
    /// # Arguments
    ///
    /// * `vk` - Verifier key.
    /// * `commitments` - The polynomial commitments.
    /// * `points` - The evaluation points.
    /// * `evaluations` - The claimed evaluations (one vector per polynomial).
    /// * `proof` - The batch opening proof.
    /// * `challenge` - Random challenge.
    fn batch_verify_multi_point(
        vk: &Self::VerifierKey,
        commitments: &[Self::Commitment],
        points: &[FieldElement<F>],
        evaluations: &[Vec<FieldElement<F>>],
        proof: &Self::BatchProof,
        challenge: &FieldElement<F>,
    ) -> Result<bool, PCSError>;
}

/// Extension trait for serializable PCS types.
///
/// Implementations should ensure that serialization is deterministic
/// and that deserialization validates the data.
#[cfg(feature = "alloc")]
pub trait SerializablePCS<F: IsField>: PolynomialCommitmentScheme<F> {
    /// Serialize a commitment to bytes.
    fn serialize_commitment(commitment: &Self::Commitment) -> Vec<u8>;

    /// Deserialize a commitment from bytes.
    fn deserialize_commitment(bytes: &[u8]) -> Result<Self::Commitment, PCSError>;

    /// Serialize a proof to bytes.
    fn serialize_proof(proof: &Self::Proof) -> Vec<u8>;

    /// Deserialize a proof from bytes.
    fn deserialize_proof(bytes: &[u8]) -> Result<Self::Proof, PCSError>;

    /// Serialize public parameters to bytes.
    fn serialize_public_params(pp: &Self::PublicParameters) -> Vec<u8>;

    /// Deserialize public parameters from bytes.
    fn deserialize_public_params(bytes: &[u8]) -> Result<Self::PublicParameters, PCSError>;

    /// Serialize committer key to bytes.
    fn serialize_committer_key(ck: &Self::CommitterKey) -> Vec<u8>;

    /// Deserialize committer key from bytes.
    fn deserialize_committer_key(bytes: &[u8]) -> Result<Self::CommitterKey, PCSError>;

    /// Serialize verifier key to bytes.
    fn serialize_verifier_key(vk: &Self::VerifierKey) -> Vec<u8>;

    /// Deserialize verifier key from bytes.
    fn deserialize_verifier_key(bytes: &[u8]) -> Result<Self::VerifierKey, PCSError>;
}

/// Marker trait for PCS implementations that support hiding commitments.
///
/// Hiding commitments provide computational hiding, meaning that the
/// commitment reveals no information about the polynomial (under
/// computational assumptions).
pub trait HidingPCS<F: IsField>: PolynomialCommitmentScheme<F> {
    /// Commit to a polynomial with hiding.
    ///
    /// # Arguments
    ///
    /// * `ck` - Committer key.
    /// * `polynomial` - The polynomial to commit to.
    /// * `hiding_bound` - Number of evaluations that can be revealed
    ///   while maintaining hiding (for schemes that support bounded hiding).
    /// * `rng` - Random number generator for blinding factors.
    #[cfg(feature = "std")]
    fn commit_hiding<R: rand::RngCore>(
        ck: &Self::CommitterKey,
        polynomial: &Polynomial<FieldElement<F>>,
        hiding_bound: Option<usize>,
        rng: &mut R,
    ) -> Result<(Self::Commitment, Self::CommitmentState), PCSError>;
}
