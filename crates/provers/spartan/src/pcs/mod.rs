//! Polynomial commitment scheme (PCS) traits and implementations.
//!
//! Provides the IsMultilinearPCS trait for committing to multilinear polynomials
//! and the TrivialPCS implementation for testing.

use lambdaworks_math::field::{element::FieldElement, traits::IsField};
use lambdaworks_math::polynomial::dense_multilinear_poly::DenseMultilinearPolynomial;

/// Error type for PCS operations.
#[derive(Debug, thiserror::Error)]
#[error("PCS error: {0}")]
pub struct PcsError(pub String);

/// Trait for multilinear polynomial commitment schemes.
///
/// Defines the interface for committing to a multilinear polynomial,
/// opening it at a point, and verifying the opening.
///
/// All implementations must also implement `serialize_commitment` so that
/// the Fiat-Shamir transcript can absorb the commitment before challenges
/// are drawn — a requirement for protocol soundness.
pub trait IsMultilinearPCS<F: IsField>
where
    F::BaseType: Send + Sync,
{
    type Commitment: Clone;
    type Proof: Clone;
    type Error: std::error::Error;

    /// Commit to a multilinear polynomial.
    fn commit(&self, poly: &DenseMultilinearPolynomial<F>)
        -> Result<Self::Commitment, Self::Error>;

    /// Open the polynomial at a point, returning the value and a proof.
    fn open(
        &self,
        poly: &DenseMultilinearPolynomial<F>,
        point: &[FieldElement<F>],
    ) -> Result<(FieldElement<F>, Self::Proof), Self::Error>;

    /// Verify an opening proof.
    fn verify(
        &self,
        commitment: &Self::Commitment,
        point: &[FieldElement<F>],
        value: &FieldElement<F>,
        proof: &Self::Proof,
    ) -> Result<bool, Self::Error>;

    /// Serialize a commitment to bytes for Fiat-Shamir transcript absorption.
    ///
    /// Must be implemented so that the prover and verifier can both absorb the
    /// commitment into the transcript before drawing challenges.
    fn serialize_commitment(commitment: &Self::Commitment) -> Vec<u8>;
}

// TODO: rename this module to `trivial` to reflect that it contains TrivialPCS,
// not an actual Zeromorph implementation. The Zeromorph PCS (KZG-based) should
// be added as a separate module when implementing production-grade proving.
pub mod zeromorph;
