//! FRI (Fast Reed-Solomon Interactive Oracle Proof) as a Polynomial Commitment Scheme.
//!
//! This module provides a wrapper around the FRI implementation in the STARK prover,
//! allowing it to be used through the [`PolynomialCommitmentScheme`] interface.
//!
//! # Note
//!
//! FRI is primarily designed for use within STARK provers. This wrapper provides
//! a unified interface for contexts where a transparent (no trusted setup)
//! polynomial commitment scheme is needed.
//!
//! For STARK proving, it's recommended to use the FRI functions directly from
//! `lambdaworks_stark::fri` for better performance and integration.
//!
//! # Overview
//!
//! FRI is a hash-based polynomial commitment scheme with:
//! - **No trusted setup**: Only uses hash functions (transparent)
//! - **Logarithmic proof size**: O(log n) for degree-n polynomials
//! - **Plausibly post-quantum secure**: Based on hash function security
//!
//! # Differences from KZG
//!
//! Unlike KZG which produces constant-size proofs, FRI produces logarithmic-size
//! proofs but doesn't require a trusted setup ceremony.

use crate::pcs::error::PCSError;
use crate::pcs::traits::PolynomialCommitmentScheme;

use core::marker::PhantomData;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::IsFFTField;
use lambdaworks_math::polynomial::Polynomial;

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

/// FRI Polynomial Commitment Scheme wrapper.
///
/// This wraps the FRI protocol from the STARK prover to provide
/// a [`PolynomialCommitmentScheme`] interface.
///
/// # Type Parameters
///
/// - `F`: An FFT-friendly field for polynomial evaluation.
#[derive(Clone, Debug)]
pub struct FRIPcs<F: IsFFTField> {
    /// Number of FRI layers (determines folding factor).
    num_layers: usize,
    /// Blowup factor for the low-degree extension.
    blowup_factor: usize,
    /// Maximum supported polynomial degree.
    max_degree: usize,
    _field: PhantomData<F>,
}

impl<F: IsFFTField> FRIPcs<F> {
    /// Create a new FRI PCS instance.
    ///
    /// # Arguments
    ///
    /// * `max_degree` - Maximum polynomial degree to support.
    /// * `blowup_factor` - LDE blowup factor (typically 2, 4, 8, or 16).
    /// * `num_layers` - Number of FRI folding layers.
    pub fn new(max_degree: usize, blowup_factor: usize, num_layers: usize) -> Self {
        Self {
            num_layers,
            blowup_factor,
            max_degree,
            _field: PhantomData,
        }
    }

    /// Returns the blowup factor.
    pub fn blowup_factor(&self) -> usize {
        self.blowup_factor
    }

    /// Returns the number of FRI layers.
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }
}

/// FRI Public Parameters.
///
/// FRI doesn't require a trusted setup, so these parameters
/// just encode the configuration.
#[derive(Clone, Debug)]
pub struct FRIPublicParams {
    /// Maximum polynomial degree supported.
    pub max_degree: usize,
    /// Blowup factor for the LDE.
    pub blowup_factor: usize,
    /// Number of FRI layers.
    pub num_layers: usize,
    /// Number of queries for soundness.
    pub num_queries: usize,
}

impl FRIPublicParams {
    /// Create new FRI public parameters.
    pub fn new(
        max_degree: usize,
        blowup_factor: usize,
        num_layers: usize,
        num_queries: usize,
    ) -> Self {
        Self {
            max_degree,
            blowup_factor,
            num_layers,
            num_queries,
        }
    }
}

/// FRI Committer Key.
///
/// Contains parameters needed by the prover.
#[derive(Clone, Debug)]
pub struct FRICommitterKey {
    /// Maximum polynomial degree.
    pub max_degree: usize,
    /// Blowup factor.
    pub blowup_factor: usize,
    /// Number of layers.
    pub num_layers: usize,
}

/// FRI Verifier Key.
///
/// Contains parameters needed by the verifier.
#[derive(Clone, Debug)]
pub struct FRIVerifierKey {
    /// Maximum polynomial degree.
    pub max_degree: usize,
    /// Blowup factor.
    pub blowup_factor: usize,
    /// Number of layers.
    pub num_layers: usize,
    /// Number of queries.
    pub num_queries: usize,
}

/// FRI Commitment - a Merkle root.
///
/// The commitment is the root of a Merkle tree over the
/// polynomial evaluations on a coset of a subgroup.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FRICommitment {
    /// The Merkle root of the initial layer.
    pub root: Vec<u8>,
}

impl FRICommitment {
    /// Create a new FRI commitment from a Merkle root.
    pub fn new(root: Vec<u8>) -> Self {
        Self { root }
    }
}

/// FRI Commitment State.
///
/// Stores intermediate data from the commit phase needed for opening.
#[derive(Clone, Debug)]
pub struct FRICommitmentState<F: IsFFTField> {
    /// The evaluations of the polynomial on the LDE domain.
    pub evaluations: Vec<FieldElement<F>>,
    /// The original polynomial (needed for computing query responses).
    pub polynomial: Polynomial<FieldElement<F>>,
}

/// FRI Opening Proof.
///
/// Contains the FRI layers and decommitment information.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FRIProof {
    /// Merkle roots of each FRI layer.
    pub layer_roots: Vec<Vec<u8>>,
    /// The final constant polynomial value.
    pub final_value: Vec<u8>,
    /// Query indices (derived from Fiat-Shamir).
    pub query_indices: Vec<usize>,
    /// Authentication paths for queries.
    pub query_proofs: Vec<Vec<u8>>,
}

impl<F: IsFFTField> PolynomialCommitmentScheme<F> for FRIPcs<F> {
    type PublicParameters = FRIPublicParams;
    type CommitterKey = FRICommitterKey;
    type VerifierKey = FRIVerifierKey;
    type Commitment = FRICommitment;
    type CommitmentState = FRICommitmentState<F>;
    type Proof = FRIProof;

    fn max_degree(&self) -> usize {
        self.max_degree
    }

    #[cfg(feature = "std")]
    fn setup<R: rand::RngCore>(
        max_degree: usize,
        _rng: &mut R,
    ) -> Result<Self::PublicParameters, PCSError> {
        // FRI doesn't need randomness for setup - it's transparent.
        // We use reasonable defaults for the parameters.
        let blowup_factor = 4; // Common choice
        let num_layers =
            (max_degree.next_power_of_two().trailing_zeros() as usize).saturating_sub(1);
        let num_queries = 30; // ~80-bit security with blowup_factor=4

        Ok(FRIPublicParams::new(
            max_degree,
            blowup_factor,
            num_layers,
            num_queries,
        ))
    }

    fn trim(
        pp: &Self::PublicParameters,
        supported_degree: usize,
    ) -> Result<(Self::CommitterKey, Self::VerifierKey), PCSError> {
        if supported_degree > pp.max_degree {
            return Err(PCSError::degree_too_large(pp.max_degree, supported_degree));
        }

        let ck = FRICommitterKey {
            max_degree: supported_degree,
            blowup_factor: pp.blowup_factor,
            num_layers: pp.num_layers,
        };

        let vk = FRIVerifierKey {
            max_degree: supported_degree,
            blowup_factor: pp.blowup_factor,
            num_layers: pp.num_layers,
            num_queries: pp.num_queries,
        };

        Ok((ck, vk))
    }

    fn commit(
        ck: &Self::CommitterKey,
        polynomial: &Polynomial<FieldElement<F>>,
    ) -> Result<(Self::Commitment, Self::CommitmentState), PCSError> {
        let degree = polynomial.degree();
        if degree > ck.max_degree {
            return Err(PCSError::degree_too_large(ck.max_degree, degree));
        }

        // TODO: Implement actual FRI commitment.
        // This would:
        // 1. Evaluate the polynomial on a coset (LDE domain)
        // 2. Build a Merkle tree over the evaluations
        // 3. Return the Merkle root as commitment

        Err(PCSError::commitment(
            "FRI commit wrapper not yet fully implemented - use stark::fri directly",
        ))
    }

    fn open(
        _ck: &Self::CommitterKey,
        _polynomial: &Polynomial<FieldElement<F>>,
        _commitment_state: &Self::CommitmentState,
        _point: &FieldElement<F>,
    ) -> Result<Self::Proof, PCSError> {
        // TODO: Implement FRI opening.
        // This would run the FRI protocol to prove the evaluation.

        Err(PCSError::opening(
            "FRI open wrapper not yet fully implemented - use stark::fri directly",
        ))
    }

    fn verify(
        _vk: &Self::VerifierKey,
        _commitment: &Self::Commitment,
        _point: &FieldElement<F>,
        _evaluation: &FieldElement<F>,
        _proof: &Self::Proof,
    ) -> Result<bool, PCSError> {
        // TODO: Implement FRI verification.
        // This would verify the FRI decommitment.

        Err(PCSError::verification(
            "FRI verify wrapper not yet fully implemented - use stark::fri directly",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lambdaworks_math::field::fields::u64_prime_field::U64PrimeField;

    // A simple test field
    type TestField = U64PrimeField<65537>;

    #[test]
    fn test_fri_pcs_creation() {
        let fri: FRIPcs<TestField> = FRIPcs::new(1024, 4, 10);
        assert_eq!(fri.max_degree(), 1024);
        assert_eq!(fri.blowup_factor(), 4);
        assert_eq!(fri.num_layers(), 10);
    }

    #[test]
    fn test_fri_public_params() {
        let pp = FRIPublicParams::new(1024, 4, 10, 30);
        assert_eq!(pp.max_degree, 1024);
        assert_eq!(pp.blowup_factor, 4);
    }
}
