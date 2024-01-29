use core::borrow::Borrow;

use lambdaworks_math::{
    field::{element::FieldElement, traits::IsField},
    polynomial::Polynomial,
};

use crate::fiat_shamir::transcript::Transcript;

pub trait IsCommitmentScheme<F: IsField> {
    type Commitment;

    fn commit(&self, p: &Polynomial<FieldElement<F>>) -> Self::Commitment;

    fn open(
        &self,
        x: &FieldElement<F>,
        y: &FieldElement<F>,
        p: &Polynomial<FieldElement<F>>,
    ) -> Self::Commitment;
    fn open_batch(
        &self,
        x: &FieldElement<F>,
        y: &[FieldElement<F>],
        p: &[Polynomial<FieldElement<F>>],
        upsilon: &FieldElement<F>,
    ) -> Self::Commitment;

    fn verify(
        &self,
        x: &FieldElement<F>,
        y: &FieldElement<F>,
        p_commitment: &Self::Commitment,
        proof: &Self::Commitment,
    ) -> bool;

    fn verify_batch(
        &self,
        x: &FieldElement<F>,
        ys: &[FieldElement<F>],
        p_commitments: &[Self::Commitment],
        proof: &Self::Commitment,
        upsilon: &FieldElement<F>,
    ) -> bool;
}

// For Non-Hiding
// For batching operations we use a transcript to supply random values. In the case of kzg
// - Using an option for the transcript was the simplest way to enforce domain separation (prover/verifier)
//  for the future I think each protocol should have its own domain separated transcript within its instance variables
pub trait IsPolynomialCommitmentScheme<F: IsField> {
    /// Allows for Univariate vs Multilinear PCS
    type Polynomial;
    /// Point the polynomial is evaluated at
    type Point;
    /// Commitment to a Polynomial
    type Commitment;
    /// Allows for different proof structures
    type Proof;
    /// Unique Errors for the PCS Scheme
    type Error;

    fn commit(&self, p: &Self::Polynomial) -> Result<Self::Commitment, Self::Error>;

    fn open(
        &self,
        point: impl Borrow<Self::Point>,
        eval: &FieldElement<F>,
        poly: &Self::Polynomial,
        transcript: Option<&mut dyn Transcript>,
    ) -> Result<Self::Proof, Self::Error>;

    fn open_batch(
        &self,
        point: impl Borrow<Self::Point>,
        eval: &[FieldElement<F>],
        polys: &[Self::Polynomial],
        transcript: Option<&mut dyn Transcript>,
    ) -> Result<Self::Proof, Self::Error>;

    fn verify(
        &self,
        point: impl Borrow<Self::Point>,
        eval: &FieldElement<F>,
        p_commitment: &Self::Commitment,
        proof: &Self::Proof,
        transcript: Option<&mut dyn Transcript>,
    ) -> Result<bool, Self::Error>;

    fn verify_batch(
        &self,
        point: impl Borrow<Self::Point>,
        evals: &[FieldElement<F>],
        p_commitments: &[Self::Commitment],
        proof: &Self::Proof,
        transcript: Option<&mut dyn Transcript>,
    ) -> Result<bool, Self::Error>;
}
