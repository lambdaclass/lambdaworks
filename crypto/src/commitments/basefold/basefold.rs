use core::{borrow::Borrow, marker::PhantomData};

use lambdaworks_math::{
    field::{element::FieldElement, traits::IsField},
    polynomial::dense_multilinear_poly::DenseMultilinearPolynomial,
};

use crate::{
    commitments::traits::IsPolynomialCommitmentScheme, fiat_shamir::transcript::Transcript,
};

pub struct BaseFold<F: IsField> {
    phantom: PhantomData<F>,
}
impl<F: IsField> IsPolynomialCommitmentScheme<F> for BaseFold<F>
where
    <F as IsField>::BaseType: Send + Sync,
{
    /// Allows for Univariate vs Multilinear PCS
    type Polynomial = DenseMultilinearPolynomial<F>;
    /// Point the polynomial is evaluated at
    type Point = FieldElement<F>;
    /// Commitment to a Polynomial
    type Commitment = FieldElement<F>;
    /// Allows for different proof structures
    type Proof = BaseFoldProof;
    /// Unique Errors for the PCS Scheme
    type Error = BaseFoldError;

    fn commit(&self, p: &Self::Polynomial) -> Result<Self::Commitment, Self::Error> {}

    fn open(
        &self,
        point: impl Borrow<Self::Point>,
        eval: &FieldElement<F>,
        poly: &Self::Polynomial,
        transcript: Option<&mut dyn Transcript>,
    ) -> Result<Self::Proof, Self::Error> {
    }

    fn open_batch(
        &self,
        point: impl Borrow<Self::Point>,
        eval: &[FieldElement<F>],
        polys: &[Self::Polynomial],
        transcript: Option<&mut dyn Transcript>,
    ) -> Result<Self::Proof, Self::Error> {
    }

    fn verify(
        &self,
        point: impl Borrow<Self::Point>,
        eval: &FieldElement<F>,
        p_commitment: &Self::Commitment,
        proof: &Self::Proof,
        transcript: Option<&mut dyn Transcript>,
    ) -> Result<bool, Self::Error> {
    }

    fn verify_batch(
        &self,
        point: impl Borrow<Self::Point>,
        evals: &[FieldElement<F>],
        p_commitments: &[Self::Commitment],
        proof: &Self::Proof,
        transcript: Option<&mut dyn Transcript>,
    ) -> Result<bool, Self::Error> {
    }
}
