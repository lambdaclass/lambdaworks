use lambdaworks_math::{
    field::{element::FieldElement, traits::IsField},
    polynomial::Polynomial,
};
use std::borrow::Borrow;

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

pub trait PolynomialCommitmentScheme {
    // Abstracting over Polynomial allows us to have batched and non-batched PCS
    type Polynomial;
    type Commitment;
    type Evaluation;
    type Challenge;
    type Proof;
    type Error;

    type ProverKey;
    type CommitmentKey;
    type VerifierKey;

    //TODO: convert to impl IntoIterator<Item = Self::Polynomial>
    fn commit(
        poly: Self::Polynomial,
        ck: impl Borrow<Self::CommitmentKey>,
    ) -> Result<Self::Commitment, Self::Error>;

    fn prove(
        poly: Self::Polynomial,
        evals: Self::Evaluation,
        challenges: Self::Challenge,
        pk: impl Borrow<Self::ProverKey>,
        transcript: &mut impl Transcript,
    ) -> Result<Self::Proof, Self::Error>;

    fn verify(
        commitments: Self::Commitment,
        evals: Self::Evaluation,
        challenges: Self::Challenge,
        vk: impl Borrow<Self::VerifierKey>,
        transcript: &mut impl Transcript,
        proof: Self::Proof,
    ) -> Result<(), Self::Error>;
}
