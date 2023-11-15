use lambdaworks_math::{
    field::{element::FieldElement, traits::IsField},
    polynomial::univariate::UnivariatePolynomial,
};

pub trait IsCommitmentScheme<F: IsField> {
    type Commitment;

    fn commit(&self, p: &UnivariatePolynomial<F>) -> Self::Commitment;

    fn open(
        &self,
        x: &FieldElement<F>,
        y: &FieldElement<F>,
        p: &UnivariatePolynomial<F>,
    ) -> Self::Commitment;
    fn open_batch(
        &self,
        x: &FieldElement<F>,
        y: &[FieldElement<F>],
        p: &[UnivariatePolynomial<F>],
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
