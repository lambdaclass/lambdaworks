use lambdaworks_math::{
    field::{element::FieldElement, traits::IsField},
    polynomial::Polynomial,
};

pub trait IsCommitmentScheme<F: IsField> {
    type Commitment;

    /// Create a commitment to a polynomial
    fn commit(&self, p: &Polynomial<FieldElement<F>>) -> Self::Commitment;

    /// Create an evaluation proof for a polynomial at x equal to y
    /// p(x) = y
    fn open(
        &self,
        x: &FieldElement<F>,
        y: &FieldElement<F>,
        p: &Polynomial<FieldElement<F>>,
    ) -> Self::Commitment;

    /// Create an evaluation proof for a slice of polynomials at x equal to y_i
    /// that is, we show that we evaluated correctly p_i (x) = y_i
    fn open_batch(
        &self,
        x: &FieldElement<F>,
        y: &[FieldElement<F>],
        p: &[Polynomial<FieldElement<F>>],
        upsilon: &FieldElement<F>,
    ) -> Self::Commitment;

    /// Verify an evaluation proof given the commitment to p, the point x and the evaluation y
    fn verify(
        &self,
        x: &FieldElement<F>,
        y: &FieldElement<F>,
        p_commitment: &Self::Commitment,
        proof: &Self::Commitment,
    ) -> bool;

    /// Verify an evaluation proof given the commitments to p, the point x and the evaluations ys
    fn verify_batch(
        &self,
        x: &FieldElement<F>,
        ys: &[FieldElement<F>],
        p_commitments: &[Self::Commitment],
        proof: &Self::Commitment,
        upsilon: &FieldElement<F>,
    ) -> bool;
}
