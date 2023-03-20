use lambdaworks_math::{
    field::{element::FieldElement, traits::IsField},
    polynomial::Polynomial,
    traits::ByteConversion,
};

pub trait IsCommitmentScheme<F: IsField> {
    type Hiding;
    type Opening;

    fn commit(&self, p: &Polynomial<FieldElement<F>>) -> Self::Hiding;
    fn open(&self, x: &FieldElement<F>, p: &Polynomial<FieldElement<F>>) -> Self::Opening;
    fn verify(
        &self,
        opening: &Self::Opening,
        x: &FieldElement<F>,
        p_commitment: &Self::Hiding,
    ) -> bool;
}
