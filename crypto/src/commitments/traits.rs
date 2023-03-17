use lambdaworks_math::{polynomial::Polynomial, field::{traits::IsField, element::FieldElement}, traits::ByteConversion};


pub trait IsCommitmentScheme<F: IsField> {
    type Hiding;
    type Opening;

    fn commit(&self, p: &Polynomial<FieldElement<F>>) -> Self::Hiding;
    fn open(&self, x: &FieldElement<F>, p: &Polynomial<FieldElement<F>>) -> Self::Opening;
    fn verify(&self, opening: &Self::Opening, x: &FieldElement<F>, p_commitment: &Self::Hiding) -> bool;    
}
