use crate::field::element::FieldElement;
use crate::field::traits::{IsField, IsPrimeField};
use crate::polynomial::term::{MultiLinearMonomial, MultiLinearTerm};

// TODO: add documentation
pub struct SparseMultilinearPolynomial<F: IsField + IsPrimeField + Default>
where
    <F as IsField>::BaseType: Send + Sync,
{
    pub terms: Vec<MultiLinearMonomial<F>>,
}

impl<F: IsField + IsPrimeField + Default> SparseMultilinearPolynomial<F>
where
    <F as IsField>::BaseType: Send + Sync,
{
    // TODO: add documentation
    fn new(terms: Vec<MultiLinearMonomial<F>>) -> Self {
        Self { terms }
    }

    // TODO: add documentation
    fn partial_evaluate(&self, assignments: &[(usize, FieldElement<F>)]) -> Self {
        let updated_monomials: Vec<MultiLinearMonomial<F>> = self
            .terms
            .iter()
            .map(|term| term.partial_evaluate(assignments))
            .collect();
        Self::new(updated_monomials)
    }
}

// TODO: add tests