use crate::field::element::FieldElement;
use crate::field::traits::{IsField, IsPrimeField};
use crate::polynomial::multilinear_term::MultiLinearMonomial;
use crate::polynomial::term::Term;

/// Represents a multilinear polynomials as a collection of multilinear monomials
pub struct MultilinearPolynomial<F: IsField + IsPrimeField>
where
    <F as IsField>::BaseType: Send + Sync,
{
    pub terms: Vec<MultiLinearMonomial<F>>,
}

impl<F: IsField + IsPrimeField> MultilinearPolynomial<F>
where
    <F as IsField>::BaseType: Send + Sync,
{
    /// Build a new multilinear polynomial, from collection of multilinear monomials
    fn new(terms: Vec<MultiLinearMonomial<F>>) -> Self {
        Self { terms }
    }

    /// Selectively assign values to variables in the polynomial, returns a reduced
    /// polynomial after assignment evaluation
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