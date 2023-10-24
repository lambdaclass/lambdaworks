use crate::{
    field::{element::FieldElement, traits::IsField},
    polynomial::{
        multilinear_poly::MultilinearPolynomial,
        multivariate_poly::MultivariatePolynomial,
        dense_term::DenseMonomial, 
        traits::{polynomial::IsPolynomial, term::Term}}, errors::TermError,
};

use super::Polynomial;

pub enum Poly<F: IsField> 
where
    <F as IsField>::BaseType: Send + Sync,
{
    Multilinear(MultilinearPolynomial<F>),
    MultiVariate(MultivariatePolynomial<F>),
}

impl<F: IsField> IsPolynomial<F> for Poly<F> 
where
    <F as IsField>::BaseType: Send + Sync,
{

    fn new(terms: &[dyn Term<F>]) -> Self {
        match self {
            Self::Multilinear(a) => a::new(terms),
            Self::Multivariate(a) => a::new(terms),
            Self::Univariate(a) => a::new(terms),
        }

    }

    /// Adds an additional term to the monomial
    fn extend(&mut self, term: dyn Term<F>) {
        match self {
            Self::Multilinear(a) => a.extend(term),
            Self::Multivariate(a) => a.extend(term),
            Self::Univariate(a) => a.extend(term),
        }

    }

    /// Evaluates `self` at the point `p`.
    fn evaluate(&self, p: &[FieldElement<F>]) -> Result<FieldElement<F>, TermError> {
        match self {
            Self::Multilinear(a) => a.evaluate(p),
            Self::Multivariate(a) => a.evaluate(p),
            Self::Univariate(a) => a.evaluate(p),
        }

    }

    fn partial_evaluate(
        &self,
        assignments: &[(usize, FieldElement<F>)],
    ) -> Result<Self, TermError> {
        match self {
            Self::Multilinear(a) => a.partial_evaluate(assignments),
            Self::Multivariate(a) => a.partial_evaluate(assignments),
            Self::Univariate(a) => a.partial_evaluate(assignments),
        }
    }

}