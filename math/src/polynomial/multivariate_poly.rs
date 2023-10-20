use crate::field::element::FieldElement;
use crate::field::traits::IsField;

use super::{multivariate_term::MultivariateMonomial, term::Term};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MultivariatePolynomial<F: IsField + Default>
where
    <F as IsField>::BaseType: Send + Sync,
{
    pub terms: Vec<MultivariateMonomial<F>>,
}

impl<F: IsField + Default> MultivariatePolynomial<F>
where
    <F as IsField>::BaseType: Send + Sync,
{
    pub fn new(terms: &[MultivariateMonomial<F>]) -> Self {
        Self {
            terms: terms.to_vec(),
        }
    }

    pub const fn zero() -> Self {
        Self { terms: Vec::new() }
    }

    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }

    pub fn len(&self) -> usize {
        self.terms.len()
    }

    pub fn terms(&self) -> &[MultivariateMonomial<F>] {
        &self.terms
    }

    pub fn to_vec(&self) -> Vec<MultivariateMonomial<F>> {
        self.terms.clone()
    }

    pub fn iter(&self) -> impl Iterator<Item = &MultivariateMonomial<F>> {
        self.terms.iter()
    }

    /// Evaluates `self` at the point `p`.
    pub fn evaluate(&self, p: &[FieldElement<F>]) -> FieldElement<F> {
        self.terms
            .iter()
            .fold(FieldElement::<F>::zero(), |mut acc, term| {
                acc += term.evaluate(p);
                acc
            })
    }

    pub fn partial_evaluate(&self, assignments: &[(usize, FieldElement<F>)]) -> Self {
        let updated_monomials: Vec<MultivariateMonomial<F>> = self
            .terms
            .iter()
            .map(|term| term.partial_evaluate(assignments))
            .collect();
        Self::new(&updated_monomials)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::element::FieldElement;
    use crate::field::fields::u64_prime_field::U64PrimeField;
    use crate::polynomial::multivariate_term::MultivariateMonomial;

    const ORDER: u64 = 101;
    type F = U64PrimeField<ORDER>;
    type FE = FieldElement<F>;

    #[test]
    fn test_partial_evaluation() {
        // 3ab + 4bc
        // partially evaluate b = 2
        // expected result = 6a + 8c
        // a = 1, b = 2, c = 3
        let poly = MultivariatePolynomial::new(&[
            MultivariateMonomial::new((FE::new(3), vec![(1, 1), (2, 1)])),
            MultivariateMonomial::new((FE::new(4), vec![(2, 1), (3, 1)])),
        ]);
        let result = poly.partial_evaluate(&[(2, FE::new(2))]);
        assert_eq!(
            result,
            MultivariatePolynomial {
                terms: vec![
                    MultivariateMonomial {
                        coeff: FE::new(6),
                        vars: vec![(1, 1)]
                    },
                    MultivariateMonomial {
                        coeff: FE::new(8),
                        vars: vec![(3, 1)]
                    }
                ]
            }
        );
    }

    #[test]
    fn test_all_vars_evaluation() {
        // 3abc + 4abc
        // evaluate: a = 1, b = 2, c = 3
        // expected result = 42
        // a = 1, b = 2, c = 3
        let poly = MultivariatePolynomial::new(&[
            MultivariateMonomial::new((FE::new(3), vec![(0, 1), (1, 1), (2, 1)])),
            MultivariateMonomial::new((FE::new(4), vec![(0, 1), (1, 1), (2, 1)])),
        ]);
        let result = poly.evaluate(&[FE::one(), FE::new(2), FE::new(3)]);
        assert_eq!(result, FE::new(42));
    }

    #[test]
    fn test_partial_vars_evaluation() {
        // 3ab + 4bc
        // evaluate: a = 1, b = 2, c = 3
        // expected result = 30
        // a = 1, b = 2, c = 3
        let poly = MultivariatePolynomial::new(&[
            MultivariateMonomial::new((FE::new(3), vec![(0, 1), (1, 1)])),
            MultivariateMonomial::new((FE::new(4), vec![(1, 1), (2, 1)])),
        ]);
        let result = poly.evaluate(&[FE::one(), FE::new(2), FE::new(3)]);
        assert_eq!(result, FE::new(30));
    }
}
