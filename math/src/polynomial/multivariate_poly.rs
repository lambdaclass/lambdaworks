use crate::{
    errors::TermError,
    field::{element::FieldElement, traits::IsField},
    polynomial::{multivariate_term::MultivariateMonomial, term::Term},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MultivariatePolynomial<F: IsField + Default>
where
    <F as IsField>::BaseType: Send + Sync,
{
    pub terms: Vec<MultivariateMonomial<F>>,
    pub num_vars: usize,
}

impl<F: IsField + Default> MultivariatePolynomial<F>
where
    <F as IsField>::BaseType: Send + Sync,
{
    pub fn new(terms: &[MultivariateMonomial<F>]) -> Self {
        let n = terms.iter().fold(
            0,
            |acc, m| if m.max_var() > acc { m.max_var() } else { acc },
        );
        Self {
            terms: terms.to_vec(),
            num_vars: n + 1,
        }
    }

    pub const fn zero() -> Self {
        Self {
            terms: Vec::new(),
            num_vars: 0,
        }
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
    pub fn evaluate(&self, p: &[FieldElement<F>]) -> Result<FieldElement<F>, TermError> {
        if p.len() != self.num_vars {
            return Err(TermError::InvalidPartialEvaluationPoint);
        }

        let res = self
            .terms
            .iter()
            .fold(FieldElement::<F>::zero(), |mut acc, term| {
                acc += term.evaluate(p);
                acc
            });
        Ok(res)
    }

    pub fn partial_evaluate(
        &self,
        assignments: &[(usize, FieldElement<F>)],
    ) -> Result<Self, TermError> {
        if assignments.len() > self.num_vars {
            return Err(TermError::InvalidPartialEvaluationPoint);
        }
        let updated_monomials: Vec<MultivariateMonomial<F>> = self
            .terms
            .iter()
            .map(|term| term.partial_evaluate(assignments))
            .collect();
        Ok(Self::new(&updated_monomials))
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
            result.unwrap(),
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
                ],
                num_vars: 4
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
        assert_eq!(result.unwrap(), FE::new(42));
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
        assert_eq!(result.unwrap(), FE::new(30));
    }

    #[test]
    fn test_evaluate_incorrect_vars_len() {
        // 3ab + 4bc
        // evaluate: a = 1, b = 2, c = 3
        // expected result = 30
        // a = 1, b = 2, c = 3
        let poly = MultivariatePolynomial::new(&[
            MultivariateMonomial::new((FE::new(3), vec![(0, 1), (1, 1)])),
            MultivariateMonomial::new((FE::new(4), vec![(1, 1), (2, 1)])),
        ]);
        assert!(poly.evaluate(&[FE::one(), FE::new(2)]).is_err());
    }

    #[test]
    fn test_partial_evaluation_incorrect_var_len() {
        // 3ab + 4bc
        // partially evaluate b = 2
        // expected result = 6a + 8c
        // a = 1, b = 2, c = 3
        let poly = MultivariatePolynomial::new(&[
            MultivariateMonomial::new((FE::new(3), vec![(1, 1), (2, 1)])),
            MultivariateMonomial::new((FE::new(4), vec![(2, 1), (3, 1)])),
        ]);
        assert!(poly
            .partial_evaluate(&[
                (2, FE::new(2)),
                (1, FE::new(2)),
                (3, FE::new(2)),
                (4, FE::new(2)),
                (5, FE::new(2)),
            ])
            .is_err());
    }
}
