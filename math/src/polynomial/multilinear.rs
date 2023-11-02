use core::fmt::Display;

use crate::{
    errors::TermError,
    field::{
        element::FieldElement,
        traits::{IsField, IsPrimeField},
    },
    polynomial::{terms::multilinear::MultilinearMonomial, traits::term::Term},
};

/// Represents a multilinear polynomials as a collection of multilinear monomials
// TODO: add checks to track the max degree and number of variables.
#[derive(Debug, PartialEq)]
pub struct MultilinearPolynomial<F: IsField + IsPrimeField>
where
    <F as IsField>::BaseType: Send + Sync,
{
    pub terms: Vec<MultilinearMonomial<F>>,
    pub n_vars: usize, // number of variables
}

impl<F: IsField + IsPrimeField> MultilinearPolynomial<F>
where
    <F as IsField>::BaseType: Send + Sync,
{
    /// Build a new multilinear polynomial, from collection of multilinear monomials
    pub fn new(terms: &[MultilinearMonomial<F>]) -> Self {
        let n = terms.iter().fold(
            0,
            |acc, m| if m.max_var() > acc { m.max_var() } else { acc },
        );
        Self {
            terms: terms.to_vec(),
            n_vars: n + 1,
        } //we add +1 because variables indices start from 0
    }

    /// Evaluates `self` at the point `p`.
    /// Note: assumes p contains points for all variables aka is not sparse.
    pub fn evaluate(&self, p: &[FieldElement<F>]) -> Result<FieldElement<F>, TermError> {
        // check the number of evaluations points is equal to the number of variables
        // var_id is index of p
        if self.n_vars != p.len() {
            return Err(TermError::InvalidEvaluationPoint);
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

    /// Selectively assign values to variables in the polynomial, returns a reduced
    /// polynomial after assignment evaluation
    // TODO: can we change this to modify in place to remove the extract allocation
    pub fn partial_evaluate(
        &self,
        assignments: &[(usize, FieldElement<F>)],
    ) -> Result<Self, TermError> {
        if assignments.len() > self.n_vars {
            return Err(TermError::InvalidPartialEvaluationPoint);
        }

        let updated_monomials: Vec<MultilinearMonomial<F>> = self
            .terms
            .iter()
            .map(|term| term.partial_evaluate(assignments))
            .collect();
        Ok(Self::new(&updated_monomials))
    }
}

impl<F: IsField + IsPrimeField> Display for MultilinearPolynomial<F>
where
    <F as IsField>::BaseType: Send + Sync,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut output: String = String::new();
        let monomials = self.terms.clone();

        for elem in &monomials[0..monomials.len() - 1] {
            output.push_str(&elem.to_string()[0..]);
            output.push_str(" + ");
        }
        output.push_str(&monomials[monomials.len() - 1].to_string());
        write!(f, "{}", output)
    }
}

#[cfg(test)]
mod tests {
    use crate::field::fields::u64_prime_field::U64PrimeField;

    use super::*;

    const ORDER: u64 = 101;
    type F = U64PrimeField<ORDER>;
    type FE = FieldElement<F>;

    #[test]
    fn test_partial_evaluation() {
        // 3ab + 4bc
        // partially evaluate b = 2
        // expected result = 6a + 8c
        // a = 1, b = 2, c = 3
        let poly = MultilinearPolynomial::new(&[
            MultilinearMonomial::new((FE::new(3), vec![1, 2])),
            MultilinearMonomial::new((FE::new(4), vec![2, 3])),
        ]);
        let result = poly.partial_evaluate(&[(2, FE::new(2))]);
        assert_eq!(
            result.unwrap(),
            MultilinearPolynomial {
                terms: vec![
                    MultilinearMonomial {
                        coeff: FE::new(6),
                        vars: vec![1]
                    },
                    MultilinearMonomial {
                        coeff: FE::new(8),
                        vars: vec![3]
                    }
                ],
                n_vars: 4,
            }
        );
    }

    #[test]
    fn test_all_vars_evaluation() {
        // 3abc + 4abc
        // evaluate: a = 1, b = 2, c = 3
        // expected result = 42
        // a = 1, b = 2, c = 3
        let poly = MultilinearPolynomial::new(&[
            MultilinearMonomial::new((FE::new(3), vec![0, 1, 2])),
            MultilinearMonomial::new((FE::new(4), vec![0, 1, 2])),
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
        let poly = MultilinearPolynomial::new(&[
            MultilinearMonomial::new((FE::new(3), vec![0, 1])),
            MultilinearMonomial::new((FE::new(4), vec![1, 2])),
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
        let poly = MultilinearPolynomial::new(&[
            MultilinearMonomial::new((FE::new(3), vec![0, 1])),
            MultilinearMonomial::new((FE::new(4), vec![1, 2])),
        ]);
        assert!(poly.evaluate(&[FE::one(), FE::new(2)]).is_err());
    }

    #[test]
    fn test_partial_evaluation_incorrect_var_len() {
        // 3ab + 4bc
        // partially evaluate b = 2
        // expected result = 6a + 8c
        // a = 1, b = 2, c = 3
        let poly = MultilinearPolynomial::new(&[
            MultilinearMonomial::new((FE::new(3), vec![1, 2])),
            MultilinearMonomial::new((FE::new(4), vec![2, 3])),
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
