use crate::field::element::FieldElement;
use crate::field::traits::{IsField, IsPrimeField};
use crate::polynomial::multilinear_term::MultiLinearMonomial;
use crate::polynomial::term::Term;
use std::fmt::Display;

/// Represents a multilinear polynomials as a collection of multilinear monomials
// TODO: add checks to track the max degree and number of variables.
#[derive(Debug, PartialEq)]
pub struct MultilinearPolynomial<F: IsPrimeField>
where
    <F as IsField>::BaseType: Send + Sync,
{
    pub terms: Vec<MultiLinearMonomial<F>>,
    pub n_vars: usize, // number of variables
}

impl<F: IsPrimeField> Display for MultilinearPolynomial<F>
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

impl<F: IsPrimeField> MultilinearPolynomial<F>
where
    <F as IsField>::BaseType: Send + Sync,
{
    /// Build a new multilinear polynomial, from collection of multilinear monomials
    #[allow(dead_code)]
    pub fn new(terms: Vec<MultiLinearMonomial<F>>) -> Self {
        let n = terms.iter().fold(
            0,
            |acc, m| if m.max_var() > acc { m.max_var() } else { acc },
        );
        Self {
            terms,
            n_vars: if n == 0 { 0 } else { n + 1 },
        } //we add +1 because variables indices start from 0
    }

    /// Evaluates `self` at the point `p`.
    /// Note: assumes p contains points for all variables aka is not sparse.
    #[allow(dead_code)]
    pub fn evaluate(&self, p: &[FieldElement<F>]) -> FieldElement<F> {
        // check the number of evaluations points is equal to the number of variables
        // var_id is index of p
        self.terms
            .iter()
            .fold(FieldElement::<F>::zero(), |mut acc, term| {
                acc += term.evaluate(p);
                acc
            })
    }

    /// Selectively assign values to variables in the polynomial, returns a reduced
    /// polynomial after assignment evaluation
    // TODO: can we change this to modify in place to remove the extract allocation
    #[allow(dead_code)]
    pub fn partial_evaluate(&self, assignments: &[(usize, FieldElement<F>)]) -> Self {
        let updated_monomials: Vec<MultiLinearMonomial<F>> = self
            .terms
            .iter()
            .map(|term| term.partial_evaluate(assignments))
            .collect();
        Self::new(updated_monomials)
    }

    /// Adds a polynomial
    /// This functions concatenates both vectors of terms
    pub fn add(&mut self, poly: MultilinearPolynomial<F>) {
        self.terms.extend(poly.terms);
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
        let poly = MultilinearPolynomial::new(vec![
            MultiLinearMonomial::new((FE::new(3), vec![1, 2])),
            MultiLinearMonomial::new((FE::new(4), vec![2, 3])),
        ]);
        let result = poly.partial_evaluate(&[(2, FE::new(2))]);
        assert_eq!(
            result,
            MultilinearPolynomial {
                terms: vec![
                    MultiLinearMonomial {
                        coeff: FE::new(6),
                        vars: vec![1]
                    },
                    MultiLinearMonomial {
                        coeff: FE::new(8),
                        vars: vec![3]
                    }
                ],
                n_vars: 3,
            }
        );
    }

    #[test]
    fn test_all_vars_evaluation() {
        // 3abc + 4abc
        // evaluate: a = 1, b = 2, c = 3
        // expected result = 42
        // a = 1, b = 2, c = 3
        let poly = MultilinearPolynomial::new(vec![
            MultiLinearMonomial::new((FE::new(3), vec![0, 1, 2])),
            MultiLinearMonomial::new((FE::new(4), vec![0, 1, 2])),
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
        let poly = MultilinearPolynomial::new(vec![
            MultiLinearMonomial::new((FE::new(3), vec![0, 1])),
            MultiLinearMonomial::new((FE::new(4), vec![1, 2])),
        ]);
        let result = poly.evaluate(&[FE::one(), FE::new(2), FE::new(3)]);
        assert_eq!(result, FE::new(30));
    }
}
