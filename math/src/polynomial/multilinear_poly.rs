use crate::field::element::FieldElement;
use crate::field::traits::IsField;
use crate::polynomial::multilinear_term::MultiLinearMonomial;
use crate::polynomial::term::Term;

/// Represents a multilinear polynomials as a collection of multilinear monomials
// TODO: add checks to track the max degree and number of variables.
#[derive(Debug, PartialEq)]
pub struct MultilinearPolynomial<F: IsField>
where
    <F as IsField>::BaseType: Send + Sync,
{
    pub terms: Vec<MultiLinearMonomial<F>>,
    pub n_vars: usize, // number of variables
}

impl<F: IsField> MultilinearPolynomial<F>
where
    <F as IsField>::BaseType: Send + Sync,
{
    /// Build a new multilinear polynomial, from collection of multilinear monomials
    pub fn new(terms: &[MultiLinearMonomial<F>]) -> Self {
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
    pub fn evaluate(&self, p: &[FieldElement<F>]) -> FieldElement<F> {
        // check the number of evaluations points is equal to the number of variables
        // var_id is index of p
        assert_eq!(self.n_vars, p.len(), "Incorrect number of vars");
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
    pub fn partial_evaluate(&self, assignments: &[(usize, FieldElement<F>)]) -> Self {
        assert!(
            assignments.len() < self.n_vars,
            "More assignments than variables present in input"
        );
        let updated_monomials: Vec<MultiLinearMonomial<F>> = self
            .terms
            .iter()
            .map(|term| term.partial_evaluate(assignments))
            .collect();
        Self::new(&updated_monomials)
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
        let poly = MultilinearPolynomial::new(&[
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
        let poly = MultilinearPolynomial::new(&[
            MultiLinearMonomial::new((FE::new(3), vec![0, 1])),
            MultiLinearMonomial::new((FE::new(4), vec![1, 2])),
        ]);
        let result = poly.evaluate(&[FE::one(), FE::new(2), FE::new(3)]);
        assert_eq!(result, FE::new(30));
    }

    #[test]
    #[should_panic(expected = "Incorrect number of vars")]
    fn test_evaluate_incorrect_vars_len() {
        // 3ab + 4bc
        // evaluate: a = 1, b = 2, c = 3
        // expected result = 30
        // a = 1, b = 2, c = 3
        let poly = MultilinearPolynomial::new(&[
            MultiLinearMonomial::new((FE::new(3), vec![0, 1])),
            MultiLinearMonomial::new((FE::new(4), vec![1, 2])),
        ]);
        let _ = poly.evaluate(&[FE::one(), FE::new(2)]);
    }

    #[test]
    #[should_panic(expected = "More assignments than variables present in input")]
    fn test_partial_evaluation_incorrect_var_len() {
        // 3ab + 4bc
        // partially evaluate b = 2
        // expected result = 6a + 8c
        // a = 1, b = 2, c = 3
        let poly = MultilinearPolynomial::new(&[
            MultiLinearMonomial::new((FE::new(3), vec![1, 2])),
            MultiLinearMonomial::new((FE::new(4), vec![2, 3])),
        ]);
        let _ = poly.partial_evaluate(&[
            (2, FE::new(2)),
            (1, FE::new(2)),
            (3, FE::new(2)),
            (4, FE::new(2)),
        ]);
    }
}
