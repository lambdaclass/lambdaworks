use crate::field::element::FieldElement;
use crate::field::traits::{IsField, IsPrimeField};
use crate::polynomial::multilinear_term::MultiLinearMonomial;
use crate::polynomial::term::Term;

/// Represents a multilinear polynomials as a collection of multilinear monomials
#[derive(Debug, PartialEq)]
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

#[cfg(test)]
mod tests {
    use crate::field::element::FieldElement;
    use crate::field::fields::u64_prime_field::U64PrimeField;
    use crate::polynomial::multilinear_poly::MultilinearPolynomial;
    use crate::polynomial::multilinear_term::MultiLinearMonomial;

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
                ]
            }
        );
    }
}
