use crate::field::element::FieldElement;
use crate::field::traits::{IsField, IsPrimeField};
use crate::polynomial::multilinear_term::MultiLinearMonomial;
use core::ops::AddAssign;
use std::fmt::Display;

/// Represents a multilinear polynomials as a collection of multilinear monomials
// TODO: add checks to track the max degree and number of variables.
#[derive(Debug, PartialEq, Clone)]
pub struct SparseMultilinearPolynomial<F: IsPrimeField>
where
    <F as IsField>::BaseType: Send + Sync,
{
    pub terms: Vec<MultiLinearMonomial<F>>,
    pub n_vars: usize, // number of variables
}

impl<F: IsPrimeField> Display for SparseMultilinearPolynomial<F>
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

impl<F: IsPrimeField> SparseMultilinearPolynomial<F>
where
    <F as IsField>::BaseType: Send + Sync,
{
    /// Build a new multilinear polynomial, from collection of multilinear monomials
    #[allow(dead_code)]
    pub fn new(terms: Vec<MultiLinearMonomial<F>>) -> Self {
        if !terms.is_empty() {
            let n = terms.iter().fold(
                0,
                |acc, m| if m.max_var() > acc { m.max_var() } else { acc },
            );
            Self {
                terms,
                n_vars: if n == 0 { 0 } else { n + 1 },
            } //we add +1 because variables indices start from 0
              // if terms is empty, we create an empty polynomial with 0 variables
        } else {
            Self {
                terms: Vec::<MultiLinearMonomial<F>>::new(),
                n_vars: 0,
            }
        }
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
                acc += {
                    // Check that p contains the proper amount of elements in dense form.
                    //assert!(self.max_var() < p.len());

                    if term.vars.is_empty() || p.is_empty() {
                        return term.coeff.clone();
                    }

                    // var_id is index of p
                    let eval = term
                        .vars
                        .iter()
                        .fold(FieldElement::<F>::one(), |acc, x| acc * p[*x].clone());
                    eval * &term.coeff
                };
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
            .map(|term| {
                let mut new_coefficient = term.coeff.clone();
                let mut unassigned_variables = term.vars.to_vec();

                for (var_id, assignment) in assignments {
                    if unassigned_variables.contains(var_id) {
                        new_coefficient = new_coefficient * assignment;
                        unassigned_variables.retain(|&id| id != *var_id);
                    }
                }

                MultiLinearMonomial::new((new_coefficient, unassigned_variables))
            })
            .collect();
        Self::new(updated_monomials)
    }

    /// Adds a polynomial
    /// This functions concatenates both vectors of terms
    pub fn add(&mut self, poly: SparseMultilinearPolynomial<F>) {
        for term in poly.terms.iter() {
            self.add_monomial(term);
        }
        self.update_nvars();
    }

    /// Updates the value of n_vars
    fn update_nvars(&mut self) {
        let n = self.terms.iter().fold(
            0,
            |acc, m| if m.max_var() > acc { m.max_var() } else { acc },
        );
        self.n_vars = if n == 0 { 0 } else { n + 1 };
    }

    /// Addition of monomial
    fn add_monomial(&mut self, mono: &MultiLinearMonomial<F>) {
        let mut added = false; // flag to check if the monomial was added or not

        for term in self.terms.iter_mut() {
            if term.vars == mono.vars {
                term.coeff.add_assign(mono.coeff.clone());
                added = true;
            }
        }

        if !added {
            self.terms.push(mono.clone());
        }
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
    fn test_add() {
        // polynomial 3t_1t_2 + 4t_2t_3
        let mut poly1 = SparseMultilinearPolynomial::new(vec![
            MultiLinearMonomial::new((FE::new(3), vec![1, 2])),
            MultiLinearMonomial::new((FE::new(4), vec![2, 3])),
        ]);

        // polynomial 2t_1t_2 - 4t_2t_3
        let poly2 = SparseMultilinearPolynomial::new(vec![
            MultiLinearMonomial::new((FE::new(2), vec![1, 2])),
            MultiLinearMonomial::new((-FE::new(4), vec![2, 3])),
        ]);

        // polynomial 5t_1t_2 + 6t_2t_3
        let expected = SparseMultilinearPolynomial::new(vec![
            MultiLinearMonomial::new((FE::new(5), vec![1, 2])),
            MultiLinearMonomial::new((FE::new(0), vec![2, 3])),
        ]);

        poly1.add(poly2);
        assert_eq!(poly1, expected);

        // polynomial 2t_1t_2 - 4t_2t_3
        let poly2 = SparseMultilinearPolynomial::new(vec![
            MultiLinearMonomial::new((FE::new(2), vec![1, 2])),
            MultiLinearMonomial::new((-FE::new(4), vec![2, 3])),
        ]);
        let mut poly_empty = SparseMultilinearPolynomial::<F>::new(vec![]);
        poly_empty.add(poly2);
        // polynomial 2t_1t_2 - 4t_2t_3
        let poly2 = SparseMultilinearPolynomial::new(vec![
            MultiLinearMonomial::new((FE::new(2), vec![1, 2])),
            MultiLinearMonomial::new((-FE::new(4), vec![2, 3])),
        ]);
        assert_eq!(poly_empty, poly2);
    }

    #[test]
    fn test_add_monomial() {
        // polynomial 3t_1t_2 + 4t_2t_3
        let mut poly = SparseMultilinearPolynomial::new(vec![
            MultiLinearMonomial::new((FE::new(3), vec![1, 2])),
            MultiLinearMonomial::new((FE::new(4), vec![2, 3])),
        ]);

        // monomial 3t_1t_2
        let mono = MultiLinearMonomial::new((FE::new(3), vec![1, 2]));

        // expected result 6t_1t_2 + 4t_2t_3
        let expected = SparseMultilinearPolynomial::new(vec![
            MultiLinearMonomial::new((FE::new(6), vec![1, 2])),
            MultiLinearMonomial::new((FE::new(4), vec![2, 3])),
        ]);

        poly.add_monomial(&mono);
        assert_eq!(poly, expected);
    }

    #[test]
    fn test_partial_evaluation() {
        // 3ab + 4bc
        // partially evaluate b = 2
        // expected result = 6a + 8c
        // a = 0, b = 1, c = 2
        let poly = SparseMultilinearPolynomial::new(vec![
            MultiLinearMonomial::new((FE::new(3), vec![0, 1])),
            MultiLinearMonomial::new((FE::new(4), vec![1, 2])),
        ]);
        assert_eq!(poly.n_vars, 3);
        let result = poly.partial_evaluate(&[(1, FE::new(2))]);
        assert_eq!(
            result,
            SparseMultilinearPolynomial {
                terms: vec![
                    MultiLinearMonomial {
                        coeff: FE::new(6),
                        vars: vec![0]
                    },
                    MultiLinearMonomial {
                        coeff: FE::new(8),
                        vars: vec![2]
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
        // a = 0, b = 1, c = 2
        let poly = SparseMultilinearPolynomial::new(vec![
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
        // a = 0, b = 1, c = 2
        let poly = SparseMultilinearPolynomial::new(vec![
            MultiLinearMonomial::new((FE::new(3), vec![0, 1])),
            MultiLinearMonomial::new((FE::new(4), vec![1, 2])),
        ]);
        let result = poly.evaluate(&[FE::one(), FE::new(2), FE::new(3)]);
        assert_eq!(result, FE::new(30));
    }
}
