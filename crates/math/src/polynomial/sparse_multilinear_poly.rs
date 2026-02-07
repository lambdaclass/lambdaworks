#[cfg(feature = "parallel")]
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::field::{element::FieldElement, traits::IsField};
use crate::polynomial::error::MultilinearError;
use alloc::vec::Vec;

pub struct SparseMultilinearPolynomial<F: IsField>
where
    <F as IsField>::BaseType: Send + Sync,
{
    num_vars: usize,
    evals: Vec<(usize, FieldElement<F>)>,
}

impl<F: IsField> SparseMultilinearPolynomial<F>
where
    <F as IsField>::BaseType: Send + Sync,
{
    /// Creates a new sparse multilinear polynomial
    pub fn new(num_vars: usize, evals: Vec<(usize, FieldElement<F>)>) -> Self {
        SparseMultilinearPolynomial { num_vars, evals }
    }

    /// Returns the number of variables of the sparse multilinear polynomial
    pub fn num_vars(&self) -> usize {
        self.num_vars
    }

    // Takes O(n log n)
    /// Evaluates the multilinear polynomial at a point r
    pub fn evaluate(&self, r: &[FieldElement<F>]) -> Result<FieldElement<F>, MultilinearError> {
        if r.len() != self.num_vars() {
            return Err(MultilinearError::IncorrectNumberOfEvaluationPoints(
                r.len(),
                self.num_vars(),
            ));
        }

        let num_bits = r.len();

        #[cfg(feature = "parallel")]
        let iter = (0..self.evals.len()).into_par_iter();

        #[cfg(not(feature = "parallel"))]
        let iter = 0..self.evals.len();

        Ok(iter
            .map(|i| {
                let (idx, ref coeff) = self.evals[i];
                let mut chi_i = FieldElement::<F>::one();
                for (j, r_j) in r.iter().enumerate() {
                    let bit = (idx >> (num_bits - 1 - j)) & 1 == 1;
                    if bit {
                        chi_i *= r_j;
                    } else {
                        chi_i *= FieldElement::<F>::one() - r_j;
                    }
                }
                chi_i * coeff
            })
            .sum())
    }

    // Takes O(n log n)
    /// Evaluates the multilinear polynomial at a point r
    pub fn evaluate_with(
        num_vars: usize,
        evals: &[(usize, FieldElement<F>)],
        r: &[FieldElement<F>],
    ) -> Result<FieldElement<F>, MultilinearError> {
        if r.len() != num_vars {
            return Err(MultilinearError::IncorrectNumberOfEvaluationPoints(
                r.len(),
                num_vars,
            ));
        }

        let num_bits = r.len();

        #[cfg(feature = "parallel")]
        let iter = (0..evals.len()).into_par_iter();

        #[cfg(not(feature = "parallel"))]
        let iter = 0..evals.len();
        Ok(iter
            .map(|i| {
                let (idx, ref coeff) = evals[i];
                let mut chi_i = FieldElement::<F>::one();
                for (j, r_j) in r.iter().enumerate() {
                    let bit = (idx >> (num_bits - 1 - j)) & 1 == 1;
                    if bit {
                        chi_i *= r_j;
                    } else {
                        chi_i *= FieldElement::<F>::one() - r_j;
                    }
                }
                chi_i * coeff
            })
            .sum())
    }
}

#[cfg(test)]
mod test {

    #[test]
    fn evaluate() {
        use crate::field::fields::u64_prime_field::U64PrimeField;
        use alloc::vec;

        use super::*;

        const ORDER: u64 = 101;
        type F = U64PrimeField<ORDER>;
        type FE = FieldElement<F>;

        // Let the polynomial have 3 variables, p(x_1, x_2, x_3) = (x_1 + x_2) * x_3
        // Evaluations of the polynomial at boolean cube are [0, 0, 0, 1, 0, 1, 0, 2].

        let two = FE::from(2);
        let z = vec![(3, FE::one()), (5, FE::one()), (7, two)];
        let m_poly = SparseMultilinearPolynomial::<F>::new(3, z.clone());

        let x = vec![FE::one(), FE::one(), FE::one()];
        assert_eq!(m_poly.evaluate(x.as_slice()).unwrap(), two);
        assert_eq!(
            SparseMultilinearPolynomial::evaluate_with(3, &z, x.as_slice()).unwrap(),
            two
        );

        let x = vec![FE::one(), FE::zero(), FE::one()];
        assert_eq!(m_poly.evaluate(x.as_slice()).unwrap(), FE::one());
        assert_eq!(
            SparseMultilinearPolynomial::evaluate_with(3, &z, x.as_slice()).unwrap(),
            FE::one()
        );
    }
}
