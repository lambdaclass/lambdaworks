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
    pub fn new(num_vars: usize, evals: Vec<(usize, FieldElement<F>)>) -> Self {
        SparseMultilinearPolynomial { num_vars, evals }
    }

    pub fn num_vars(&self) -> usize {
        self.num_vars
    }

    /// Computes the eq extension polynomial of the polynomial.
    /// return 1 when a == r, otherwise return 0.
    fn compute_chi(a: &[bool], r: &[FieldElement<F>]) -> Result<FieldElement<F>, MultilinearError> {
        assert_eq!(a.len(), r.len());
        if a.len() != r.len() {
            return Err(MultilinearError::ChisAndEvalsLengthMismatch(
                a.len(),
                r.len(),
            ));
        }
        let mut chi_i = FieldElement::one();
        for j in 0..r.len() {
            if a[j] {
                chi_i *= &r[j];
            } else {
                chi_i *= FieldElement::<F>::one() - &r[j];
            }
        }
        Ok(chi_i)
    }

    // Takes O(n log n)
    pub fn evaluate(&self, r: &[FieldElement<F>]) -> Result<FieldElement<F>, MultilinearError> {
        if r.len() != self.num_vars() {
            return Err(MultilinearError::IncorrectNumberofEvaluationPoints(
                r.len(),
                self.num_vars(),
            ));
        }

        #[cfg(feature = "parallel")]
        let iter = (0..self.evals.len()).into_par_iter();

        #[cfg(not(feature = "parallel"))]
        let iter = 0..self.evals.len();

        Ok(iter
            .map(|i| {
                let bits = get_bits(self.evals[i].0, r.len());
                let mut chi_i = FieldElement::<F>::one();
                for j in 0..r.len() {
                    if bits[j] {
                        chi_i *= &r[j];
                    } else {
                        chi_i *= FieldElement::<F>::one() - &r[j];
                    }
                }
                chi_i * &self.evals[i].1
            })
            .sum())
    }

    // Takes O(n log n)
    pub fn evaluate_with(
        num_vars: usize,
        evals: &[(usize, FieldElement<F>)],
        r: &[FieldElement<F>],
    ) -> Result<FieldElement<F>, MultilinearError> {
        assert_eq!(num_vars, r.len());
        if r.len() != num_vars {
            return Err(MultilinearError::IncorrectNumberofEvaluationPoints(
                r.len(),
                num_vars,
            ));
        }

        #[cfg(feature = "parallel")]
        let iter = (0..evals.len()).into_par_iter();

        #[cfg(not(feature = "parallel"))]
        let iter = 0..evals.len();
        Ok(iter
            .map(|i| {
                let bits = get_bits(evals[i].0, r.len());
                SparseMultilinearPolynomial::compute_chi(&bits, r).unwrap() * &evals[i].1
            })
            .sum())
    }
}

/// Returns the bit decomposition (Vec<bool>) of the `index` of an evaluation within the sparse multilinear polynomial.
fn get_bits(n: usize, num_bits: usize) -> Vec<bool> {
    (0..num_bits)
        .map(|shift_amount| ((n & (1 << (num_bits - shift_amount - 1))) > 0))
        .collect::<Vec<bool>>()
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
