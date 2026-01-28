use crate::{
    field::{element::FieldElement, traits::IsField},
    polynomial::{error::MultilinearError, Polynomial},
};
use alloc::{vec, vec::Vec};
use core::ops::{Add, Index, Mul};
#[cfg(feature = "parallel")]
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

/// Represents a multilinear polynomial as a vector of evaluations (FieldElements) in Lagrange basis.
#[derive(Debug, PartialEq, Clone)]
pub struct DenseMultilinearPolynomial<F: IsField>
where
    <F as IsField>::BaseType: Send + Sync,
{
    evals: Vec<FieldElement<F>>,
    n_vars: usize,
    len: usize,
}

impl<F: IsField> DenseMultilinearPolynomial<F>
where
    <F as IsField>::BaseType: Send + Sync,
{
    /// Constructs a new multilinear polynomial from a collection of evaluations.
    /// Pads non-power-of-2 evaluations with zeros.
    pub fn new(mut evals: Vec<FieldElement<F>>) -> Self {
        while !evals.len().is_power_of_two() {
            evals.push(FieldElement::zero());
        }
        let len = evals.len();
        DenseMultilinearPolynomial {
            n_vars: log_2(len),
            evals,
            len,
        }
    }

    /// Returns the number of variables.
    pub fn num_vars(&self) -> usize {
        self.n_vars
    }

    /// Returns a reference to the evaluations vector.
    pub fn evals(&self) -> &Vec<FieldElement<F>> {
        &self.evals
    }

    /// Returns the total number of evaluations (2^num_vars).
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Evaluates `self` at the point `r` (a vector of FieldElements) in O(n) time.
    /// `r` must have a value for each variable.
    pub fn evaluate(&self, r: Vec<FieldElement<F>>) -> Result<FieldElement<F>, MultilinearError> {
        if r.len() != self.num_vars() {
            return Err(MultilinearError::IncorrectNumberofEvaluationPoints(
                r.len(),
                self.num_vars(),
            ));
        }
        let mut chis: Vec<FieldElement<F>> =
            vec![FieldElement::one(); (2usize).pow(r.len() as u32)];
        let mut size = 1;
        for j in r {
            size *= 2;
            for i in (0..size).rev().step_by(2) {
                let half_i = i / 2;
                let temp = &chis[half_i] * &j;
                chis[i] = temp;
                chis[i - 1] = &chis[half_i] - &chis[i];
            }
        }
        #[cfg(feature = "parallel")]
        let iter = (0..chis.len()).into_par_iter();
        #[cfg(not(feature = "parallel"))]
        let iter = 0..chis.len();
        Ok(iter.map(|i| &self.evals[i] * &chis[i]).sum())
    }

    /// Evaluates a slice of evaluations with the given point `r`.
    pub fn evaluate_with(
        evals: &[FieldElement<F>],
        r: &[FieldElement<F>],
    ) -> Result<FieldElement<F>, MultilinearError> {
        let mut chis: Vec<FieldElement<F>> =
            vec![FieldElement::one(); (2usize).pow(r.len() as u32)];
        if chis.len() != evals.len() {
            return Err(MultilinearError::ChisAndEvalsLengthMismatch(
                chis.len(),
                evals.len(),
            ));
        }
        let mut size = 1;
        for j in r {
            size *= 2;
            for i in (0..size).rev().step_by(2) {
                let half_i = i / 2;
                let temp = &chis[half_i] * j;
                chis[i] = temp;
                chis[i - 1] = &chis[half_i] - &chis[i];
            }
        }
        Ok((0..evals.len()).map(|i| &evals[i] * &chis[i]).sum())
    }

    /// Fixes the first variable to the given value `r` and returns a new DenseMultilinearPolynomial
    /// with one fewer variable.
    ///
    /// Combines each pair of evaluations as: new_eval = a + r * (b - a)
    ///  This reduces the polynomial by one variable, allowing it to later be collapsed
    /// into a univariate polynomial by summing over the remaining variables.
    ///
    /// Example (2 variables): evaluations are ordered as:
    ///     [f(0,0), f(0,1), f(1,0), f(1,1)]
    /// Fixing the first variable `x = r` produces evaluations of a 1-variable polynomial:
    ///     [f(r,0), f(r,1)]
    /// computed explicitly as:
    ///     f(r,0) = f(0,0) + r * ( f(1,0) - f(0,0)),
    ///     f(r,1) = f(0,1) + r * (f(1,1) - f(0,1))
    pub fn fix_first_variable(&self, r: &FieldElement<F>) -> DenseMultilinearPolynomial<F> {
        let n = self.num_vars();
        assert!(n > 0, "Cannot fix variable in a 0-variable polynomial");
        let half = 1 << (n - 1);
        let new_evals: Vec<FieldElement<F>> = (0..half)
            .map(|j| {
                let a = &self.evals[j];
                let b = &self.evals[j + half];
                a + r * (b - a)
            })
            .collect();
        DenseMultilinearPolynomial::from((n - 1, new_evals))
    }

    /// Returns the evaluations of the polynomial on the Boolean hypercube \(\{0,1\}^n\).
    /// Since we are in Lagrange basis, this is just the elements stored in self.evals.
    pub fn to_evaluations(&self) -> Vec<FieldElement<F>> {
        self.evals.clone()
    }

    /// Collapses the last variable by fixing it to 0 and 1,
    /// sums the evaluations, and returns a univariate polynomial (as a Polynomial)
    /// of the form: sum0 + (sum1 - sum0) * x.
    pub fn to_univariate(&self) -> Polynomial<FieldElement<F>> {
        let poly0 = self.fix_first_variable(&FieldElement::zero());
        let poly1 = self.fix_first_variable(&FieldElement::one());
        let sum0: FieldElement<F> = poly0.to_evaluations().into_iter().sum();
        let sum1: FieldElement<F> = poly1.to_evaluations().into_iter().sum();
        let diff = sum1 - &sum0;
        Polynomial::new(&[sum0, diff])
    }

    /// Multiplies the polynomial by a scalar.
    pub fn scalar_mul(&self, scalar: &FieldElement<F>) -> Self {
        let mut new_poly = self.clone();
        new_poly.evals.iter_mut().for_each(|eval| *eval *= scalar);
        new_poly
    }

    /// Extends this DenseMultilinearPolynomial by concatenating another polynomial of the same length.
    pub fn extend(&mut self, other: &DenseMultilinearPolynomial<F>) {
        debug_assert_eq!(self.evals.len(), self.len);
        debug_assert_eq!(other.evals.len(), self.len);
        self.evals.extend(other.evals.iter().cloned());
        self.n_vars += 1;
        self.len *= 2;
        debug_assert_eq!(self.evals.len(), self.len);
    }

    /// Merges a series of DenseMultilinearPolynomials into one polynomial by concatenating
    /// their evaluation vectors in order.
    /// Zero-pads the final merged polynomial to the next power-of-two length if necessary.
    pub fn merge(polys: &[DenseMultilinearPolynomial<F>]) -> DenseMultilinearPolynomial<F> {
        // Calculate total size needed for pre-allocation
        let total_len: usize = polys.iter().map(|p| p.evals.len()).sum();
        let final_len = total_len.next_power_of_two();
        let mut z: Vec<FieldElement<F>> = Vec::with_capacity(final_len);
        for poly in polys {
            z.extend(poly.evals.iter().cloned());
        }
        z.resize(final_len, FieldElement::zero());
        DenseMultilinearPolynomial::new(z)
    }

    /// Constructs a DenseMultilinearPolynomial from a slice of u64 values.
    pub fn from_u64(evals: &[u64]) -> Self {
        DenseMultilinearPolynomial::new(evals.iter().map(|&i| FieldElement::from(i)).collect())
    }
}

impl<F: IsField> Index<usize> for DenseMultilinearPolynomial<F>
where
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = FieldElement<F>;

    #[inline(always)]
    fn index(&self, index: usize) -> &FieldElement<F> {
        &self.evals[index]
    }
}

/// Adds two DenseMultilinearPolynomials.
/// Assumes that both polynomials have the same number of variables.
impl<F: IsField> Add for DenseMultilinearPolynomial<F>
where
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = Result<Self, &'static str>;

    fn add(self, other: Self) -> Self::Output {
        if self.num_vars() != other.num_vars() {
            return Err("Polynomials must have the same number of variables");
        }
        #[cfg(feature = "parallel")]
        let evals = self.evals.into_par_iter().zip(other.evals.into_par_iter());
        #[cfg(not(feature = "parallel"))]
        let evals = self.evals.iter().zip(other.evals.iter());
        let sum: Vec<FieldElement<F>> = evals.map(|(a, b)| a + b).collect();
        Ok(DenseMultilinearPolynomial::new(sum))
    }
}

impl<F: IsField> Mul<FieldElement<F>> for DenseMultilinearPolynomial<F>
where
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = DenseMultilinearPolynomial<F>;

    fn mul(self, rhs: FieldElement<F>) -> Self::Output {
        Self::scalar_mul(&self, &rhs)
    }
}

impl<F: IsField> Mul<&FieldElement<F>> for DenseMultilinearPolynomial<F>
where
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = DenseMultilinearPolynomial<F>;

    fn mul(self, rhs: &FieldElement<F>) -> Self::Output {
        Self::scalar_mul(&self, rhs)
    }
}

/// Helper function to calculate log₂(n).
fn log_2(n: usize) -> usize {
    if n == 0 {
        return 0;
    }
    if n.is_power_of_two() {
        (1usize.leading_zeros() - n.leading_zeros()) as usize
    } else {
        (0usize.leading_zeros() - n.leading_zeros()) as usize
    }
}

impl<F: IsField> From<(usize, Vec<FieldElement<F>>)> for DenseMultilinearPolynomial<F>
where
    <F as IsField>::BaseType: Send + Sync,
{
    fn from((num_vars, evaluations): (usize, Vec<FieldElement<F>>)) -> Self {
        assert_eq!(
            evaluations.len(),
            1 << num_vars,
            "The size of evaluations should be 2^num_vars."
        );
        DenseMultilinearPolynomial {
            n_vars: num_vars,
            evals: evaluations,
            len: 1 << num_vars,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::fields::u64_prime_field::U64PrimeField;
    const ORDER: u64 = 101;
    type F = U64PrimeField<ORDER>;
    type FE = FieldElement<F>;

    pub fn evals(r: Vec<FE>) -> Vec<FE> {
        let mut evals: Vec<FE> = vec![FE::one(); (2usize).pow(r.len() as u32)];
        let mut size = 1;
        for j in r {
            size *= 2;
            for i in (0..size).rev().step_by(2) {
                let scalar = evals[i / 2];
                evals[i] = scalar * j;
                evals[i - 1] = scalar - evals[i];
            }
        }
        evals
    }

    pub fn compute_factored_evals(r: Vec<FE>) -> (Vec<FE>, Vec<FE>) {
        let size = r.len();
        let (left_num_vars, _right_num_vars) = (size / 2, size - size / 2);
        let l = evals(r[..left_num_vars].to_vec());
        let r = evals(r[left_num_vars..size].to_vec());
        (l, r)
    }

    fn evaluate_with_lr(z: &[FE], r: &[FE]) -> FE {
        let (l, r) = compute_factored_evals(r.to_vec());
        let size = r.len();
        // Ensure size is even.
        assert!(size % 2 == 0);
        // n = 2^size
        let n = (2usize).pow(size as u32);
        // Compute m = sqrt(n) = 2^(l/2)
        let m = (n as f64).sqrt() as usize;
        // Compute vector-matrix product between L and Z (viewed as a matrix)
        let lz = (0..m)
            .map(|i| {
                (0..m).fold(FE::zero(), |mut acc, j| {
                    acc += l[j] * z[j * m + i];
                    acc
                })
            })
            .collect::<Vec<FE>>();
        // Compute dot product between LZ and R
        (0..lz.len()).map(|i| lz[i] * r[i]).sum()
    }

    #[test]
    fn evaluation() {
        // Example: Z = [1, 2, 1, 4]
        let z = vec![FE::one(), FE::from(2u64), FE::one(), FE::from(4u64)];
        // r = [4, 3]
        let r = vec![FE::from(4u64), FE::from(3u64)];
        let eval_with_lr = evaluate_with_lr(&z, &r);
        let poly = DenseMultilinearPolynomial::new(z);
        let eval = poly.evaluate(r).unwrap();
        assert_eq!(eval, FE::from(28u64));
        assert_eq!(eval_with_lr, eval);
    }

    #[test]
    fn evaluate_with() {
        let two = FE::from(2);
        let z = vec![
            FE::zero(),
            FE::zero(),
            FE::zero(),
            FE::one(),
            FE::one(),
            FE::one(),
            FE::zero(),
            two,
        ];
        let x = vec![FE::one(), FE::one(), FE::one()];
        let y = DenseMultilinearPolynomial::<F>::evaluate_with(z.as_slice(), x.as_slice()).unwrap();
        assert_eq!(y, two);
    }

    #[test]
    fn add() {
        let a = DenseMultilinearPolynomial::new(vec![FE::from(3); 4]);
        let b = DenseMultilinearPolynomial::new(vec![FE::from(7); 4]);
        let c = a.add(b).unwrap();
        assert_eq!(*c.evals(), vec![FE::from(10); 4]);
    }

    #[test]
    fn mul() {
        let a = DenseMultilinearPolynomial::new(vec![FE::from(3); 4]);
        let b = a.mul(&FE::from(2));
        assert_eq!(*b.evals(), vec![FE::from(6); 4]);
    }

    // Take a multilinear polynomial of length 2^2 and merge with a polynomial of 2^1.
    // The resulting polynomial should be padded to length 2^3 = 8 and the last two evaluations should be FE::zero().
    #[test]
    fn merge() {
        let a = DenseMultilinearPolynomial::new(vec![FE::from(3); 4]);
        let b = DenseMultilinearPolynomial::new(vec![FE::from(2); 2]);
        let c = DenseMultilinearPolynomial::merge(&[a, b]);
        assert_eq!(c.len(), 8);
        assert_eq!(c[c.len() - 1], FE::zero());
        assert_eq!(c[c.len() - 2], FE::zero());
        let d = DenseMultilinearPolynomial::new(vec![
            FE::from(3),
            FE::from(3),
            FE::from(3),
            FE::from(3),
            FE::from(2),
            FE::from(2),
            FE::zero(),
            FE::zero(),
        ]);
        assert_eq!(c, d);
    }

    #[test]
    fn extend() {
        let mut a = DenseMultilinearPolynomial::new(vec![FE::from(3); 4]);
        let b = DenseMultilinearPolynomial::new(vec![FE::from(3); 4]);
        a.extend(&b);
        assert_eq!(a.len(), 8);
        assert_eq!(a.num_vars(), 3);
    }

    #[test]
    #[should_panic]
    fn extend_unequal() {
        let mut a = DenseMultilinearPolynomial::new(vec![FE::from(3); 4]);
        let b = DenseMultilinearPolynomial::new(vec![FE::from(3); 2]);
        a.extend(&b);
    }
}
