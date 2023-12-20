use crate::field::element::FieldElement;
use crate::field::traits::IsField;
use core::ops::{Add, Index, Mul};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

/// Represents a multilinear polynomials as a vector of evaluations (FieldElements) in lagrange basis
#[derive(Debug, PartialEq, Clone)]
pub struct DenseMultilinearPolynomial<F: IsField>
where
    <F as IsField>::BaseType: Send + Sync,
{
    evals: Vec<FieldElement<F>>,
    n_vars: usize, // number of variables
    len: usize,
}

impl<F: IsField> DenseMultilinearPolynomial<F>
where
    <F as IsField>::BaseType: Send + Sync,
{
    /// Build a new multilinear polynomial, from collection of multilinear monomials
    pub fn new(evals: Vec<FieldElement<F>>) -> Self {
        // Pad non-power-2 evaluations to fill out the dense multilinear polynomial
        let mut poly_evals = evals;
        while !(poly_evals.len().is_power_of_two()) {
            poly_evals.push(FieldElement::zero());
        }

        DenseMultilinearPolynomial {
            evals: poly_evals.clone(),
            n_vars: log_2(poly_evals.len()),
            len: poly_evals.len(),
        }
    }

    pub fn num_vars(&self) -> usize {
        self.n_vars
    }

    pub fn evals(&self) -> &Vec<FieldElement<F>> {
        &self.evals
    }

    pub fn len(&self) -> usize {
        self.len
    }

    /// Evaluates `self` at the point `p` in O(n) time.
    /// Note: assumes p contains points for all variables aka is not sparse.
    // Ported from a16z/Lasso
    #[allow(dead_code)]
    pub fn evaluate(&self, r: Vec<FieldElement<F>>) -> FieldElement<F> {
        // r must have a value for each variable
        assert_eq!(r.len(), self.num_vars());

        let mut chis: Vec<FieldElement<F>> =
            vec![FieldElement::one(); (2usize).pow(r.len() as u32)];
        let mut size = 1;
        for j in 0..r.len() {
            size *= 2;
            for i in (0..size).rev().step_by(2) {
                let scalar = &chis[i / 2].clone();
                chis[i] = scalar * &r[j];
                chis[i - 1] = scalar - &chis[i];
            }
        }
        assert_eq!(chis.len(), self.evals.len());
        (0..chis.len())
            .into_par_iter()
            .map(|i| &self.evals[i] * &chis[i])
            .sum()
    }

    pub fn evaluate_with(evals: &[FieldElement<F>], r: &[FieldElement<F>]) -> FieldElement<F> {
        let mut chis: Vec<FieldElement<F>> =
            vec![FieldElement::one(); (2usize).pow(r.len() as u32)];
        let mut size = 1;
        for j in 0..r.len() {
            size *= 2;
            for i in (0..size).rev().step_by(2) {
                let scalar = &chis[i / 2].clone();
                chis[i] = scalar * &r[j];
                chis[i - 1] = scalar - &chis[i];
            }
        }
        assert_eq!(chis.len(), evals.len());
        (0..evals.len()).map(|i| &evals[i] * &chis[i]).sum()
    }

    pub fn extend(&mut self, other: &DenseMultilinearPolynomial<F>) {
        assert_eq!(self.evals.len(), self.len);
        let other = other.evals.clone();
        assert_eq!(other.len(), self.len);
        self.evals.extend(other);
        self.n_vars += 1;
        self.len *= 2;
        assert_eq!(self.evals.len(), self.len);
    }

    pub fn merge(polys: &[DenseMultilinearPolynomial<F>]) -> DenseMultilinearPolynomial<F> {
        let mut z: Vec<FieldElement<F>> = Vec::new();
        for poly in polys.iter() {
            z.extend(poly.evals().clone().into_iter());
        }

        // pad the polynomial with zero polynomial at the end
        z.resize(z.len().next_power_of_two(), FieldElement::zero());

        DenseMultilinearPolynomial::new(z)
    }

    pub fn from_u64(evals: &[u64]) -> Self {
        DenseMultilinearPolynomial::new(
            (0..evals.len())
                .map(|i| FieldElement::from(evals[i]))
                .collect::<Vec<FieldElement<F>>>(),
        )
    }

    pub fn scalar_mul(&self, scalar: &FieldElement<F>) -> Self {
        let mut new_poly = self.clone();
        new_poly.evals.iter_mut().for_each(|eval| *eval *= scalar);
        new_poly
    }
}

impl<F: IsField> Index<usize> for DenseMultilinearPolynomial<F>
where
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = FieldElement<F>;

    #[inline(always)]
    fn index(&self, _index: usize) -> &FieldElement<F> {
        &(self.evals[_index])
    }
}

/// Adds another multilinear polynomial to `self`.
/// Assumes the two polynomials have the same number of variables.
impl<F: IsField> Add for DenseMultilinearPolynomial<F>
where
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = Result<Self, &'static str>;

    fn add(self, other: Self) -> Self::Output {
        if self.num_vars() != other.num_vars() {
            return Err("Polynomials must have the same number of variables");
        }

        let sum: Vec<FieldElement<F>> = self
            .evals
            .iter()
            .zip(other.evals.iter())
            .map(|(a, b)| a + b)
            .collect();

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

fn log_2(n: usize) -> usize {
    assert_ne!(n, 0);

    if n.is_power_of_two() {
        (1usize.leading_zeros() - n.leading_zeros()) as usize
    } else {
        (0usize.leading_zeros() - n.leading_zeros()) as usize
    }
}

#[cfg(test)]
mod tests {

    use crate::field::fields::u64_prime_field::U64PrimeField;

    use super::*;

    const ORDER: u64 = 101;
    type F = U64PrimeField<ORDER>;
    type FE = FieldElement<F>;

    pub fn evals(r: Vec<FE>) -> Vec<FE> {
        let ell = r.len();

        let mut evals: Vec<FE> = vec![FE::one(); (2usize).pow(r.len() as u32)];
        let mut size = 1;
        for j in 0..ell {
            size *= 2;
            for i in (0..size).rev().step_by(2) {
                let scalar = evals[i / 2];
                evals[i] = scalar * r[j];
                evals[i - 1] = scalar - evals[i];
            }
        }
        evals
    }

    pub fn compute_factored_evals(r: Vec<FE>) -> (Vec<FE>, Vec<FE>) {
        let size = r.len();
        let (left_num_vars, _right_num_vars) = (size / 2, size - size / 2);

        let L = evals(r[..left_num_vars].to_vec());
        let R = evals(r[left_num_vars..size].to_vec());

        (L, R)
    }

    fn evaluate_with_lr(z: &[FE], r: &[FE]) -> FE {
        let (l, r) = compute_factored_evals(r.to_vec());

        let size = r.len();
        // ensure size is even
        assert!(size % 2 == 0);
        // n = 2^size
        let n = (2usize).pow(size as u32);
        // compute m = sqrt(n) = 2^{\ell/2}
        let m = (n as f64).sqrt() as usize;

        // compute vector-matrix product between L and Z viewed as a matrix
        let lz = (0..m)
            .map(|i| {
                (0..m).fold(FE::zero(), |mut acc, j| {
                    acc += l[j] * z[j * m + i];
                    acc
                })
            })
            .collect::<Vec<FE>>();

        // compute dot product between LZ and R
        (0..lz.len()).map(|i| &lz[i] * &r[i]).sum()
    }

    #[test]
    fn evaluation() {
        // Z = [1, 2, 1, 4]
        let Z = vec![FE::one(), FE::from(2u64), FE::one(), FE::from(4u64)];

        // r = [4,3]
        let r = vec![FE::from(4u64), FE::from(3u64)];

        let eval_with_lr = evaluate_with_lr(&Z, &r);
        let poly = DenseMultilinearPolynomial::new(Z);

        let eval = poly.evaluate(r);
        assert_eq!(eval, FE::from(28u64));
        assert_eq!(eval_with_lr, eval);
    }

    #[test]
    fn evaluate_with() {
        todo!()
    }

    #[test]
    fn add() {
        todo!()
    }

    #[test]
    fn mul() {
        todo!()
    }

    #[test]
    fn merge() {
        todo!()
    }

    #[test]
    fn extend() {
        todo!()
    }
}
