use crate::field::traits::IsField;
use crate::field::{element::FieldElement, traits::IsPrimeField};
use std::ops::{self, Neg};

#[cfg(feature = "rayon")]
use rayon::prelude::*;

use super::{MultiLinearTerm, Term};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MultilinearExtension<F: IsField + IsPrimeField + Default>
where
    <F as IsField>::BaseType: Send + Sync,
{
    pub evals: Vec<MultiLinearTerm<F>>,
}

impl<F: IsField + IsPrimeField + Default> MultilinearExtension<F>
where
    <F as IsField>::BaseType: Send + Sync,
{
    pub fn new(evals: &[MultiVariateMonomial<F>]) -> Self {
        assert_eq!(
            evals.len(),
            (2usize).pow((evals.len() as f64).log2() as u32)
        );
        Self {
            evals: evals.to_vec(),
        }
    }

    pub const fn zero() -> Self {
        Self { evals: Vec::new() }
    }

    pub fn is_empty(&self) -> bool {
        self.evals.is_empty()
    }

    pub fn len(&self) -> usize {
        self.evals.len()
    }

    pub fn terms(&self) -> &[MultiVariateMonomial<F>] {
        &self.evals
    }

    pub fn to_vec(&self) -> Vec<MultiVariateMonomial<F>> {
        self.evals.clone()
    }

    pub fn iter(&self) -> impl Iterator<Item = &MultiVariateMonomial<F>> {
        self.evals.iter()
    }

    // Notation we evaluate ~f at point x.
    // this follows the implementation used in Spartan that evaluates in O(n) using memoized algorithm in Thaler.
    pub fn evaluate(&self, x: &[FieldElement<F>]) -> FieldElement<F> {
        assert_eq!(x.len(), self.evals.len());
        // TODO: convert this to support a sparse representation by setting ell = max_degree(Vec<Term>)
        let ell = self.evals.len();

        let mut chis: Vec<FieldElement<F>> = vec![FieldElement::<F>::one(); 2usize.pow(ell as u32)];
        let mut size = 1;
        //NOTE: this was borrowed from Lasso see is spartans original with par_iter is faster:
        //https://github.com/microsoft/Spartan2/blob/main/src/spartan/polynomial.rs#L25
        for r in self.evals.iter().rev() {
            let (evals_left, evals_right) = chis.split_at_mut(size);
            let (evals_right, _) = evals_right.split_at_mut(size);

            evals_left
                .par_iter_mut()
                .zip(evals_right.par_iter_mut())
                .for_each(|(x, y)| {
                    // TODO: remove this slice wrapper
                    *y = x.clone() * r.evaluate(&[x.clone()]);
                    *x -= &*y;
                });
            size *= 2;
        }

        assert_eq!(chis.len(), self.evals.len());
        //TODO: index based on degreee of eval[i[ chis[]
        //TODO: merge this implementation with mul_with_ref
        //TODO: cache and access computed evaluations from chi calculation
        #[cfg(feature = "rayon")]
        let dot_product = (0..self.evals.len())
            .into_par_iter()
            .map(|i| &chis[i] * &self.evals[i].evaluate(x))
            .sum();
        #[cfg(not(feature = "rayon"))]
        let dot_product = (0..a.len()).map(|i| a[i] * b[i]).sum();
        dot_product
    }

    //TODO: this needs on the term level add vars to each respective var vector
    pub fn mul_with_ref(&self, other: &Self) -> Self {
        //This should be abstracted as a helper in term but I'm still susing this out
        let merge_vars = |a: Vec<(usize, usize)>, b: &[(usize, usize)]| {
            a.iter().zip(b.iter()).fold(Vec::new(), |mut acc, (a, b)| {
                // If we have two variables have the same var_id we add there powers
                // If they have different var_ids add both variables to term
                if a.0 == b.0 {
                    acc.push((a.0, a.1 + b.1));
                } else {
                    acc.push(*a);
                    acc.push(*b);
                }
                acc
            })
        };

        let degree = self.len() + other.len();
        let mut coefficients = vec![MultiVariateMonomial::default(); degree + 1];

        if self.evals.is_empty() || other.evals.is_empty() {
            Self::new(&[MultiVariateMonomial::default()])
        } else {
            for i in 0..=other.len() {
                for j in 0..=self.len() {
                    let vars = merge_vars(self.evals[j].vars.clone(), &other.evals[i].vars);
                    coefficients[i + j] = MultiVariateMonomial {
                        coeff: &other.evals[i].coeff * &self.evals[j].coeff,
                        vars,
                    };
                }
            }
            Self::new(&coefficients)
        }
    }
}

impl<F: IsField + IsPrimeField + Default> ops::Index<usize> for MultilinearExtension<F>
where
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = MultiVariateMonomial<F>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.evals[index]
    }
}

impl<F: IsField + IsPrimeField + Default> ops::Add for MultilinearExtension<F>
where
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = MultilinearExtension<F>;

    fn add(self, other: MultilinearExtension<F>) -> Self {
        &self + &other
    }
}

impl<'a, 'b, F: IsField + IsPrimeField + Default> ops::Add<&'a MultilinearExtension<F>>
    for &'b MultilinearExtension<F>
where
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = MultilinearExtension<F>;

    fn add(self, rhs: &'a MultilinearExtension<F>) -> Self::Output {
        if rhs.evals.is_empty() {
            return self.clone();
        }
        if self.evals.is_empty() {
            return rhs.clone();
        }
        //TODO: This needs assertions specific to sparse implementation aka terms need to match
        assert_eq!(self.evals.len(), rhs.evals.len());
        let evals = self
            .evals
            .iter()
            .zip(rhs.evals.iter())
            //TODO: This should probs be implemented in term but for now...
            .map(|(a, b)| MultiVariateMonomial {
                coeff: &a.coeff + &b.coeff,
                vars: a.vars.clone(),
            })
            .collect();
        MultilinearExtension { evals }
    }
}

impl<F: IsField + IsPrimeField + Default> ops::AddAssign for MultilinearExtension<F>
where
    <F as IsField>::BaseType: Send + Sync,
{
    fn add_assign(&mut self, rhs: Self) {
        *self = &*self + &rhs;
    }
}

impl<'a, F: IsField + IsPrimeField + Default>
    ops::AddAssign<(FieldElement<F>, &'a MultilinearExtension<F>)> for MultilinearExtension<F>
where
    <F as IsField>::BaseType: Send + Sync,
{
    fn add_assign(&mut self, (f, other): (FieldElement<F>, &'a MultilinearExtension<F>)) {
        #[allow(clippy::suspicious_op_assign_impl)]
        let other = Self {
            //TODO: this should be in place and in term... gotta sus out this dispatch interface... thinking more and more I want to try using enum dispatch since we have a bounded number of types
            evals: other
                .evals
                .iter()
                .map(|x| MultiVariateMonomial {
                    coeff: &f * &x.coeff,
                    vars: x.vars.clone(),
                })
                .collect(),
        };
        *self = &*self + &other;
    }
}

impl<F: IsField + IsPrimeField + Default> ops::Sub for MultilinearExtension<F>
where
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = MultilinearExtension<F>;

    fn sub(self, other: Self) -> Self::Output {
        &self - &other
    }
}

impl<'a, 'b, F: IsField + IsPrimeField + Default> ops::Sub<&'a MultilinearExtension<F>>
    for &'b MultilinearExtension<F>
where
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = MultilinearExtension<F>;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn sub(self, rhs: &'a MultilinearExtension<F>) -> Self::Output {
        self + &rhs.clone().neg()
    }
}

impl<F: IsField + IsPrimeField + Default> ops::SubAssign for MultilinearExtension<F>
where
    <F as IsField>::BaseType: Send + Sync,
{
    fn sub_assign(&mut self, rhs: Self) {
        *self = &*self - &rhs;
    }
}

impl<'a, F: IsField + IsPrimeField + Default> ops::SubAssign<&'a MultilinearExtension<F>>
    for MultilinearExtension<F>
where
    <F as IsField>::BaseType: Send + Sync,
{
    fn sub_assign(&mut self, other: &'a MultilinearExtension<F>) {
        *self = &*self - other;
    }
}

impl<F: IsField + IsPrimeField + Default> ops::Neg for MultilinearExtension<F>
where
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = MultilinearExtension<F>;

    fn neg(self) -> Self::Output {
        Self::Output {
            evals: self
                .evals
                .iter()
                .map(|x| MultiVariateMonomial {
                    coeff: -&x.coeff,
                    vars: x.vars.clone(),
                })
                .collect(),
        }
    }
}

impl<F: IsField + IsPrimeField + Default> ops::Mul for MultilinearExtension<F>
where
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = MultilinearExtension<F>;

    fn mul(self, rhs: Self) -> Self::Output {
        self.mul_with_ref(&rhs)
    }
}
