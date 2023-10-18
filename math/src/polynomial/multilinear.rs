use crate::field::element::FieldElement;
use crate::field::traits::IsField;
use std::ops::{self, Neg};

#[cfg(feature = "rayon")]
use rayon::prelude::*;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MultilinearExtension<FE> {
    pub evals: Vec<FE>,
}

impl<F: IsField> MultilinearExtension<FieldElement<F>>
where
    <F as IsField>::BaseType: Send + Sync,
{
    pub fn new(evals: &[FieldElement<F>]) -> Self {
        assert_eq!(
            evals.len(),
            (2usize).pow((evals.len() as f64).log2() as u32)
        );
        Self { evals: evals.to_vec() }
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

    pub fn evals(&self) -> &[FieldElement<F>] {
        &self.evals
    }

    pub fn to_vec(&self) -> Vec<FieldElement<F>> {
        self.evals.clone()
    }

    pub fn iter(&self) -> impl Iterator<Item = &FieldElement<F>> {
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
                    // TODO: remove this clone
                    *y = x.clone() * r;
                    *x -= &*y;
                });
            size *= 2;
        }

        assert_eq!(chis.len(), self.evals.len());
        //TODO: index based on degreee of eval[i[ chis[]
        //TODO: merge this implementation with mul_with_ref
        #[cfg(feature = "rayon")]
        let dot_product = (0..self.evals.len())
            .into_par_iter()
            .map(|i| &chis[i] * &self.evals[i])
            .sum();
        #[cfg(not(feature = "rayon"))]
        let dot_product = (0..a.len()).map(|i| a[i] * b[i]).sum();
        dot_product
    }

    pub fn mul_with_ref(&self, other: &Self) -> Self {
        let degree = self.len() + other.len();
        let mut coefficients = vec![FieldElement::zero(); degree + 1];

        if self.evals.is_empty() || other.evals.is_empty() {
            Self::new(&[FieldElement::zero()])
        } else {
            for i in 0..=other.len() {
                for j in 0..=self.len() {
                    coefficients[i + j] += &other.evals[i] * &self.evals[j];
                }
            }
            Self::new(&coefficients)
        }
    }
}

impl<F: IsField> ops::Index<usize> for MultilinearExtension<FieldElement<F>> {
    type Output = FieldElement<F>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.evals[index]
    }
}

impl<F: IsField> ops::Add for MultilinearExtension<FieldElement<F>> {
    type Output = MultilinearExtension<FieldElement<F>>;

    fn add(self, other: MultilinearExtension<FieldElement<F>>) -> Self {
        &self + &other
    }
}

impl<'a, 'b, F: IsField> ops::Add<&'a MultilinearExtension<FieldElement<F>>>
    for &'b MultilinearExtension<FieldElement<F>>
{
    type Output = MultilinearExtension<FieldElement<F>>;

    fn add(self, rhs: &'a MultilinearExtension<FieldElement<F>>) -> Self::Output {
        if rhs.evals.is_empty() {
            return self.clone();
        }
        if self.evals.is_empty() {
            return rhs.clone();
        }
        assert_eq!(self.evals.len(), rhs.evals.len());
        let evals = self
            .evals
            .iter()
            .zip(rhs.evals.iter())
            .map(|(a, b)| a + b)
            .collect();
        MultilinearExtension { evals }
    }
}

impl<F: IsField> ops::AddAssign for MultilinearExtension<FieldElement<F>> {
    fn add_assign(&mut self, rhs: Self) {
        *self = &*self + &rhs;
    }
}

impl<'a, F: IsField> ops::AddAssign<(FieldElement<F>, &'a MultilinearExtension<FieldElement<F>>)>
    for MultilinearExtension<FieldElement<F>>
{
    fn add_assign(
        &mut self,
        (f, other): (FieldElement<F>, &'a MultilinearExtension<FieldElement<F>>),
    ) {
        #[allow(clippy::suspicious_op_assign_impl)]
        let other = Self {
            evals: other.evals.iter().map(|x| &f * x).collect(),
        };
        *self = &*self + &other;
    }
}

impl<F: IsField> ops::Sub for MultilinearExtension<FieldElement<F>> {
    type Output = MultilinearExtension<FieldElement<F>>;

    fn sub(self, other: Self) -> Self::Output {
        &self - &other
    }
}

impl<'a, 'b, F: IsField> ops::Sub<&'a MultilinearExtension<FieldElement<F>>>
    for &'b MultilinearExtension<FieldElement<F>>
{
    type Output = MultilinearExtension<FieldElement<F>>;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn sub(self, rhs: &'a MultilinearExtension<FieldElement<F>>) -> Self::Output {
        self + &rhs.clone().neg()
    }
}

impl<F: IsField> ops::SubAssign for MultilinearExtension<FieldElement<F>> {
    fn sub_assign(&mut self, rhs: Self) {
        *self = &*self - &rhs;
    }
}

impl<'a, F: IsField> ops::SubAssign<&'a MultilinearExtension<FieldElement<F>>>
    for MultilinearExtension<FieldElement<F>>
{
    fn sub_assign(&mut self, other: &'a MultilinearExtension<FieldElement<F>>) {
        *self = &*self - other;
    }
}

impl<F: IsField> ops::Neg for MultilinearExtension<FieldElement<F>> {
    type Output = MultilinearExtension<FieldElement<F>>;

    fn neg(self) -> Self::Output {
        Self::Output {
            evals: self.evals.iter().map(|x| -x).collect(),
        }
    }
}

impl<F: IsField> ops::Mul for MultilinearExtension<FieldElement<F>> where
<F as IsField>::BaseType: Send + Sync {
    type Output = MultilinearExtension<FieldElement<F>>;

    fn mul(self, rhs: Self) -> Self::Output {
        self.mul_with_ref(&rhs)
    }
}