use core::fmt::Display;
use std::ops;

use rayon::prelude::*;

use crate::{
    errors::TermError,
    field::{
        element::FieldElement,
        traits::{IsField, IsPrimeField},
    },
    polynomial::{terms::multivariate::MultivariateMonomial, traits::term::Term},
};

use super::traits::polynomial::IsPolynomial;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MultivariatePolynomial<F: IsField + IsPrimeField>
where
    <F as IsField>::BaseType: Send + Sync,
{
    pub terms: Vec<MultivariateMonomial<F>>,
    pub num_vars: usize,
    pub len: usize,
}

//TODO: The holy grale is eliminate Sparse/Dense with Enum dispatch
impl<F: IsField + IsPrimeField> MultivariatePolynomial<F>
where
    <F as IsField>::BaseType: Send + Sync,
{
    fn iter(&self) -> impl Iterator<Item = &MultivariateMonomial<F>> {
        self.terms.iter()
    }

    fn partial_evaluate(
        &self,
        assignments: &[(usize, FieldElement<F>)],
    ) -> Result<Self, TermError> {
        if assignments.len() > self.num_vars {
            return Err(TermError::InvalidPartialEvaluationPoint);
        }
        let updated_monomials: Vec<MultivariateMonomial<F>> = self
            .terms
            .iter()
            .map(|term| term.partial_evaluate(assignments))
            .collect();
        Ok(Self::new(self.num_vars, &updated_monomials))
    }

    //Add checks for different terms
    //TODO: does this work with terms of different len -> add check
    fn mul_with_ref(&self, factor: &Self) -> Self {
        let len = self.len() + factor.len();
        let mut terms = vec![MultivariateMonomial::zero(); len + 1];

        if self.terms.is_empty() || factor.terms.is_empty() {
            Self::new(1, &[MultivariateMonomial::zero()])
        } else {
            // TODO: add rayon
            for i in 0..=factor.len() {
                for j in 0..=self.len() {
                    terms[i + j] += &factor.terms[i] * &self.terms[j];
                }
            }
            Self::new(self.num_vars, &terms)
        }
    }
}

impl<F: IsField + IsPrimeField> IsPolynomial<F, MultivariateMonomial<F>>
    for MultivariatePolynomial<F>
where
    <F as IsField>::BaseType: Send + Sync,
{
    type T = MultivariateMonomial<F>;

    fn new(num_vars: usize, terms: &[Self::T]) -> Self {
        Self {
            terms: terms.to_vec(),
            num_vars,
            len: terms.len(),
        }
    }

    fn degree(&self) -> usize {
        if let Some(term) = self.terms.last() {
            term.degree()
        } else {
            0
        }
    }

    // convertion is end is highest power
    fn leading_coefficient(&self) -> FieldElement<F> {
        if let Some(term) = self.terms.last() {
            term.coeff.clone()
        } else {
            FieldElement::zero()
        }
    }

    fn coeffs(&self) -> &[FieldElement<F>] {
        //TODO: eliminate grabbing owned reference via implementing deref/caching
        let coeff = self.terms.clone().iter().map(|t| t.coeff.clone()).collect::<Vec<_>>();
        &coeff
    }

    fn len(&self) -> usize {
        self.terms.len()
    }

    fn extend(&mut self, term: MultivariateMonomial<F>) {
        self.terms.extend_from_slice(&[term]);
        //Check if term respects ordering convention
        //TODO: need to implement Ord
        /*
        if term.coeff < self.leading_coefficient() {
            self.terms.sort()
        }
        */
    }

    /// Evaluates `self` at the point `p`.
    fn evaluate(&self, p: &[FieldElement<F>]) -> Result<FieldElement<F>, TermError> {
        if p.len() != self.num_vars {
            return Err(TermError::InvalidPartialEvaluationPoint);
        }

        let res = self
            .terms
            .iter()
            .fold(FieldElement::<F>::zero(), |mut acc, term| {
                acc += term.evaluate(p);
                acc
            });
        Ok(res)
    }

    /// Evaluates terms at point without allocating a new poly
    fn evaluate_with(
        terms: &[MultivariateMonomial<F>],
        p: &[FieldElement<F>],
    ) -> Result<FieldElement<F>, TermError> {
        //TODO: improve this check
        if p.len() != terms[0].max_var() {
            return Err(TermError::InvalidPartialEvaluationPoint);
        }
        let res = terms
            .iter()
            .fold(FieldElement::<F>::zero(), |mut acc, term| {
                acc += term.evaluate(p);
                acc
            });
        Ok(res)
    }

    fn eval_at_zero(&self) -> FieldElement<F> {
        self.terms[0].coeff.clone()
    }

    fn eval_at_one(&self) -> FieldElement<F> {
        (0..self.terms.len())
            .into_par_iter()
            .map(|i| self.terms[i].coeff.clone())
            .sum()
    }

    fn zero() -> Self {
        Self {
            terms: vec![],
            num_vars: 0,
            len: 0,
        }
    }

    // Two cases
    // 1.) empty vector
    // 2.) all coeffs are zero
    fn is_zero(&self) -> bool {
        self.is_empty()
            || self
                .terms
                .iter()
                .any(|t| t.coeff == FieldElement::<F>::zero())
    }

    fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }

    fn pad_with_zero_coefficients_to_length(pa: &mut Self, n: usize) {
        pa.terms.resize(n, Self::T::zero());
    }

    /// Pads polynomial representations with minimum number of zeros to match lengths.
    fn pad_with_zero_coefficients(pa: &Self, pb: &Self) -> (Self, Self)
    where
        Self: Sized,
    {
        let mut pa = pa.clone();
        let mut pb = pb.clone();

        if pa.coeffs().len() > pb.coeffs().len() {
            Self::pad_with_zero_coefficients_to_length(&mut pb, pa.coeffs().len());
        } else {
            Self::pad_with_zero_coefficients_to_length(&mut pa, pb.coeffs().len());
        }
        (pa, pb)
    }

    fn scale(&self, factor: &FieldElement<F>) -> Self {
        let scaled_terms = self
            .terms
            .iter()
            .zip(core::iter::successors(Some(FieldElement::one()), |x| {
                Some(x * factor)
            }))
            .map(|(term, power)| Self::T {
                coeff: power * term.coeff.clone(),
                vars: term.vars.clone(),
            })
            .collect::<Vec<_>>();
        let len = scaled_terms.len();
        Self {
            terms: scaled_terms,
            num_vars: self.num_vars,
            len
        }
    }

    //TODO: should we add an in place method???
    fn scale_coeffs(&self, factor: &FieldElement<F>) -> Self {
        let scaled_terms = self
            .terms
            .iter()
            .map(|term| Self::T {
                coeff: factor * term.coeff.clone(),
                vars: term.vars.clone(),
            })
            .collect::<Vec<_>>();
        let len = scaled_terms.len();
        Self {
            terms: scaled_terms,
            num_vars: self.num_vars,
            len
        }
    }

    /// Returns a vector of polynomials [p₀, p₁, ..., p_{d-1}], where d is `number_of_parts`, such that `self` equals
    /// p₀(Xᵈ) + Xp₁(Xᵈ) + ... + X^(d-1)p_{d-1}(Xᵈ).
    ///
    /// Example: if d = 2 and `self` is 3 X^3 + X^2 + 2X + 1, then `poly.break_in_parts(2)`
    /// returns a vector with two polynomials `(p₀, p₁)`, where p₀ = X + 1 and p₁ = 3X + 2.
    fn split_n_ways(&self, n: usize) -> Vec<Self>
    where
        Self: Sized,
    {
        let terms = &self.terms;
        let mut parts: Vec<Self> = Vec::with_capacity(n);
        for i in 0..n {
            let terms: Vec<_> = terms.par_iter().skip(i).step_by(n).cloned().collect();
            parts.push(Self::new(1, &terms));
        }
        parts
    }
}

impl<F: IsField + IsPrimeField> ops::Add<&MultivariatePolynomial<F>> for &MultivariatePolynomial<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = MultivariatePolynomial<F>;

    //TODO: consider parallelism... probs not possible as every term must check all terms
    //NOTE: since we check the vars for each term implicitly we don't error
    //NOTE: Since we are using a sparse format we iterate over each
    fn add(self, a_polynomial: &MultivariatePolynomial<F>) -> Self::Output {
        let mut new_terms = self.terms.clone();
        for mono in a_polynomial.terms.clone().into_iter() {
            let mut added = false; // flag to check if the monomial was added or not

            //TODO: rayon
            for term in new_terms.iter_mut() {
                if term.vars == mono.vars {
                    *term += mono.clone();
                    added = true;
                }
            }

            if !added {
                new_terms.push(mono);
            }
        }
        //Sort terms in order once complete -> Requires Ord
        MultivariatePolynomial::new(self.num_vars, &new_terms)
    }
}

impl<F: IsField + IsPrimeField> ops::Add<MultivariatePolynomial<F>> for MultivariatePolynomial<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = MultivariatePolynomial<F>;

    fn add(self, a_polynomial: MultivariatePolynomial<F>) -> MultivariatePolynomial<F> {
        &self + &a_polynomial
    }
}

impl<F: IsField + IsPrimeField> ops::Add<&MultivariatePolynomial<F>> for MultivariatePolynomial<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = MultivariatePolynomial<F>;

    fn add(self, a_polynomial: &MultivariatePolynomial<F>) -> MultivariatePolynomial<F> {
        &self + a_polynomial
    }
}

impl<F: IsField + IsPrimeField> ops::Add<MultivariatePolynomial<F>> for &MultivariatePolynomial<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = MultivariatePolynomial<F>;

    fn add(self, a_polynomial: MultivariatePolynomial<F>) -> MultivariatePolynomial<F> {
        self + &a_polynomial
    }
}

//Add Assign:
impl<F: IsField + IsPrimeField> ops::AddAssign<MultivariatePolynomial<F>>
    for MultivariatePolynomial<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    //NOTE: since we check the vars for each term implicitly we don't error
    fn add_assign(&mut self, rhs: MultivariatePolynomial<F>) {
        for mono in rhs.terms {
            let mut added = false; // flag to check if the monomial was added or not

            //TODO: rayon or flip this loop
            for term in self.terms.iter_mut() {
                if term.vars == mono.vars {
                    *term += mono.clone();
                    added = true;
                }
            }

            if !added {
                self.terms.push(mono);
            }
        }
    }
}

impl<F: IsField + IsPrimeField> ops::Neg for &MultivariatePolynomial<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = MultivariatePolynomial<F>;

    fn neg(self) -> MultivariatePolynomial<F> {
        let neg = self
            .terms
            .iter()
            .map(|x| -x)
            .collect::<Vec<MultivariateMonomial<F>>>();
        MultivariatePolynomial::new(1, &neg)
    }
}

impl<F: IsField + IsPrimeField> ops::Neg for MultivariatePolynomial<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = MultivariatePolynomial<F>;

    fn neg(self) -> MultivariatePolynomial<F> {
        -&self
    }
}

impl<F: IsField + IsPrimeField> ops::Sub<MultivariatePolynomial<F>> for MultivariatePolynomial<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = MultivariatePolynomial<F>;

    fn sub(self, substrahend: MultivariatePolynomial<F>) -> MultivariatePolynomial<F> {
        &self - &substrahend
    }
}

impl<F: IsField + IsPrimeField> ops::Sub<&MultivariatePolynomial<F>> for MultivariatePolynomial<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = MultivariatePolynomial<F>;

    fn sub(self, substrahend: &MultivariatePolynomial<F>) -> MultivariatePolynomial<F> {
        &self - substrahend
    }
}

impl<F: IsField + IsPrimeField> ops::Sub<MultivariatePolynomial<F>> for &MultivariatePolynomial<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = MultivariatePolynomial<F>;

    fn sub(self, substrahend: MultivariatePolynomial<F>) -> MultivariatePolynomial<F> {
        self - &substrahend
    }
}

impl<F: IsField + IsPrimeField> ops::Sub<&MultivariatePolynomial<F>> for &MultivariatePolynomial<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = MultivariatePolynomial<F>;

    fn sub(self, substrahend: &MultivariatePolynomial<F>) -> MultivariatePolynomial<F> {
        self + (-substrahend)
    }
}

impl<F: IsField + IsPrimeField> ops::Mul<&MultivariatePolynomial<F>> for &MultivariatePolynomial<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = MultivariatePolynomial<F>;
    fn mul(self, factor: &MultivariatePolynomial<F>) -> MultivariatePolynomial<F> {
        self.mul_with_ref(factor)
    }
}

impl<F: IsField + IsPrimeField> ops::Mul<MultivariatePolynomial<F>> for MultivariatePolynomial<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = MultivariatePolynomial<F>;
    fn mul(self, factor: MultivariatePolynomial<F>) -> MultivariatePolynomial<F> {
        &self * &factor
    }
}

impl<F: IsField + IsPrimeField> ops::Mul<MultivariatePolynomial<F>> for &MultivariatePolynomial<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = MultivariatePolynomial<F>;
    fn mul(self, factor: MultivariatePolynomial<F>) -> MultivariatePolynomial<F> {
        self * &factor
    }
}

impl<F: IsField + IsPrimeField> ops::Mul<&MultivariatePolynomial<F>> for MultivariatePolynomial<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = MultivariatePolynomial<F>;
    fn mul(self, factor: &MultivariatePolynomial<F>) -> MultivariatePolynomial<F> {
        &self * factor
    }
}

/* Operations between Selfs and field elements */
/* Multiplication field element at left */
impl<F: IsField + IsPrimeField> ops::Mul<FieldElement<F>> for MultivariatePolynomial<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = MultivariatePolynomial<F>;

    fn mul(self, multiplicand: FieldElement<F>) -> MultivariatePolynomial<F> {
        let new_terms = self
            .terms
            .iter()
            .map(|value| value * &multiplicand)
            .collect::<Vec<_>>();
        let len = new_terms.len();

        Self {
            terms: new_terms,
            num_vars: self.num_vars,
            len,
        }
    }
}

impl<F: IsField + IsPrimeField> ops::Mul<&FieldElement<F>> for &MultivariatePolynomial<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = MultivariatePolynomial<F>;

    fn mul(self, multiplicand: &FieldElement<F>) -> MultivariatePolynomial<F> {
        self.clone() * multiplicand.clone()
    }
}

impl<F: IsField + IsPrimeField> ops::Mul<FieldElement<F>> for &MultivariatePolynomial<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = MultivariatePolynomial<F>;

    fn mul(self, multiplicand: FieldElement<F>) -> MultivariatePolynomial<F> {
        self * &multiplicand
    }
}

impl<F: IsField + IsPrimeField> ops::Mul<&FieldElement<F>> for MultivariatePolynomial<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = MultivariatePolynomial<F>;

    fn mul(self, multiplicand: &FieldElement<F>) -> MultivariatePolynomial<F> {
        &self * multiplicand
    }
}

/* Multiplication field element at right */
impl<F: IsField + IsPrimeField> ops::Mul<&MultivariatePolynomial<F>> for &FieldElement<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = MultivariatePolynomial<F>;

    fn mul(self, multiplicand: &MultivariatePolynomial<F>) -> MultivariatePolynomial<F> {
        multiplicand * self
    }
}

impl<F: IsField + IsPrimeField> ops::Mul<MultivariatePolynomial<F>> for &FieldElement<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = MultivariatePolynomial<F>;

    fn mul(self, multiplicand: MultivariatePolynomial<F>) -> MultivariatePolynomial<F> {
        &multiplicand * self
    }
}

impl<F: IsField + IsPrimeField> ops::Mul<&MultivariatePolynomial<F>> for FieldElement<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = MultivariatePolynomial<F>;

    fn mul(self, multiplicand: &MultivariatePolynomial<F>) -> MultivariatePolynomial<F> {
        multiplicand * self
    }
}

impl<F: IsField + IsPrimeField> ops::Mul<MultivariatePolynomial<F>> for FieldElement<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = MultivariatePolynomial<F>;

    fn mul(self, multiplicand: MultivariatePolynomial<F>) -> MultivariatePolynomial<F> {
        &multiplicand * &self
    }
}

/* Addition field element at left */
impl<F: IsField + IsPrimeField> ops::Add<&FieldElement<F>> for &MultivariatePolynomial<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = MultivariatePolynomial<F>;

    fn add(self, other: &FieldElement<F>) -> MultivariatePolynomial<F> {
        let term = MultivariateMonomial::new((other.clone(), vec![]));
        MultivariatePolynomial::new(self.num_vars, &[term]) + self
    }
}

impl<F: IsField + IsPrimeField> ops::Add<FieldElement<F>> for MultivariatePolynomial<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = MultivariatePolynomial<F>;

    fn add(self, other: FieldElement<F>) -> MultivariatePolynomial<F> {
        &self + &other
    }
}

impl<F: IsField + IsPrimeField> ops::Add<FieldElement<F>> for &MultivariatePolynomial<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = MultivariatePolynomial<F>;

    fn add(self, other: FieldElement<F>) -> MultivariatePolynomial<F> {
        self + &other
    }
}

impl<F: IsField + IsPrimeField> ops::Add<&FieldElement<F>> for MultivariatePolynomial<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = MultivariatePolynomial<F>;

    fn add(self, other: &FieldElement<F>) -> MultivariatePolynomial<F> {
        &self + other
    }
}

/* Addition field element at right */
impl<F: IsField + IsPrimeField> ops::Add<&MultivariatePolynomial<F>> for &FieldElement<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = MultivariatePolynomial<F>;

    fn add(self, other: &MultivariatePolynomial<F>) -> MultivariatePolynomial<F> {
        let term = MultivariateMonomial::new((self.clone(), vec![]));
        MultivariatePolynomial::new(other.num_vars, &[term]) + other
    }
}

impl<F: IsField + IsPrimeField> ops::Add<MultivariatePolynomial<F>> for FieldElement<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = MultivariatePolynomial<F>;

    fn add(self, other: MultivariatePolynomial<F>) -> MultivariatePolynomial<F> {
        &self + &other
    }
}

impl<F: IsField + IsPrimeField> ops::Add<MultivariatePolynomial<F>> for &FieldElement<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = MultivariatePolynomial<F>;

    fn add(self, other: MultivariatePolynomial<F>) -> MultivariatePolynomial<F> {
        self + &other
    }
}

impl<F: IsField + IsPrimeField> ops::Add<&MultivariatePolynomial<F>> for FieldElement<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = MultivariatePolynomial<F>;

    fn add(self, other: &MultivariatePolynomial<F>) -> MultivariatePolynomial<F> {
        &self + other
    }
}

/* Substraction field element at left */
impl<F: IsField + IsPrimeField> ops::Sub<&FieldElement<F>> for &MultivariatePolynomial<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = MultivariatePolynomial<F>;

    fn sub(self, other: &FieldElement<F>) -> MultivariatePolynomial<F> {
        let term = MultivariateMonomial::new((other.clone(), vec![]));
        self - MultivariatePolynomial::new(self.num_vars, &[term])
    }
}

impl<F: IsField + IsPrimeField> ops::Sub<FieldElement<F>> for MultivariatePolynomial<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = MultivariatePolynomial<F>;

    fn sub(self, other: FieldElement<F>) -> MultivariatePolynomial<F> {
        &self - &other
    }
}

impl<F: IsField + IsPrimeField> ops::Sub<FieldElement<F>> for &MultivariatePolynomial<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = MultivariatePolynomial<F>;

    fn sub(self, other: FieldElement<F>) -> MultivariatePolynomial<F> {
        self - &other
    }
}

impl<F: IsField + IsPrimeField> ops::Sub<&FieldElement<F>> for MultivariatePolynomial<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = MultivariatePolynomial<F>;

    fn sub(self, other: &FieldElement<F>) -> MultivariatePolynomial<F> {
        &self - other
    }
}

/* Substraction field element at right */
impl<F: IsField + IsPrimeField> ops::Sub<&MultivariatePolynomial<F>> for &FieldElement<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = MultivariatePolynomial<F>;

    fn sub(self, other: &MultivariatePolynomial<F>) -> MultivariatePolynomial<F> {
        let term = MultivariateMonomial::new((self.clone(), vec![]));
        MultivariatePolynomial::new(other.num_vars, &[term]) - other
    }
}

impl<F: IsField + IsPrimeField> ops::Sub<MultivariatePolynomial<F>> for FieldElement<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = MultivariatePolynomial<F>;

    fn sub(self, other: MultivariatePolynomial<F>) -> MultivariatePolynomial<F> {
        &self - &other
    }
}

impl<F: IsField + IsPrimeField> ops::Sub<MultivariatePolynomial<F>> for &FieldElement<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = MultivariatePolynomial<F>;

    fn sub(self, other: MultivariatePolynomial<F>) -> MultivariatePolynomial<F> {
        self - &other
    }
}

impl<F: IsField + IsPrimeField> ops::Sub<&MultivariatePolynomial<F>> for FieldElement<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = MultivariatePolynomial<F>;

    fn sub(self, other: &MultivariatePolynomial<F>) -> MultivariatePolynomial<F> {
        &self - other
    }
}

impl<F: IsField + IsPrimeField> ops::Index<usize> for MultivariatePolynomial<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = MultivariateMonomial<F>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.terms[index]
    }
}

impl<F: IsField + IsPrimeField> ops::IndexMut<usize> for MultivariatePolynomial<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.terms[index]
    }
}

impl<F: IsField + IsPrimeField> Display for MultivariatePolynomial<F>
where
    F: IsField,
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

//TODO: Question should we make coefficients private and implement Iterator, IterMut, IntoIterator and parallel counterparts

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::element::FieldElement;
    use crate::field::fields::u64_prime_field::U64PrimeField;
    use crate::polynomial::multivariate::MultivariateMonomial;

    const ORDER: u64 = 101;
    type F = U64PrimeField<ORDER>;
    type FE = FieldElement<F>;

    //TODO:
    // - [ ] degreee
    // - [ ] leasing coeff
    // - [ ] vars
    // - [ ] powers
    // - [ ] max_var
    // - [ ] evaluate
    //      - [ ] zero poly
    //      - [ ] one poly
    //      - [ ] invalid number of points
    // - [ ] partial_evaluate
    //      - [ ] zero poly
    //      - [ ] one poly
    //      - [ ] invalid number of points
    // - [ ] evaluate_with
    //
    // Ops:
    // - [ ] add
    //      - two values
    //      - assc
    //      - commutative
    //      - negative
    // - [ ] sub two assc
    //      - two values
    //      - assc
    //      - commutative
    //      - negative
    // - [ ] mul
    //      - two
    //      - assc
    //      - commutative
    //      - negative
    //      - check mul by scalar F
    // - [ ] neg
    //      - check sign is flipped
    // - [ ] Interpolate
    // - [ ] extend
    // - [ ] split_n_ways
    // - [ ]
    // - [ ]
    // - [ ]

    #[test]
    fn test_partial_evaluation() {
        // 3ab + 4bc
        // partially evaluate b = 2
        // expected result = 6a + 8c
        // a = 1, b = 2, c = 3
        let poly = MultivariatePolynomial::new(
            4,
            &[
                MultivariateMonomial::new((FE::new(3), vec![(1, 1), (2, 1)])),
                MultivariateMonomial::new((FE::new(4), vec![(2, 1), (3, 1)])),
            ],
        );
        let result = poly.partial_evaluate(&[(2, FE::new(2))]);
        assert_eq!(
            result.unwrap(),
            MultivariatePolynomial {
                terms: vec![
                    MultivariateMonomial {
                        coeff: FE::new(6),
                        vars: vec![(1, 1)]
                    },
                    MultivariateMonomial {
                        coeff: FE::new(8),
                        vars: vec![(3, 1)]
                    }
                ],
                num_vars: 4,
                len: 2
            }
        );
    }

    #[test]
    fn test_all_vars_evaluation() {
        // 3abc + 4abc
        // evaluate: a = 1, b = 2, c = 3
        // expected result = 42
        // a = 1, b = 2, c = 3
        let poly = MultivariatePolynomial::new(
            3,
            &[
                MultivariateMonomial::new((FE::new(3), vec![(0, 1), (1, 1), (2, 1)])),
                MultivariateMonomial::new((FE::new(4), vec![(0, 1), (1, 1), (2, 1)])),
            ],
        );
        let result = poly.evaluate(&[FE::one(), FE::new(2), FE::new(3)]);
        assert_eq!(result.unwrap(), FE::new(42));
    }

    #[test]
    fn test_partial_vars_evaluation() {
        // 3ab + 4bc
        // evaluate: a = 1, b = 2, c = 3
        // expected result = 30
        // a = 1, b = 2, c = 3
        let poly = MultivariatePolynomial::new(
            3,
            &[
                MultivariateMonomial::new((FE::new(3), vec![(0, 1), (1, 1)])),
                MultivariateMonomial::new((FE::new(4), vec![(1, 1), (2, 1)])),
            ],
        );
        let result = poly.evaluate(&[FE::one(), FE::new(2), FE::new(3)]);
        assert_eq!(result.unwrap(), FE::new(30));
    }

    #[test]
    fn test_evaluate_incorrect_vars_len() {
        // 3ab + 4bc
        // evaluate: a = 1, b = 2, c = 3
        // expected result = 30
        // a = 1, b = 2, c = 3
        let poly = MultivariatePolynomial::new(
            3,
            &[
                MultivariateMonomial::new((FE::new(3), vec![(0, 1), (1, 1)])),
                MultivariateMonomial::new((FE::new(4), vec![(1, 1), (2, 1)])),
            ],
        );
        assert!(poly.evaluate(&[FE::one(), FE::new(2)]).is_err());
    }

    #[test]
    fn test_partial_evaluation_incorrect_var_len() {
        // 3ab + 4bc
        // partially evaluate b = 2
        // expected result = 6a + 8c
        // a = 1, b = 2, c = 3
        let poly = MultivariatePolynomial::new(
            3,
            &[
                MultivariateMonomial::new((FE::new(3), vec![(1, 1), (2, 1)])),
                MultivariateMonomial::new((FE::new(4), vec![(2, 1), (3, 1)])),
            ],
        );
        assert!(poly
            .partial_evaluate(&[
                (2, FE::new(2)),
                (1, FE::new(2)),
                (3, FE::new(2)),
                (4, FE::new(2)),
                (5, FE::new(2)),
            ])
            .is_err());
    }

    // Some of these tests work when the finite field has order greater than 2.
    fn polynomial_a() -> MultivariatePolynomial<F> {
        MultivariatePolynomial::new(1, &[FE::new(1), FE::new(2), FE::new(3)])
    }

    fn polynomial_minus_a() -> MultivariatePolynomial<F> {
        MultivariatePolynomial::new(
            1,
            &[FE::new(ORDER - 1), FE::new(ORDER - 2), FE::new(ORDER - 3)],
        )
    }

    fn polynomial_b() -> MultivariatePolynomial<F> {
        MultivariatePolynomial::new(1, &[FE::new(3), FE::new(4), FE::new(5)])
    }

    fn polynomial_a_plus_b() -> MultivariatePolynomial<F> {
        MultivariatePolynomial::new(1, &[FE::new(4), FE::new(6), FE::new(8)])
    }

    fn polynomial_b_minus_a() -> MultivariatePolynomial<F> {
        MultivariatePolynomial::new(1, &[FE::new(2), FE::new(2), FE::new(2)])
    }

    #[test]
    fn adding_a_and_b_equals_a_plus_b() {
        assert_eq!(polynomial_a() + polynomial_b(), polynomial_a_plus_b());
    }

    #[test]
    fn adding_a_and_a_plus_b_does_not_equal_b() {
        assert_ne!(polynomial_a() + polynomial_a_plus_b(), polynomial_b());
    }

    #[test]
    fn add_5_to_0_is_5() {
        let p1 = MultivariatePolynomial::new(1, &[FE::new(5)]);
        let p2 = MultivariatePolynomial::new(1, &[FE::new(0)]);
        assert_eq!(p1 + p2, MultivariatePolynomial::new(1, &[FE::new(5)]));
    }

    #[test]
    fn add_0_to_5_is_5() {
        let p1 = MultivariatePolynomial::new(1, &[FE::new(5)]);
        let p2 = MultivariatePolynomial::new(1, &[FE::new(0)]);
        assert_eq!(p2 + p1, MultivariatePolynomial::new(1, &[FE::new(5)]));
    }

    #[test]
    fn negating_0_returns_0() {
        let p1 = MultivariatePolynomial::new(1, &[FE::new(0)]);
        assert_eq!(-p1, MultivariatePolynomial::new(1, &[FE::new(0)]));
    }

    #[test]
    fn negating_a_is_equal_to_minus_a() {
        assert_eq!(-polynomial_a(), polynomial_minus_a());
    }

    #[test]
    fn negating_a_is_not_equal_to_a() {
        assert_ne!(-polynomial_a(), polynomial_a());
    }

    #[test]
    fn substracting_5_5_gives_0() {
        let p1 = MultivariatePolynomial::new(1, &[FE::new(5)]);
        let p2 = MultivariatePolynomial::new(1, &[FE::new(5)]);
        let p3 = MultivariatePolynomial::new(1, &[FE::new(0)]);
        assert_eq!(p1 - p2, p3);
    }

    #[test]
    fn substracting_b_and_a_equals_b_minus_a() {
        assert_eq!(polynomial_b() - polynomial_a(), polynomial_b_minus_a());
    }

    #[test]
    fn constructor_removes_zeros_at_the_end_of_polynomial() {
        let p1 = MultivariatePolynomial::new(1, &[FE::new(3), FE::new(4), FE::new(0)]);
        assert_eq!(p1.coefficients, &[FE::new(3), FE::new(4)]);
    }

    #[test]
    fn pad_with_zero_coefficients_returns_polynomials_with_zeros_until_matching_size() {
        let p1 = MultivariatePolynomial::new(1, &[FE::new(3), FE::new(4)]);
        let p2 = MultivariatePolynomial::new(1, &[FE::new(3)]);

        assert_eq!(p2.coefficients, &[FE::new(3)]);
        let (pp1, pp2) = MultivariatePolynomial::pad_with_zero_coefficients(&p1, &p2);
        assert_eq!(pp1, p1);
        assert_eq!(pp2.coefficients, &[FE::new(3), FE::new(0)]);
    }

    #[test]
    fn multiply_5_and_0_is_0() {
        let p1 = MultivariatePolynomial::new(1, &[FE::new(5)]);
        let p2 = MultivariatePolynomial::new(1, &[FE::new(0)]);
        assert_eq!(p1 * p2, MultivariatePolynomial::new(1, &[FE::new(0)]));
    }

    #[test]
    fn multiply_0_and_x_is_0() {
        let p1 = MultivariatePolynomial::new(1, &[FE::new(0)]);
        let p2 = MultivariatePolynomial::new(1, &[FE::new(0), FE::new(1)]);
        assert_eq!(p1 * p2, MultivariatePolynomial::new(1, &[FE::new(0)]));
    }

    #[test]
    fn multiply_2_by_3_is_6() {
        let p1 = MultivariatePolynomial::new(1, &[FE::new(2)]);
        let p2 = MultivariatePolynomial::new(1, &[FE::new(3)]);
        assert_eq!(p1 * p2, MultivariatePolynomial::new(1, &[FE::new(6)]));
    }

    #[test]
    fn multiply_2xx_3x_3_times_x_4() {
        let p1 = MultivariatePolynomial::new(1, &[FE::new(3), FE::new(3), FE::new(2)]);
        let p2 = MultivariatePolynomial::new(1, &[FE::new(4), FE::new(1)]);
        assert_eq!(
            p1 * p2,
            MultivariatePolynomial::new(1, &[FE::new(12), FE::new(15), FE::new(11), FE::new(2)])
        );
    }

    #[test]
    fn multiply_x_4_times_2xx_3x_3() {
        let p1 = MultivariatePolynomial::new(1, &[FE::new(3), FE::new(3), FE::new(2)]);
        let p2 = MultivariatePolynomial::new(1, &[FE::new(4), FE::new(1)]);
        assert_eq!(
            p2 * p1,
            MultivariatePolynomial::new(1, &[FE::new(12), FE::new(15), FE::new(11), FE::new(2)])
        );
    }

    #[test]
    fn evaluate_constant_polynomial_returns_constant() {
        let three = FE::new(3);
        let p = MultivariatePolynomial::new(1, &[three]);
        assert_eq!(p.evaluate(&[FE::new(10)]).unwrap(), three);
    }

    #[test]
    fn evaluate_slice() {
        let three = FE::new(3);
        let p = MultivariatePolynomial::new(1, &[three]);
        let ret = p.evaluate_slice(&[FE::new(10), FE::new(15)]);
        assert_eq!(ret, [three, three]);
    }

    #[test]
    fn create_degree_0_new_monomial() {
        assert_eq!(
            MultivariatePolynomial::new_monomial(FE::new(3), 0),
            MultivariatePolynomial::new(1, &[FE::new(3)])
        );
    }

    #[test]
    fn zero_poly_evals_0_in_3() {
        assert_eq!(
            MultivariatePolynomial::new_monomial(FE::new(0), 0)
                .evaluate(&[FE::new(3)])
                .unwrap(),
            FE::new(0)
        );
    }

    #[test]
    fn evaluate_degree_1_new_monomial() {
        let two = FE::new(2);
        let four = FE::new(4);
        let p = MultivariatePolynomial::new_monomial(two, 1);
        assert_eq!(p.evaluate(&[two]).unwrap(), four);
    }

    #[test]
    fn evaluate_degree_2_monomyal() {
        let two = FE::new(2);
        let eight = FE::new(8);
        let p = MultivariatePolynomial::new_monomial(two, 2);
        assert_eq!(p.evaluate(&[two]).unwrap(), eight);
    }

    #[test]
    fn evaluate_3_term_polynomial() {
        let p = MultivariatePolynomial::new(1, &[FE::new(3), -FE::new(2), FE::new(4)]);
        assert_eq!(p.evaluate(&[FE::new(2)]).unwrap(), FE::new(15));
    }

    #[test]
    fn break_in_parts() {
        // p = 3 X^3 + X^2 + 2X + 1
        let p = MultivariatePolynomial::new(1, &[FE::new(1), FE::new(2), FE::new(1), FE::new(3)]);
        let p0_expected = MultivariatePolynomial::new(1, &[FE::new(1), FE::new(1)]);
        let p1_expected = MultivariatePolynomial::new(1, &[FE::new(2), FE::new(3)]);
        let parts = p.split_n_ways(2);
        assert_eq!(parts.len(), 2);
        let p0 = &parts[0];
        let p1 = &parts[1];
        assert_eq!(p0, &p0_expected);
        assert_eq!(p1, &p1_expected);
    }

}
