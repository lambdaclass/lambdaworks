use crate::errors::TermError;
use crate::field::element::FieldElement;
use crate::field::traits::{IsField, IsPrimeField};
use core::fmt::Display;
use rayon::prelude::*;
use std::ops;

use super::traits::evaluation::InterpolateError;
use super::traits::polynomial::IsPolynomial;

/// Represents the polynomial c_0 + c_1 * X + c_2 * X^2 + ... + c_n * X^n
/// as a vector of coefficients `[c_0, c_1, ... , c_n]`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnivariatePolynomial<F: IsField> {
    pub coefficients: Vec<FieldElement<F>>,
}

impl<F: IsField> UnivariatePolynomial<F>
where
    <F as IsField>::BaseType: Send + Sync,
{
    pub fn new_monomial(coefficient: FieldElement<F>, degree: usize) -> Self {
        let mut coefficients = vec![FieldElement::zero(); degree];
        coefficients.push(coefficient);
        Self { coefficients }
    }

    /// Returns a polynomial that interpolates the points with x coordinates and y coordinates given by
    /// `xs` and `ys`.
    /// `xs` and `ys` must be the same length, and `xs` values should be unique. If not, panics.
    pub fn interpolate(
        xs: &[FieldElement<F>],
        ys: &[FieldElement<F>],
    ) -> Result<Self, InterpolateError> {
        // TODO: try to use the type system to avoid this assert
        if xs.len() != ys.len() {
            return Err(InterpolateError::UnequalLengths(xs.len(), ys.len()));
        }
        if xs.is_empty() {
            return Ok(Self::new(1, &[]));
        }

        let mut denominators = Vec::with_capacity(xs.len() * (xs.len() - 1) / 2);
        let mut indexes = Vec::with_capacity(xs.len());

        let mut idx = 0;

        //TODO: add rayon
        for (i, xi) in xs.iter().enumerate().skip(1) {
            indexes.push(idx);
            // TODO: add rayon
            for xj in xs.iter().take(i) {
                if xi == xj {
                    return Err(InterpolateError::NonUniqueXs);
                }
                denominators.push(xi - xj);
                idx += 1;
            }
        }

        FieldElement::inplace_batch_inverse(&mut denominators).unwrap();

        let mut result = Self::zero();

        // TODO: add rayon
        for (i, y) in ys.iter().enumerate() {
            let mut y_term = Self::new(1, &[y.clone()]);
            // TODO: add rayon
            for (j, x) in xs.iter().enumerate() {
                if i == j {
                    continue;
                }
                let denominator = if i > j {
                    denominators[indexes[i - 1] + j].clone()
                } else {
                    -&denominators[indexes[j - 1] + i]
                };
                let denominator_poly = Self::new(1, &[denominator]);
                let numerator = Self::new(1, &[-x, FieldElement::one()]);
                y_term = y_term.mul_with_ref(&(numerator * denominator_poly));
            }
            result = result + y_term;
        }
        Ok(result)
    }

    //URGENT_TODO: address the clone
    pub fn evaluate_slice(&self, input: &[FieldElement<F>]) -> Vec<FieldElement<F>> {
        input
            .into_par_iter()
            .map(|x| self.evaluate(&[x.clone()]).unwrap())
            .collect()
    }

    /// Computes quotient with `x - b` in place.
    fn ruffini_division_inplace(&mut self, b: &FieldElement<F>) {
        let mut c = FieldElement::zero();
        // TODO: can we add rayon to this without a race condition since we are swapping mem??
        self.coefficients.iter_mut().rev().map(|coeff| {
            *coeff = &*coeff + b * &c;
            core::mem::swap(coeff, &mut c);
        });
        self.coefficients.pop();
    }

    /// Computes quotient and remainder of polynomial division.
    ///
    /// Output: (quotient, remainder)
    pub fn long_division_with_remainder(self, dividend: &Self) -> (Self, Self) {
        if dividend.degree() > self.degree() {
            (Self::zero(), self)
        } else {
            let mut n = self;
            let mut q: Vec<FieldElement<F>> = vec![FieldElement::zero(); n.degree() + 1];
            let denominator = dividend.leading_coefficient().inv().unwrap();
            // TODO: can we add rayon???
            while !n.is_zero() && n.degree() >= dividend.degree() {
                let new_coefficient = n.leading_coefficient() * &denominator;
                q[n.degree() - dividend.degree()] = new_coefficient.clone();
                let d = dividend.mul_with_ref(&Self::new_monomial(
                    new_coefficient,
                    n.degree() - dividend.degree(),
                ));
                n = n - d;
            }
            (Self::new(1, &q), n)
        }
    }

    pub fn div_with_ref(self, dividend: &Self) -> Self {
        let (quotient, _remainder) = self.long_division_with_remainder(dividend);
        quotient
    }

    pub fn mul_with_ref(&self, factor: &Self) -> Self {
        let degree = self.degree() + factor.degree();
        let mut coefficients = vec![FieldElement::zero(); degree + 1];

        if self.coefficients.is_empty() || factor.coefficients.is_empty() {
            Self::new(1, &[FieldElement::zero()])
        } else {
            // TODO: add rayon
            for i in 0..=factor.degree() {
                for j in 0..=self.degree() {
                    coefficients[i + j] += &factor.coefficients[i] * &self.coefficients[j];
                }
            }
            Self::new(1, &coefficients)
        }
    }
}

impl<F: IsField> IsPolynomial<F, FieldElement<F>> for UnivariatePolynomial<F>
where
    <F as IsField>::BaseType: Send + Sync,
{
    type T = FieldElement<F>;

    /// Creates a new polynomial with the given coefficients
    //URGENT_TODO: address the unused var
    fn new(_num_vars: usize, terms: &[Self::T]) -> Self {
        // Removes trailing zero coefficients at the end
        let mut unpadded_coefficients = terms
            .iter()
            .rev()
            .skip_while(|x| **x == FieldElement::zero())
            .cloned()
            .collect::<Vec<FieldElement<F>>>();
        unpadded_coefficients.reverse();
        Self {
            coefficients: unpadded_coefficients,
        }
    }

    fn zero() -> Self {
        Self::new(1, &[])
    }

    fn evaluate(&self, x: &[FieldElement<F>]) -> Result<FieldElement<F>, TermError> {
        if x.len() != 1 {
            //TODO: add separate error
            return Err(TermError::InvalidEvaluationPoint);
        }
        //TODO: add rayon
        //URGENT_TODO: address the 0 index
        Ok(self
            .coefficients
            .iter()
            .rev()
            .fold(FieldElement::zero(), |acc, coeff| acc * x[0].to_owned() + coeff))
    }

    fn degree(&self) -> usize {
        if self.coefficients.is_empty() {
            0
        } else {
            self.coefficients.len() - 1
        }
    }

    fn leading_coefficient(&self) -> FieldElement<F> {
        if let Some(coefficient) = self.coefficients.last() {
            coefficient.clone()
        } else {
            FieldElement::zero()
        }
    }

    /// Returns coefficients of the polynomial as an array
    /// \[c_0, c_1, c_2, ..., c_n\]
    /// that represents the polynomial
    /// c_0 + c_1 * X + c_2 * X^2 + ... + c_n * X^n
    fn coeffs(&self) -> &[FieldElement<F>] {
        &self.coefficients
    }

    fn len(&self) -> usize {
        self.coeffs().len()
    }

    fn pad_with_zero_coefficients_to_length(pa: &mut Self, n: usize) {
        pa.coefficients.resize(n, FieldElement::zero());
    }

    /// Pads polynomial representations with minimum number of zeros to match lengths.
    fn pad_with_zero_coefficients(pa: &Self, pb: &Self) -> (Self, Self) {
        let mut pa = pa.clone();
        let mut pb = pb.clone();

        if pa.coefficients.len() > pb.coefficients.len() {
            Self::pad_with_zero_coefficients_to_length(&mut pb, pa.coefficients.len());
        } else {
            Self::pad_with_zero_coefficients_to_length(&mut pa, pb.coefficients.len());
        }
        (pa, pb)
    }

    fn scale(&self, factor: &FieldElement<F>) -> Self {
        //TODO: add rayon this will involve some thought to navigate the succcessors
        let scaled_coefficients = self
            .coefficients
            .iter()
            .zip(core::iter::successors(Some(FieldElement::one()), |x| {
                Some(x * factor)
            }))
            .map(|(coeff, power)| power * coeff)
            .collect();
        Self {
            coefficients: scaled_coefficients,
        }
    }

    fn scale_coeffs(&self, factor: &FieldElement<F>) -> Self {
        let scaled_coefficients = self
            .coefficients
            .par_iter()
            .map(|coeff| factor * coeff)
            .collect();
        Self {
            coefficients: scaled_coefficients,
        }
    }

    /// Returns a vector of polynomials [p₀, p₁, ..., p_{d-1}], where d is `number_of_parts`, such that `self` equals
    /// p₀(Xᵈ) + Xp₁(Xᵈ) + ... + X^(d-1)p_{d-1}(Xᵈ).
    ///
    /// Example: if d = 2 and `self` is 3 X^3 + X^2 + 2X + 1, then `poly.break_in_parts(2)`
    /// returns a vector with two polynomials `(p₀, p₁)`, where p₀ = X + 1 and p₁ = 3X + 2.
    fn split_n_ways(&self, n: usize) -> Vec<Self> {
        let terms = self.coeffs();
        let mut parts: Vec<Self> = Vec::with_capacity(n);
        for i in 0..n {
            let terms: Vec<_> = terms.par_iter().skip(i).step_by(n).cloned().collect();
            parts.push(Self::new(1, &terms));
        }
        parts
    }

    fn extend(&mut self, term: FieldElement<F>) {
        self.coefficients.extend_from_slice(&[term]);
    }

    //URGENT_TODO: address the 0 index
    fn evaluate_with(
        terms: &[FieldElement<F>],
        p: &[FieldElement<F>],
    ) -> Result<FieldElement<F>, crate::errors::TermError> {
        if p.len() != 1 {
            //TODO: add separate error
            return Err(TermError::InvalidEvaluationPoint);
        }
        Ok(terms
            .par_iter()
            .rev()
            .fold(|| FieldElement::zero(), |acc, coeff| acc * p[0].to_owned() + coeff)
            .reduce(|| FieldElement::zero(), |a, b| a + b))
    }

    fn eval_at_one(&self) -> FieldElement<F> {
        self.coefficients.clone().into_par_iter().sum()
    }

    fn eval_at_zero(&self) -> FieldElement<F> {
        self.coefficients[0].clone()
    }

    fn is_zero(&self) -> bool {
        self.is_empty()
            || self
                .coefficients
                .par_iter()
                .any(|c| *c == FieldElement::<F>::zero())
    }

    fn is_empty(&self) -> bool {
        self.coefficients.is_empty()
    }
}

pub fn combine<F>(
    poly_1: &UnivariatePolynomial<F>,
    poly_2: &UnivariatePolynomial<F>,
) -> UnivariatePolynomial<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    let max_degree: u64 = (poly_1.degree() * poly_2.degree()) as u64;

    let mut interpolation_points = vec![];
    for i in 0_u64..max_degree + 1 {
        interpolation_points.push(FieldElement::<F>::from(i));
    }

    //URGENT_TODO: address the clone
    let values: Vec<_> = interpolation_points
        .iter()
        .map(|value| {
            let intermediate_value = poly_2.evaluate(&[value.clone()]).unwrap();
            poly_1.evaluate(&[intermediate_value]).unwrap()
        })
        .collect();

    UnivariatePolynomial::interpolate(interpolation_points.as_slice(), values.as_slice())
        .expect("xs and ys have equal length and xs are unique")
}

impl<F: IsField> ops::Add<&UnivariatePolynomial<F>> for &UnivariatePolynomial<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = UnivariatePolynomial<F>;

    fn add(self, a_polynomial: &UnivariatePolynomial<F>) -> Self::Output {
        let (pa, pb) = UnivariatePolynomial::pad_with_zero_coefficients(self, a_polynomial);
        let iter_coeff_pa = pa.coefficients.iter();
        let iter_coeff_pb = pb.coefficients.iter();
        let new_coefficients = iter_coeff_pa.zip(iter_coeff_pb).map(|(x, y)| x + y);
        let new_coefficients_vec = new_coefficients.collect::<Vec<FieldElement<F>>>();
        UnivariatePolynomial::new(1, &new_coefficients_vec)
    }
}

impl<F: IsField> ops::Add<UnivariatePolynomial<F>> for UnivariatePolynomial<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = UnivariatePolynomial<F>;

    fn add(self, a_polynomial: UnivariatePolynomial<F>) -> UnivariatePolynomial<F> {
        &self + &a_polynomial
    }
}

impl<F: IsField> ops::Add<&UnivariatePolynomial<F>> for UnivariatePolynomial<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = UnivariatePolynomial<F>;

    fn add(self, a_polynomial: &UnivariatePolynomial<F>) -> UnivariatePolynomial<F> {
        &self + a_polynomial
    }
}

impl<F: IsField> ops::Add<UnivariatePolynomial<F>> for &UnivariatePolynomial<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = UnivariatePolynomial<F>;

    fn add(self, a_polynomial: UnivariatePolynomial<F>) -> UnivariatePolynomial<F> {
        self + &a_polynomial
    }
}

impl<F: IsField> ops::Neg for &UnivariatePolynomial<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = UnivariatePolynomial<F>;

    fn neg(self) -> UnivariatePolynomial<F> {
        let neg = self
            .coefficients
            .iter()
            .map(|x| -x)
            .collect::<Vec<FieldElement<F>>>();
        UnivariatePolynomial::new(1, &neg)
    }
}

impl<F: IsField> ops::Neg for UnivariatePolynomial<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = UnivariatePolynomial<F>;

    fn neg(self) -> UnivariatePolynomial<F> {
        -&self
    }
}

impl<F: IsField> ops::Sub<UnivariatePolynomial<F>> for UnivariatePolynomial<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = UnivariatePolynomial<F>;

    fn sub(self, substrahend: UnivariatePolynomial<F>) -> UnivariatePolynomial<F> {
        &self - &substrahend
    }
}

impl<F: IsField> ops::Sub<&UnivariatePolynomial<F>> for UnivariatePolynomial<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = UnivariatePolynomial<F>;

    fn sub(self, substrahend: &UnivariatePolynomial<F>) -> UnivariatePolynomial<F> {
        &self - substrahend
    }
}

impl<F: IsField> ops::Sub<UnivariatePolynomial<F>> for &UnivariatePolynomial<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = UnivariatePolynomial<F>;

    fn sub(self, substrahend: UnivariatePolynomial<F>) -> UnivariatePolynomial<F> {
        self - &substrahend
    }
}

impl<F: IsField> ops::Sub<&UnivariatePolynomial<F>> for &UnivariatePolynomial<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = UnivariatePolynomial<F>;

    fn sub(self, substrahend: &UnivariatePolynomial<F>) -> UnivariatePolynomial<F> {
        self + (-substrahend)
    }
}

impl<F: IsField> ops::Div<UnivariatePolynomial<F>> for UnivariatePolynomial<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = UnivariatePolynomial<F>;

    fn div(self, dividend: UnivariatePolynomial<F>) -> UnivariatePolynomial<F> {
        self.div_with_ref(&dividend)
    }
}

impl<F: IsField> ops::Mul<&UnivariatePolynomial<F>> for &UnivariatePolynomial<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = UnivariatePolynomial<F>;
    fn mul(self, factor: &UnivariatePolynomial<F>) -> UnivariatePolynomial<F> {
        self.mul_with_ref(factor)
    }
}

impl<F: IsField> ops::Mul<UnivariatePolynomial<F>> for UnivariatePolynomial<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = UnivariatePolynomial<F>;
    fn mul(self, factor: UnivariatePolynomial<F>) -> UnivariatePolynomial<F> {
        &self * &factor
    }
}

impl<F: IsField> ops::Mul<UnivariatePolynomial<F>> for &UnivariatePolynomial<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = UnivariatePolynomial<F>;
    fn mul(self, factor: UnivariatePolynomial<F>) -> UnivariatePolynomial<F> {
        self * &factor
    }
}

impl<F: IsField> ops::Mul<&UnivariatePolynomial<F>> for UnivariatePolynomial<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = UnivariatePolynomial<F>;
    fn mul(self, factor: &UnivariatePolynomial<F>) -> UnivariatePolynomial<F> {
        &self * factor
    }
}

/* Operations between UnivariatePolynomials and field elements */
/* Multiplication field element at left */
impl<F: IsField> ops::Mul<FieldElement<F>> for UnivariatePolynomial<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = UnivariatePolynomial<F>;

    fn mul(self, multiplicand: FieldElement<F>) -> UnivariatePolynomial<F> {
        let new_coefficients = self
            .coefficients
            .iter()
            .map(|value| value * &multiplicand)
            .collect();
        UnivariatePolynomial {
            coefficients: new_coefficients,
        }
    }
}

impl<F: IsField> ops::Mul<&FieldElement<F>> for &UnivariatePolynomial<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = UnivariatePolynomial<F>;

    fn mul(self, multiplicand: &FieldElement<F>) -> UnivariatePolynomial<F> {
        self.clone() * multiplicand.clone()
    }
}

impl<F: IsField> ops::Mul<FieldElement<F>> for &UnivariatePolynomial<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = UnivariatePolynomial<F>;

    fn mul(self, multiplicand: FieldElement<F>) -> UnivariatePolynomial<F> {
        self * &multiplicand
    }
}

impl<F: IsField> ops::Mul<&FieldElement<F>> for UnivariatePolynomial<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = UnivariatePolynomial<F>;

    fn mul(self, multiplicand: &FieldElement<F>) -> UnivariatePolynomial<F> {
        &self * multiplicand
    }
}

/* Multiplication field element at right */
impl<F: IsField> ops::Mul<&UnivariatePolynomial<F>> for &FieldElement<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = UnivariatePolynomial<F>;

    fn mul(self, multiplicand: &UnivariatePolynomial<F>) -> UnivariatePolynomial<F> {
        multiplicand * self
    }
}

impl<F: IsField> ops::Mul<UnivariatePolynomial<F>> for &FieldElement<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = UnivariatePolynomial<F>;

    fn mul(self, multiplicand: UnivariatePolynomial<F>) -> UnivariatePolynomial<F> {
        &multiplicand * self
    }
}

impl<F: IsField> ops::Mul<&UnivariatePolynomial<F>> for FieldElement<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = UnivariatePolynomial<F>;

    fn mul(self, multiplicand: &UnivariatePolynomial<F>) -> UnivariatePolynomial<F> {
        multiplicand * self
    }
}

impl<F: IsField> ops::Mul<UnivariatePolynomial<F>> for FieldElement<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = UnivariatePolynomial<F>;

    fn mul(self, multiplicand: UnivariatePolynomial<F>) -> UnivariatePolynomial<F> {
        &multiplicand * &self
    }
}

/* Addition field element at left */
impl<F: IsField> ops::Add<&FieldElement<F>> for &UnivariatePolynomial<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = UnivariatePolynomial<F>;

    fn add(self, other: &FieldElement<F>) -> UnivariatePolynomial<F> {
        UnivariatePolynomial::new_monomial(other.clone(), 0) + self
    }
}

impl<F: IsField> ops::Add<FieldElement<F>> for UnivariatePolynomial<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = UnivariatePolynomial<F>;

    fn add(self, other: FieldElement<F>) -> UnivariatePolynomial<F> {
        &self + &other
    }
}

impl<F: IsField> ops::Add<FieldElement<F>> for &UnivariatePolynomial<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = UnivariatePolynomial<F>;

    fn add(self, other: FieldElement<F>) -> UnivariatePolynomial<F> {
        self + &other
    }
}

impl<F: IsField> ops::Add<&FieldElement<F>> for UnivariatePolynomial<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = UnivariatePolynomial<F>;

    fn add(self, other: &FieldElement<F>) -> UnivariatePolynomial<F> {
        &self + other
    }
}

/* Addition field element at right */
impl<F: IsField> ops::Add<&UnivariatePolynomial<F>> for &FieldElement<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = UnivariatePolynomial<F>;

    fn add(self, other: &UnivariatePolynomial<F>) -> UnivariatePolynomial<F> {
        UnivariatePolynomial::new_monomial(self.clone(), 0) + other
    }
}

impl<F: IsField> ops::Add<UnivariatePolynomial<F>> for FieldElement<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = UnivariatePolynomial<F>;

    fn add(self, other: UnivariatePolynomial<F>) -> UnivariatePolynomial<F> {
        &self + &other
    }
}

impl<F: IsField> ops::Add<UnivariatePolynomial<F>> for &FieldElement<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = UnivariatePolynomial<F>;

    fn add(self, other: UnivariatePolynomial<F>) -> UnivariatePolynomial<F> {
        self + &other
    }
}

impl<F: IsField> ops::Add<&UnivariatePolynomial<F>> for FieldElement<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = UnivariatePolynomial<F>;

    fn add(self, other: &UnivariatePolynomial<F>) -> UnivariatePolynomial<F> {
        &self + other
    }
}

/* Substraction field element at left */
impl<F: IsField> ops::Sub<&FieldElement<F>> for &UnivariatePolynomial<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = UnivariatePolynomial<F>;

    fn sub(self, other: &FieldElement<F>) -> UnivariatePolynomial<F> {
        self - UnivariatePolynomial::new_monomial(other.clone(), 0)
    }
}

impl<F: IsField> ops::Sub<FieldElement<F>> for UnivariatePolynomial<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = UnivariatePolynomial<F>;

    fn sub(self, other: FieldElement<F>) -> UnivariatePolynomial<F> {
        &self - &other
    }
}

impl<F: IsField> ops::Sub<FieldElement<F>> for &UnivariatePolynomial<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = UnivariatePolynomial<F>;

    fn sub(self, other: FieldElement<F>) -> UnivariatePolynomial<F> {
        self - &other
    }
}

impl<F: IsField> ops::Sub<&FieldElement<F>> for UnivariatePolynomial<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = UnivariatePolynomial<F>;

    fn sub(self, other: &FieldElement<F>) -> UnivariatePolynomial<F> {
        &self - other
    }
}

/* Substraction field element at right */
impl<F: IsField> ops::Sub<&UnivariatePolynomial<F>> for &FieldElement<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = UnivariatePolynomial<F>;

    fn sub(self, other: &UnivariatePolynomial<F>) -> UnivariatePolynomial<F> {
        UnivariatePolynomial::new_monomial(self.clone(), 0) - other
    }
}

impl<F: IsField> ops::Sub<UnivariatePolynomial<F>> for FieldElement<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = UnivariatePolynomial<F>;

    fn sub(self, other: UnivariatePolynomial<F>) -> UnivariatePolynomial<F> {
        &self - &other
    }
}

impl<F: IsField> ops::Sub<UnivariatePolynomial<F>> for &FieldElement<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = UnivariatePolynomial<F>;

    fn sub(self, other: UnivariatePolynomial<F>) -> UnivariatePolynomial<F> {
        self - &other
    }
}

impl<F: IsField> ops::Sub<&UnivariatePolynomial<F>> for FieldElement<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = UnivariatePolynomial<F>;

    fn sub(self, other: &UnivariatePolynomial<F>) -> UnivariatePolynomial<F> {
        &self - other
    }
}

impl<F: IsField> ops::Index<usize> for UnivariatePolynomial<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    type Output = FieldElement<F>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.coefficients[index]
    }
}

impl<F: IsField> ops::IndexMut<usize> for UnivariatePolynomial<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.coefficients[index]
    }
}

impl<F: IsField + IsPrimeField> Display for UnivariatePolynomial<F>
where
    F: IsField,
    <F as IsField>::BaseType: Send + Sync,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut output: String = String::new();
        let monomials = self.coefficients.clone();

        for elem in &monomials[0..monomials.len() - 1] {
            output.push_str(&elem.representative().to_string()[0..]);
            output.push_str(" + ");
        }
        output.push_str(&monomials[monomials.len() - 1].representative().to_string());
        write!(f, "{}", output)
    }
}

//TODO: Question should we make coefficients private and implement Iterator, IterMut, IntoIterator and parallel counterparts

#[cfg(test)]
mod tests {
    use crate::field::fields::u64_prime_field::U64PrimeField;

    // Some of these tests work when the finite field has order greater than 2.
    use super::*;
    const ORDER: u64 = 23;
    type F = U64PrimeField<ORDER>;
    type FE = FieldElement<F>;

    fn polynomial_a() -> UnivariatePolynomial<F> {
        UnivariatePolynomial::new(1, &[FE::new(1), FE::new(2), FE::new(3)])
    }

    fn polynomial_minus_a() -> UnivariatePolynomial<F> {
        UnivariatePolynomial::new(
            1,
            &[FE::new(ORDER - 1), FE::new(ORDER - 2), FE::new(ORDER - 3)],
        )
    }

    fn polynomial_b() -> UnivariatePolynomial<F> {
        UnivariatePolynomial::new(1, &[FE::new(3), FE::new(4), FE::new(5)])
    }

    fn polynomial_a_plus_b() -> UnivariatePolynomial<F> {
        UnivariatePolynomial::new(1, &[FE::new(4), FE::new(6), FE::new(8)])
    }

    fn polynomial_b_minus_a() -> UnivariatePolynomial<F> {
        UnivariatePolynomial::new(1, &[FE::new(2), FE::new(2), FE::new(2)])
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
        let p1 = UnivariatePolynomial::new(1, &[FE::new(5)]);
        let p2 = UnivariatePolynomial::new(1, &[FE::new(0)]);
        assert_eq!(p1 + p2, UnivariatePolynomial::new(1, &[FE::new(5)]));
    }

    #[test]
    fn add_0_to_5_is_5() {
        let p1 = UnivariatePolynomial::new(1, &[FE::new(5)]);
        let p2 = UnivariatePolynomial::new(1, &[FE::new(0)]);
        assert_eq!(p2 + p1, UnivariatePolynomial::new(1, &[FE::new(5)]));
    }

    #[test]
    fn negating_0_returns_0() {
        let p1 = UnivariatePolynomial::new(1, &[FE::new(0)]);
        assert_eq!(-p1, UnivariatePolynomial::new(1, &[FE::new(0)]));
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
        let p1 = UnivariatePolynomial::new(1, &[FE::new(5)]);
        let p2 = UnivariatePolynomial::new(1, &[FE::new(5)]);
        let p3 = UnivariatePolynomial::new(1, &[FE::new(0)]);
        assert_eq!(p1 - p2, p3);
    }

    #[test]
    fn substracting_b_and_a_equals_b_minus_a() {
        assert_eq!(polynomial_b() - polynomial_a(), polynomial_b_minus_a());
    }

    #[test]
    fn constructor_removes_zeros_at_the_end_of_polynomial() {
        let p1 = UnivariatePolynomial::new(1, &[FE::new(3), FE::new(4), FE::new(0)]);
        assert_eq!(p1.coefficients, &[FE::new(3), FE::new(4)]);
    }

    #[test]
    fn pad_with_zero_coefficients_returns_polynomials_with_zeros_until_matching_size() {
        let p1 = UnivariatePolynomial::new(1, &[FE::new(3), FE::new(4)]);
        let p2 = UnivariatePolynomial::new(1, &[FE::new(3)]);

        assert_eq!(p2.coefficients, &[FE::new(3)]);
        let (pp1, pp2) = UnivariatePolynomial::pad_with_zero_coefficients(&p1, &p2);
        assert_eq!(pp1, p1);
        assert_eq!(pp2.coefficients, &[FE::new(3), FE::new(0)]);
    }

    #[test]
    fn multiply_5_and_0_is_0() {
        let p1 = UnivariatePolynomial::new(1, &[FE::new(5)]);
        let p2 = UnivariatePolynomial::new(1, &[FE::new(0)]);
        assert_eq!(p1 * p2, UnivariatePolynomial::new(1, &[FE::new(0)]));
    }

    #[test]
    fn multiply_0_and_x_is_0() {
        let p1 = UnivariatePolynomial::new(1, &[FE::new(0)]);
        let p2 = UnivariatePolynomial::new(1, &[FE::new(0), FE::new(1)]);
        assert_eq!(p1 * p2, UnivariatePolynomial::new(1, &[FE::new(0)]));
    }

    #[test]
    fn multiply_2_by_3_is_6() {
        let p1 = UnivariatePolynomial::new(1, &[FE::new(2)]);
        let p2 = UnivariatePolynomial::new(1, &[FE::new(3)]);
        assert_eq!(p1 * p2, UnivariatePolynomial::new(1, &[FE::new(6)]));
    }

    #[test]
    fn multiply_2xx_3x_3_times_x_4() {
        let p1 = UnivariatePolynomial::new(1, &[FE::new(3), FE::new(3), FE::new(2)]);
        let p2 = UnivariatePolynomial::new(1, &[FE::new(4), FE::new(1)]);
        assert_eq!(
            p1 * p2,
            UnivariatePolynomial::new(1, &[FE::new(12), FE::new(15), FE::new(11), FE::new(2)])
        );
    }

    #[test]
    fn multiply_x_4_times_2xx_3x_3() {
        let p1 = UnivariatePolynomial::new(1, &[FE::new(3), FE::new(3), FE::new(2)]);
        let p2 = UnivariatePolynomial::new(1, &[FE::new(4), FE::new(1)]);
        assert_eq!(
            p2 * p1,
            UnivariatePolynomial::new(1, &[FE::new(12), FE::new(15), FE::new(11), FE::new(2)])
        );
    }

    #[test]
    fn division_works() {
        let p1 = UnivariatePolynomial::new(1, &[FE::new(1), FE::new(3)]);
        let p2 = UnivariatePolynomial::new(1, &[FE::new(1), FE::new(3)]);
        let p3 = p1.mul_with_ref(&p2);
        assert_eq!(p3 / p2, p1);
    }

    #[test]
    fn division_by_zero_degree_polynomial_works() {
        let four = FE::new(4);
        let two = FE::new(2);
        let p1 = UnivariatePolynomial::new(1, &[four, four]);
        let p2 = UnivariatePolynomial::new(1, &[two]);
        assert_eq!(UnivariatePolynomial::new(1, &[two, two]), p1 / p2);
    }

    #[test]
    fn evaluate_constant_polynomial_returns_constant() {
        let three = FE::new(3);
        let p = UnivariatePolynomial::new(1, &[three]);
        assert_eq!(p.evaluate(&[FE::new(10)]).unwrap(), three);
    }

    #[test]
    fn evaluate_slice() {
        let three = FE::new(3);
        let p = UnivariatePolynomial::new(1, &[three]);
        let ret = p.evaluate_slice(&[FE::new(10), FE::new(15)]);
        assert_eq!(ret, [three, three]);
    }

    #[test]
    fn create_degree_0_new_monomial() {
        assert_eq!(
            UnivariatePolynomial::new_monomial(FE::new(3), 0),
            UnivariatePolynomial::new(1, &[FE::new(3)])
        );
    }

    #[test]
    fn zero_poly_evals_0_in_3() {
        assert_eq!(
            UnivariatePolynomial::new_monomial(FE::new(0), 0)
                .evaluate(&[FE::new(3)])
                .unwrap(),
            FE::new(0)
        );
    }

    #[test]
    fn evaluate_degree_1_new_monomial() {
        let two = FE::new(2);
        let four = FE::new(4);
        let p = UnivariatePolynomial::new_monomial(two, 1);
        assert_eq!(p.evaluate(&[two]).unwrap(), four);
    }

    #[test]
    fn evaluate_degree_2_monomyal() {
        let two = FE::new(2);
        let eight = FE::new(8);
        let p = UnivariatePolynomial::new_monomial(two, 2);
        assert_eq!(p.evaluate(&[two]).unwrap(), eight);
    }

    #[test]
    fn evaluate_3_term_polynomial() {
        let p = UnivariatePolynomial::new(1, &[FE::new(3), -FE::new(2), FE::new(4)]);
        assert_eq!(p.evaluate(&[FE::new(2)]).unwrap(), FE::new(15));
    }

    #[test]
    fn simple_interpolating_polynomial_by_hand_works() {
        let denominator = UnivariatePolynomial::new(1, &[FE::new(1) / (FE::new(2) - FE::new(4))]);
        let numerator = UnivariatePolynomial::new(1, &[-FE::new(4), FE::new(1)]);
        let interpolating = numerator * denominator;
        assert_eq!(
            (FE::new(2) - FE::new(4)) * (FE::new(1) / (FE::new(2) - FE::new(4))),
            FE::new(1)
        );
        assert_eq!(interpolating.evaluate(&[FE::new(2)]).unwrap(), FE::new(1));
        assert_eq!(interpolating.evaluate(&[FE::new(4)]).unwrap(), FE::new(0));
    }

    #[test]
    fn interpolate_x_2_y_3() {
        let p = UnivariatePolynomial::interpolate(&[FE::new(2)], &[FE::new(3)]).unwrap();
        assert_eq!(FE::new(3), p.evaluate(&[FE::new(2)]).unwrap());
    }

    #[test]
    fn interpolate_x_0_2_y_3_4() {
        let p =
            UnivariatePolynomial::interpolate(&[FE::new(0), FE::new(2)], &[FE::new(3), FE::new(4)])
                .unwrap();
        assert_eq!(FE::new(3), p.evaluate(&[FE::new(0)]).unwrap());
        assert_eq!(FE::new(4), p.evaluate(&[FE::new(2)]).unwrap());
    }

    #[test]
    fn interpolate_x_2_5_7_y_10_19_43() {
        let p = UnivariatePolynomial::interpolate(
            &[FE::new(2), FE::new(5), FE::new(7)],
            &[FE::new(10), FE::new(19), FE::new(43)],
        )
        .unwrap();

        assert_eq!(FE::new(10), p.evaluate(&[FE::new(2)]).unwrap());
        assert_eq!(FE::new(19), p.evaluate(&[FE::new(5)]).unwrap());
        assert_eq!(FE::new(43), p.evaluate(&[FE::new(7)]).unwrap());
    }

    #[test]
    fn interpolate_x_0_0_y_1_1() {
        let p =
            UnivariatePolynomial::interpolate(&[FE::new(0), FE::new(1)], &[FE::new(0), FE::new(1)])
                .unwrap();

        assert_eq!(FE::new(0), p.evaluate(&[FE::new(0)]).unwrap());
        assert_eq!(FE::new(1), p.evaluate(&[FE::new(1)]).unwrap());
    }

    #[test]
    fn interpolate_x_0_y_0() {
        let p = UnivariatePolynomial::interpolate(&[FE::new(0)], &[FE::new(0)]).unwrap();
        assert_eq!(FE::new(0), p.evaluate(&[FE::new(0)]).unwrap());
    }

    #[test]
    fn composition_works() {
        let p = UnivariatePolynomial::new(1, &[FE::new(0), FE::new(2)]);
        let q = UnivariatePolynomial::new(1, &[FE::new(0), FE::new(0), FE::new(1)]);
        assert_eq!(
            combine(&p, &q),
            UnivariatePolynomial::new(1, &[FE::new(0), FE::new(0), FE::new(2)])
        );
    }

    #[test]
    fn break_in_parts() {
        // p = 3 X^3 + X^2 + 2X + 1
        let p = UnivariatePolynomial::new(1, &[FE::new(1), FE::new(2), FE::new(1), FE::new(3)]);
        let p0_expected = UnivariatePolynomial::new(1, &[FE::new(1), FE::new(1)]);
        let p1_expected = UnivariatePolynomial::new(1, &[FE::new(2), FE::new(3)]);
        let parts = p.split_n_ways(2);
        assert_eq!(parts.len(), 2);
        let p0 = &parts[0];
        let p1 = &parts[1];
        assert_eq!(p0, &p0_expected);
        assert_eq!(p1, &p1_expected);
    }

    use proptest::prelude::*;
    proptest! {
        #[test]
        fn ruffini_equals_division(p in any::<Vec<u64>>(), b in any::<u64>()) {
            let p: Vec<_> = p.into_iter().map(FE::from).collect();
            let mut p = UnivariatePolynomial::new(1, &p);
            let b = FE::from(b);

            let p_ref = p.clone();
            let m = UnivariatePolynomial::new_monomial(FE::one(), 1) - b;

            p.ruffini_division_inplace(&b);
            prop_assert_eq!(p, p_ref / m);
        }
    }
}
