use super::field::element::FieldElement;
use crate::field::traits::IsField;
use std::ops;

/// Represents the polynomial c_0 + c_1 * X + c_2 * X^2 + ... + c_n * X^n
/// as a vector of coefficients `[c_0, c_1, ... , c_n]`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Polynomial<FE> {
    pub coefficients: Vec<FE>,
}

impl<F: IsField> Polynomial<FieldElement<F>> {
    /// Creates a new polynomial with the given coefficients
    pub fn new(coefficients: &[FieldElement<F>]) -> Self {
        // Removes trailing zero coefficients at the end
        let mut unpadded_coefficients = coefficients
            .iter()
            .rev()
            .skip_while(|x| **x == FieldElement::zero())
            .cloned()
            .collect::<Vec<FieldElement<F>>>();
        unpadded_coefficients.reverse();
        Polynomial {
            coefficients: unpadded_coefficients,
        }
    }

    pub fn new_monomial(coefficient: FieldElement<F>, degree: usize) -> Self {
        let mut coefficients = vec![FieldElement::zero(); degree];
        coefficients.push(coefficient);
        Self::new(&coefficients)
    }

    pub fn zero() -> Self {
        Self::new(&[])
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
            return Ok(Polynomial::new(&[]));
        }

        let mut denominators = Vec::with_capacity(xs.len() * (xs.len() - 1) / 2);
        let mut indexes = Vec::with_capacity(xs.len());

        let mut idx = 0;

        for (i, xi) in xs.iter().enumerate().skip(1) {
            indexes.push(idx);
            for xj in xs.iter().take(i) {
                if xi == xj {
                    return Err(InterpolateError::NonUniqueXs);
                }
                denominators.push(xi - xj);
                idx += 1;
            }
        }

        FieldElement::inplace_batch_inverse(&mut denominators).unwrap();

        let mut result = Polynomial::zero();

        for (i, y) in ys.iter().enumerate() {
            let mut y_term = Polynomial::new(&[y.clone()]);
            for (j, x) in xs.iter().enumerate() {
                if i == j {
                    continue;
                }
                let denominator = if i > j {
                    denominators[indexes[i - 1] + j].clone()
                } else {
                    -&denominators[indexes[j - 1] + i]
                };
                let denominator_poly = Polynomial::new(&[denominator]);
                let numerator = Polynomial::new(&[-x, FieldElement::one()]);
                y_term = y_term.mul_with_ref(&(numerator * denominator_poly));
            }
            result = result + y_term;
        }
        Ok(result)
    }

    pub fn evaluate(&self, x: &FieldElement<F>) -> FieldElement<F> {
        self.coefficients
            .iter()
            .rev()
            .fold(FieldElement::zero(), |acc, coeff| {
                acc * x.to_owned() + coeff
            })
    }

    pub fn evaluate_slice(&self, input: &[FieldElement<F>]) -> Vec<FieldElement<F>> {
        input.iter().map(|x| self.evaluate(x)).collect()
    }

    pub fn degree(&self) -> usize {
        if self.coefficients.is_empty() {
            0
        } else {
            self.coefficients.len() - 1
        }
    }

    pub fn leading_coefficient(&self) -> FieldElement<F> {
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
    pub fn coefficients(&self) -> &[FieldElement<F>] {
        &self.coefficients
    }

    pub fn coeff_len(&self) -> usize {
        self.coefficients().len()
    }

    pub fn pad_with_zero_coefficients_to_length(pa: &mut Self, n: usize) {
        pa.coefficients.resize(n, FieldElement::zero());
    }

    /// Pads polynomial representations with minimum number of zeros to match lengths.
    pub fn pad_with_zero_coefficients(pa: &Self, pb: &Self) -> (Self, Self) {
        let mut pa = pa.clone();
        let mut pb = pb.clone();

        if pa.coefficients.len() > pb.coefficients.len() {
            Self::pad_with_zero_coefficients_to_length(&mut pb, pa.coefficients.len());
        } else {
            Self::pad_with_zero_coefficients_to_length(&mut pa, pb.coefficients.len());
        }
        (pa, pb)
    }

    /// Computes quotient with `x - b` in place.
    pub fn ruffini_division_inplace(&mut self, b: &FieldElement<F>) {
        let mut c = FieldElement::zero();
        for coeff in self.coefficients.iter_mut().rev() {
            *coeff = &*coeff + b * &c;
            core::mem::swap(coeff, &mut c);
        }
        self.coefficients.pop();
    }

    /// Computes quotient and remainder of polynomial division.
    ///
    /// Output: (quotient, remainder)
    pub fn long_division_with_remainder(self, dividend: &Self) -> (Self, Self) {
        if dividend.degree() > self.degree() {
            (Polynomial::zero(), self)
        } else {
            let mut n = self;
            let mut q: Vec<FieldElement<F>> = vec![FieldElement::zero(); n.degree() + 1];
            let denominator = dividend.leading_coefficient().inv().unwrap();
            while n != Polynomial::zero() && n.degree() >= dividend.degree() {
                let new_coefficient = n.leading_coefficient() * &denominator;
                q[n.degree() - dividend.degree()] = new_coefficient.clone();
                let d = dividend.mul_with_ref(&Polynomial::new_monomial(
                    new_coefficient,
                    n.degree() - dividend.degree(),
                ));
                n = n - d;
            }
            (Polynomial::new(&q), n)
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
            Polynomial::new(&[FieldElement::zero()])
        } else {
            for i in 0..=factor.degree() {
                for j in 0..=self.degree() {
                    coefficients[i + j] += &factor.coefficients[i] * &self.coefficients[j];
                }
            }
            Polynomial::new(&coefficients)
        }
    }

    pub fn scale(&self, factor: &FieldElement<F>) -> Self {
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

    pub fn scale_coeffs(&self, factor: &FieldElement<F>) -> Self {
        let scaled_coefficients = self
            .coefficients
            .iter()
            .map(|coeff| factor * coeff)
            .collect();
        Self {
            coefficients: scaled_coefficients,
        }
    }

    /// For the given polynomial, returns a tuple `(even, odd)` of polynomials
    /// with the even and odd coefficients respectively.
    /// Note that `even` and `odd` ARE NOT actually even/odd polynomials themselves.
    ///
    /// Example: if poly = 3 X^3 + X^2 + 2X + 1, then
    /// `poly.even_odd_decomposition = (even, odd)` with
    /// `even` = X + 1 and `odd` = 3X + 1.
    ///
    /// In general, the decomposition satisfies the following:
    /// `poly(x)` = `even(x^2)` + X * `odd(x^2)`
    pub fn even_odd_decomposition(&self) -> (Self, Self) {
        let coef = self.coefficients();
        let even_coef: Vec<FieldElement<F>> = coef.iter().step_by(2).cloned().collect();

        // odd coeficients of poly are multiplied by beta
        let odd_coef: Vec<FieldElement<F>> = coef.iter().skip(1).step_by(2).cloned().collect();

        Polynomial::pad_with_zero_coefficients(
            &Polynomial::new(&even_coef),
            &Polynomial::new(&odd_coef),
        )
    }
}

pub fn compose<F>(
    poly_1: &Polynomial<FieldElement<F>>,
    poly_2: &Polynomial<FieldElement<F>>,
) -> Polynomial<FieldElement<F>>
where
    F: IsField,
{
    let max_degree: u64 = (poly_1.degree() * poly_2.degree()) as u64;

    let mut interpolation_points = vec![];
    for i in 0_u64..max_degree + 1 {
        interpolation_points.push(FieldElement::<F>::from(i));
    }

    let values: Vec<_> = interpolation_points
        .iter()
        .map(|value| {
            let intermediate_value = poly_2.evaluate(value);
            poly_1.evaluate(&intermediate_value)
        })
        .collect();

    Polynomial::interpolate(interpolation_points.as_slice(), values.as_slice())
        .expect("xs and ys have equal length and xs are unique")
}

impl<F: IsField> ops::Add<&Polynomial<FieldElement<F>>> for &Polynomial<FieldElement<F>> {
    type Output = Polynomial<FieldElement<F>>;

    fn add(self, a_polynomial: &Polynomial<FieldElement<F>>) -> Self::Output {
        let (pa, pb) = Polynomial::pad_with_zero_coefficients(self, a_polynomial);
        let iter_coeff_pa = pa.coefficients.iter();
        let iter_coeff_pb = pb.coefficients.iter();
        let new_coefficients = iter_coeff_pa.zip(iter_coeff_pb).map(|(x, y)| x + y);
        let new_coefficients_vec = new_coefficients.collect::<Vec<FieldElement<F>>>();
        Polynomial::new(&new_coefficients_vec)
    }
}

impl<F: IsField> ops::Add<Polynomial<FieldElement<F>>> for Polynomial<FieldElement<F>> {
    type Output = Polynomial<FieldElement<F>>;

    fn add(self, a_polynomial: Polynomial<FieldElement<F>>) -> Polynomial<FieldElement<F>> {
        &self + &a_polynomial
    }
}

impl<F: IsField> ops::Add<&Polynomial<FieldElement<F>>> for Polynomial<FieldElement<F>> {
    type Output = Polynomial<FieldElement<F>>;

    fn add(self, a_polynomial: &Polynomial<FieldElement<F>>) -> Polynomial<FieldElement<F>> {
        &self + a_polynomial
    }
}

impl<F: IsField> ops::Add<Polynomial<FieldElement<F>>> for &Polynomial<FieldElement<F>> {
    type Output = Polynomial<FieldElement<F>>;

    fn add(self, a_polynomial: Polynomial<FieldElement<F>>) -> Polynomial<FieldElement<F>> {
        self + &a_polynomial
    }
}

impl<F: IsField> ops::Neg for &Polynomial<FieldElement<F>> {
    type Output = Polynomial<FieldElement<F>>;

    fn neg(self) -> Polynomial<FieldElement<F>> {
        let neg = self
            .coefficients
            .iter()
            .map(|x| -x)
            .collect::<Vec<FieldElement<F>>>();
        Polynomial::new(&neg)
    }
}

impl<F: IsField> ops::Neg for Polynomial<FieldElement<F>> {
    type Output = Polynomial<FieldElement<F>>;

    fn neg(self) -> Polynomial<FieldElement<F>> {
        -&self
    }
}

impl<F: IsField> ops::Sub<Polynomial<FieldElement<F>>> for Polynomial<FieldElement<F>> {
    type Output = Polynomial<FieldElement<F>>;

    fn sub(self, substrahend: Polynomial<FieldElement<F>>) -> Polynomial<FieldElement<F>> {
        &self - &substrahend
    }
}

impl<F: IsField> ops::Sub<&Polynomial<FieldElement<F>>> for Polynomial<FieldElement<F>> {
    type Output = Polynomial<FieldElement<F>>;

    fn sub(self, substrahend: &Polynomial<FieldElement<F>>) -> Polynomial<FieldElement<F>> {
        &self - substrahend
    }
}

impl<F: IsField> ops::Sub<Polynomial<FieldElement<F>>> for &Polynomial<FieldElement<F>> {
    type Output = Polynomial<FieldElement<F>>;

    fn sub(self, substrahend: Polynomial<FieldElement<F>>) -> Polynomial<FieldElement<F>> {
        self - &substrahend
    }
}

impl<F: IsField> ops::Sub<&Polynomial<FieldElement<F>>> for &Polynomial<FieldElement<F>> {
    type Output = Polynomial<FieldElement<F>>;

    fn sub(self, substrahend: &Polynomial<FieldElement<F>>) -> Polynomial<FieldElement<F>> {
        self + (-substrahend)
    }
}

impl<F: IsField> ops::Div<Polynomial<FieldElement<F>>> for Polynomial<FieldElement<F>> {
    type Output = Polynomial<FieldElement<F>>;

    fn div(self, dividend: Polynomial<FieldElement<F>>) -> Polynomial<FieldElement<F>> {
        self.div_with_ref(&dividend)
    }
}

impl<F: IsField> ops::Mul<&Polynomial<FieldElement<F>>> for &Polynomial<FieldElement<F>> {
    type Output = Polynomial<FieldElement<F>>;
    fn mul(self, factor: &Polynomial<FieldElement<F>>) -> Polynomial<FieldElement<F>> {
        self.mul_with_ref(factor)
    }
}

impl<F: IsField> ops::Mul<Polynomial<FieldElement<F>>> for Polynomial<FieldElement<F>> {
    type Output = Polynomial<FieldElement<F>>;
    fn mul(self, factor: Polynomial<FieldElement<F>>) -> Polynomial<FieldElement<F>> {
        &self * &factor
    }
}

impl<F: IsField> ops::Mul<Polynomial<FieldElement<F>>> for &Polynomial<FieldElement<F>> {
    type Output = Polynomial<FieldElement<F>>;
    fn mul(self, factor: Polynomial<FieldElement<F>>) -> Polynomial<FieldElement<F>> {
        self * &factor
    }
}

impl<F: IsField> ops::Mul<&Polynomial<FieldElement<F>>> for Polynomial<FieldElement<F>> {
    type Output = Polynomial<FieldElement<F>>;
    fn mul(self, factor: &Polynomial<FieldElement<F>>) -> Polynomial<FieldElement<F>> {
        &self * factor
    }
}

/* Operations between Polynomials and field elements */
/* Multiplication field element at left */
impl<F: IsField> ops::Mul<FieldElement<F>> for Polynomial<FieldElement<F>> {
    type Output = Polynomial<FieldElement<F>>;

    fn mul(self, multiplicand: FieldElement<F>) -> Polynomial<FieldElement<F>> {
        let new_coefficients = self
            .coefficients
            .iter()
            .map(|value| value * &multiplicand)
            .collect();
        Polynomial {
            coefficients: new_coefficients,
        }
    }
}

impl<F: IsField> ops::Mul<&FieldElement<F>> for &Polynomial<FieldElement<F>> {
    type Output = Polynomial<FieldElement<F>>;

    fn mul(self, multiplicand: &FieldElement<F>) -> Polynomial<FieldElement<F>> {
        self.clone() * multiplicand.clone()
    }
}

impl<F: IsField> ops::Mul<FieldElement<F>> for &Polynomial<FieldElement<F>> {
    type Output = Polynomial<FieldElement<F>>;

    fn mul(self, multiplicand: FieldElement<F>) -> Polynomial<FieldElement<F>> {
        self * &multiplicand
    }
}

impl<F: IsField> ops::Mul<&FieldElement<F>> for Polynomial<FieldElement<F>> {
    type Output = Polynomial<FieldElement<F>>;

    fn mul(self, multiplicand: &FieldElement<F>) -> Polynomial<FieldElement<F>> {
        &self * multiplicand
    }
}

/* Multiplication field element at right */
impl<F: IsField> ops::Mul<&Polynomial<FieldElement<F>>> for &FieldElement<F> {
    type Output = Polynomial<FieldElement<F>>;

    fn mul(self, multiplicand: &Polynomial<FieldElement<F>>) -> Polynomial<FieldElement<F>> {
        multiplicand * self
    }
}

impl<F: IsField> ops::Mul<Polynomial<FieldElement<F>>> for &FieldElement<F> {
    type Output = Polynomial<FieldElement<F>>;

    fn mul(self, multiplicand: Polynomial<FieldElement<F>>) -> Polynomial<FieldElement<F>> {
        &multiplicand * self
    }
}

impl<F: IsField> ops::Mul<&Polynomial<FieldElement<F>>> for FieldElement<F> {
    type Output = Polynomial<FieldElement<F>>;

    fn mul(self, multiplicand: &Polynomial<FieldElement<F>>) -> Polynomial<FieldElement<F>> {
        multiplicand * self
    }
}

impl<F: IsField> ops::Mul<Polynomial<FieldElement<F>>> for FieldElement<F> {
    type Output = Polynomial<FieldElement<F>>;

    fn mul(self, multiplicand: Polynomial<FieldElement<F>>) -> Polynomial<FieldElement<F>> {
        &multiplicand * &self
    }
}

/* Addition field element at left */
impl<F: IsField> ops::Add<&FieldElement<F>> for &Polynomial<FieldElement<F>> {
    type Output = Polynomial<FieldElement<F>>;

    fn add(self, other: &FieldElement<F>) -> Polynomial<FieldElement<F>> {
        Polynomial::new_monomial(other.clone(), 0) + self
    }
}

impl<F: IsField> ops::Add<FieldElement<F>> for Polynomial<FieldElement<F>> {
    type Output = Polynomial<FieldElement<F>>;

    fn add(self, other: FieldElement<F>) -> Polynomial<FieldElement<F>> {
        &self + &other
    }
}

impl<F: IsField> ops::Add<FieldElement<F>> for &Polynomial<FieldElement<F>> {
    type Output = Polynomial<FieldElement<F>>;

    fn add(self, other: FieldElement<F>) -> Polynomial<FieldElement<F>> {
        self + &other
    }
}

impl<F: IsField> ops::Add<&FieldElement<F>> for Polynomial<FieldElement<F>> {
    type Output = Polynomial<FieldElement<F>>;

    fn add(self, other: &FieldElement<F>) -> Polynomial<FieldElement<F>> {
        &self + other
    }
}

/* Addition field element at right */
impl<F: IsField> ops::Add<&Polynomial<FieldElement<F>>> for &FieldElement<F> {
    type Output = Polynomial<FieldElement<F>>;

    fn add(self, other: &Polynomial<FieldElement<F>>) -> Polynomial<FieldElement<F>> {
        Polynomial::new_monomial(self.clone(), 0) + other
    }
}

impl<F: IsField> ops::Add<Polynomial<FieldElement<F>>> for FieldElement<F> {
    type Output = Polynomial<FieldElement<F>>;

    fn add(self, other: Polynomial<FieldElement<F>>) -> Polynomial<FieldElement<F>> {
        &self + &other
    }
}

impl<F: IsField> ops::Add<Polynomial<FieldElement<F>>> for &FieldElement<F> {
    type Output = Polynomial<FieldElement<F>>;

    fn add(self, other: Polynomial<FieldElement<F>>) -> Polynomial<FieldElement<F>> {
        self + &other
    }
}

impl<F: IsField> ops::Add<&Polynomial<FieldElement<F>>> for FieldElement<F> {
    type Output = Polynomial<FieldElement<F>>;

    fn add(self, other: &Polynomial<FieldElement<F>>) -> Polynomial<FieldElement<F>> {
        &self + other
    }
}

/* Substraction field element at left */
impl<F: IsField> ops::Sub<&FieldElement<F>> for &Polynomial<FieldElement<F>> {
    type Output = Polynomial<FieldElement<F>>;

    fn sub(self, other: &FieldElement<F>) -> Polynomial<FieldElement<F>> {
        self - Polynomial::new_monomial(other.clone(), 0)
    }
}

impl<F: IsField> ops::Sub<FieldElement<F>> for Polynomial<FieldElement<F>> {
    type Output = Polynomial<FieldElement<F>>;

    fn sub(self, other: FieldElement<F>) -> Polynomial<FieldElement<F>> {
        &self - &other
    }
}

impl<F: IsField> ops::Sub<FieldElement<F>> for &Polynomial<FieldElement<F>> {
    type Output = Polynomial<FieldElement<F>>;

    fn sub(self, other: FieldElement<F>) -> Polynomial<FieldElement<F>> {
        self - &other
    }
}

impl<F: IsField> ops::Sub<&FieldElement<F>> for Polynomial<FieldElement<F>> {
    type Output = Polynomial<FieldElement<F>>;

    fn sub(self, other: &FieldElement<F>) -> Polynomial<FieldElement<F>> {
        &self - other
    }
}

/* Substraction field element at right */
impl<F: IsField> ops::Sub<&Polynomial<FieldElement<F>>> for &FieldElement<F> {
    type Output = Polynomial<FieldElement<F>>;

    fn sub(self, other: &Polynomial<FieldElement<F>>) -> Polynomial<FieldElement<F>> {
        Polynomial::new_monomial(self.clone(), 0) - other
    }
}

impl<F: IsField> ops::Sub<Polynomial<FieldElement<F>>> for FieldElement<F> {
    type Output = Polynomial<FieldElement<F>>;

    fn sub(self, other: Polynomial<FieldElement<F>>) -> Polynomial<FieldElement<F>> {
        &self - &other
    }
}

impl<F: IsField> ops::Sub<Polynomial<FieldElement<F>>> for &FieldElement<F> {
    type Output = Polynomial<FieldElement<F>>;

    fn sub(self, other: Polynomial<FieldElement<F>>) -> Polynomial<FieldElement<F>> {
        self - &other
    }
}

impl<F: IsField> ops::Sub<&Polynomial<FieldElement<F>>> for FieldElement<F> {
    type Output = Polynomial<FieldElement<F>>;

    fn sub(self, other: &Polynomial<FieldElement<F>>) -> Polynomial<FieldElement<F>> {
        &self - other
    }
}

use thiserror::Error;

#[derive(Debug, Error)]
pub enum InterpolateError {
    #[error("xs and ys must be the same length. Got: {0} != {1}")]
    UnequalLengths(usize, usize),
    #[error("xs values should be unique.")]
    NonUniqueXs,
}

#[cfg(test)]
mod tests {
    use crate::field::fields::u64_prime_field::U64PrimeField;

    // Some of these tests work when the finite field has order greater than 2.
    use super::*;
    const ORDER: u64 = 23;
    type F = U64PrimeField<ORDER>;
    type FE = FieldElement<F>;

    fn polynomial_a() -> Polynomial<FE> {
        Polynomial::new(&[FE::new(1), FE::new(2), FE::new(3)])
    }

    fn polynomial_minus_a() -> Polynomial<FE> {
        Polynomial::new(&[FE::new(ORDER - 1), FE::new(ORDER - 2), FE::new(ORDER - 3)])
    }

    fn polynomial_b() -> Polynomial<FE> {
        Polynomial::new(&[FE::new(3), FE::new(4), FE::new(5)])
    }

    fn polynomial_a_plus_b() -> Polynomial<FE> {
        Polynomial::new(&[FE::new(4), FE::new(6), FE::new(8)])
    }

    fn polynomial_b_minus_a() -> Polynomial<FE> {
        Polynomial::new(&[FE::new(2), FE::new(2), FE::new(2)])
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
        let p1 = Polynomial::new(&[FE::new(5)]);
        let p2 = Polynomial::new(&[FE::new(0)]);
        assert_eq!(p1 + p2, Polynomial::new(&[FE::new(5)]));
    }

    #[test]
    fn add_0_to_5_is_5() {
        let p1 = Polynomial::new(&[FE::new(5)]);
        let p2 = Polynomial::new(&[FE::new(0)]);
        assert_eq!(p2 + p1, Polynomial::new(&[FE::new(5)]));
    }

    #[test]
    fn negating_0_returns_0() {
        let p1 = Polynomial::new(&[FE::new(0)]);
        assert_eq!(-p1, Polynomial::new(&[FE::new(0)]));
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
        let p1 = Polynomial::new(&[FE::new(5)]);
        let p2 = Polynomial::new(&[FE::new(5)]);
        let p3 = Polynomial::new(&[FE::new(0)]);
        assert_eq!(p1 - p2, p3);
    }

    #[test]
    fn substracting_b_and_a_equals_b_minus_a() {
        assert_eq!(polynomial_b() - polynomial_a(), polynomial_b_minus_a());
    }

    #[test]
    fn constructor_removes_zeros_at_the_end_of_polynomial() {
        let p1 = Polynomial::new(&[FE::new(3), FE::new(4), FE::new(0)]);
        assert_eq!(p1.coefficients, &[FE::new(3), FE::new(4)]);
    }

    #[test]
    fn pad_with_zero_coefficients_returns_polynomials_with_zeros_until_matching_size() {
        let p1 = Polynomial::new(&[FE::new(3), FE::new(4)]);
        let p2 = Polynomial::new(&[FE::new(3)]);

        assert_eq!(p2.coefficients, &[FE::new(3)]);
        let (pp1, pp2) = Polynomial::pad_with_zero_coefficients(&p1, &p2);
        assert_eq!(pp1, p1);
        assert_eq!(pp2.coefficients, &[FE::new(3), FE::new(0)]);
    }

    #[test]
    fn multiply_5_and_0_is_0() {
        let p1 = Polynomial::new(&[FE::new(5)]);
        let p2 = Polynomial::new(&[FE::new(0)]);
        assert_eq!(p1 * p2, Polynomial::new(&[FE::new(0)]));
    }

    #[test]
    fn multiply_0_and_x_is_0() {
        let p1 = Polynomial::new(&[FE::new(0)]);
        let p2 = Polynomial::new(&[FE::new(0), FE::new(1)]);
        assert_eq!(p1 * p2, Polynomial::new(&[FE::new(0)]));
    }

    #[test]
    fn multiply_2_by_3_is_6() {
        let p1 = Polynomial::new(&[FE::new(2)]);
        let p2 = Polynomial::new(&[FE::new(3)]);
        assert_eq!(p1 * p2, Polynomial::new(&[FE::new(6)]));
    }

    #[test]
    fn multiply_2xx_3x_3_times_x_4() {
        let p1 = Polynomial::new(&[FE::new(3), FE::new(3), FE::new(2)]);
        let p2 = Polynomial::new(&[FE::new(4), FE::new(1)]);
        assert_eq!(
            p1 * p2,
            Polynomial::new(&[FE::new(12), FE::new(15), FE::new(11), FE::new(2)])
        );
    }

    #[test]
    fn multiply_x_4_times_2xx_3x_3() {
        let p1 = Polynomial::new(&[FE::new(3), FE::new(3), FE::new(2)]);
        let p2 = Polynomial::new(&[FE::new(4), FE::new(1)]);
        assert_eq!(
            p2 * p1,
            Polynomial::new(&[FE::new(12), FE::new(15), FE::new(11), FE::new(2)])
        );
    }

    #[test]
    fn division_works() {
        let p1 = Polynomial::new(&[FE::new(1), FE::new(3)]);
        let p2 = Polynomial::new(&[FE::new(1), FE::new(3)]);
        let p3 = p1.mul_with_ref(&p2);
        assert_eq!(p3 / p2, p1);
    }

    #[test]
    fn division_by_zero_degree_polynomial_works() {
        let four = FE::new(4);
        let two = FE::new(2);
        let p1 = Polynomial::new(&[four, four]);
        let p2 = Polynomial::new(&[two]);
        assert_eq!(Polynomial::new(&[two, two]), p1 / p2);
    }

    #[test]
    fn evaluate_constant_polynomial_returns_constant() {
        let three = FE::new(3);
        let p = Polynomial::new(&[three]);
        assert_eq!(p.evaluate(&FE::new(10)), three);
    }

    #[test]
    fn evaluate_slice() {
        let three = FE::new(3);
        let p = Polynomial::new(&[three]);
        let ret = p.evaluate_slice(&[FE::new(10), FE::new(15)]);
        assert_eq!(ret, [three, three]);
    }

    #[test]
    fn create_degree_0_new_monomial() {
        assert_eq!(
            Polynomial::new_monomial(FE::new(3), 0),
            Polynomial::new(&[FE::new(3)])
        );
    }

    #[test]
    fn zero_poly_evals_0_in_3() {
        assert_eq!(
            Polynomial::new_monomial(FE::new(0), 0).evaluate(&FE::new(3)),
            FE::new(0)
        );
    }

    #[test]
    fn evaluate_degree_1_new_monomial() {
        let two = FE::new(2);
        let four = FE::new(4);
        let p = Polynomial::new_monomial(two, 1);
        assert_eq!(p.evaluate(&two), four);
    }

    #[test]
    fn evaluate_degree_2_monomyal() {
        let two = FE::new(2);
        let eight = FE::new(8);
        let p = Polynomial::new_monomial(two, 2);
        assert_eq!(p.evaluate(&two), eight);
    }

    #[test]
    fn evaluate_3_term_polynomial() {
        let p = Polynomial::new(&[FE::new(3), -FE::new(2), FE::new(4)]);
        assert_eq!(p.evaluate(&FE::new(2)), FE::new(15));
    }

    #[test]
    fn simple_interpolating_polynomial_by_hand_works() {
        let denominator = Polynomial::new(&[FE::new(1) / (FE::new(2) - FE::new(4))]);
        let numerator = Polynomial::new(&[-FE::new(4), FE::new(1)]);
        let interpolating = numerator * denominator;
        assert_eq!(
            (FE::new(2) - FE::new(4)) * (FE::new(1) / (FE::new(2) - FE::new(4))),
            FE::new(1)
        );
        assert_eq!(interpolating.evaluate(&FE::new(2)), FE::new(1));
        assert_eq!(interpolating.evaluate(&FE::new(4)), FE::new(0));
    }

    #[test]
    fn interpolate_x_2_y_3() {
        let p = Polynomial::interpolate(&[FE::new(2)], &[FE::new(3)]).unwrap();
        assert_eq!(FE::new(3), p.evaluate(&FE::new(2)));
    }

    #[test]
    fn interpolate_x_0_2_y_3_4() {
        let p =
            Polynomial::interpolate(&[FE::new(0), FE::new(2)], &[FE::new(3), FE::new(4)]).unwrap();
        assert_eq!(FE::new(3), p.evaluate(&FE::new(0)));
        assert_eq!(FE::new(4), p.evaluate(&FE::new(2)));
    }

    #[test]
    fn interpolate_x_2_5_7_y_10_19_43() {
        let p = Polynomial::interpolate(
            &[FE::new(2), FE::new(5), FE::new(7)],
            &[FE::new(10), FE::new(19), FE::new(43)],
        )
        .unwrap();

        assert_eq!(FE::new(10), p.evaluate(&FE::new(2)));
        assert_eq!(FE::new(19), p.evaluate(&FE::new(5)));
        assert_eq!(FE::new(43), p.evaluate(&FE::new(7)));
    }

    #[test]
    fn interpolate_x_0_0_y_1_1() {
        let p =
            Polynomial::interpolate(&[FE::new(0), FE::new(1)], &[FE::new(0), FE::new(1)]).unwrap();

        assert_eq!(FE::new(0), p.evaluate(&FE::new(0)));
        assert_eq!(FE::new(1), p.evaluate(&FE::new(1)));
    }

    #[test]
    fn interpolate_x_0_y_0() {
        let p = Polynomial::interpolate(&[FE::new(0)], &[FE::new(0)]).unwrap();
        assert_eq!(FE::new(0), p.evaluate(&FE::new(0)));
    }

    #[test]
    fn composition_works() {
        let p = Polynomial::new(&[FE::new(0), FE::new(2)]);
        let q = Polynomial::new(&[FE::new(0), FE::new(0), FE::new(1)]);
        assert_eq!(
            compose(&p, &q),
            Polynomial::new(&[FE::new(0), FE::new(0), FE::new(2)])
        );
    }

    use proptest::prelude::*;
    proptest! {
        #[test]
        fn ruffini_equals_division(p in any::<Vec<u64>>(), b in any::<u64>()) {
            let p: Vec<_> = p.into_iter().map(FE::from).collect();
            let mut p = Polynomial::new(&p);
            let b = FE::from(b);

            let p_ref = p.clone();
            let m = Polynomial::new_monomial(FE::one(), 1) - b;

            p.ruffini_division_inplace(&b);
            prop_assert_eq!(p, p_ref / m);
        }
    }
}
