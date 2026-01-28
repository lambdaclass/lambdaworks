use super::field::element::FieldElement;
use crate::field::traits::{IsField, IsPrimeField, IsSubFieldOf};
use alloc::string::{String, ToString};
use alloc::{borrow::ToOwned, format, vec, vec::Vec};
use core::{fmt::Display, ops, slice};
use core::ops::{AddAssign, SubAssign, MulAssign};
pub mod dense_multilinear_poly;
mod error;
pub use error::PolynomialError;
pub mod sparse_multilinear_poly;

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

    /// Creates a new monomial term coefficient*x^degree
    pub fn new_monomial(coefficient: FieldElement<F>, degree: usize) -> Self {
        let mut coefficients = vec![FieldElement::zero(); degree];
        coefficients.push(coefficient);
        Self::new(&coefficients)
    }

    /// Creates the null polynomial
    pub fn zero() -> Self {
        Self::new(&[])
    }

    /// Returns a polynomial that interpolates the points with x coordinates and y coordinates given by
    /// `xs` and `ys`.
    /// `xs` and `ys` must be the same length, and `xs` values should be unique. If not, panics.
    /// In short, it finds `P(x)` such that `P(xs[i]) = ys[i]`
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
            let mut y_term = Polynomial::new(slice::from_ref(y));
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

    /// Evaluates a polynomial P(t) at a point x, using Horner's algorithm
    /// Returns y = P(x)
    pub fn evaluate<E>(&self, x: &FieldElement<E>) -> FieldElement<E>
    where
        E: IsField,
        F: IsSubFieldOf<E>,
    {
        self.coefficients
            .iter()
            .rev()
            .fold(FieldElement::zero(), |acc, coeff| {
                coeff + acc * x.to_owned()
            })
    }

    /// Evaluates a polynomial `P(t)` at a slice of points `x`
    /// Returns a vector `y` such that `y[i] = P(input[i])`
    pub fn evaluate_slice(&self, input: &[FieldElement<F>]) -> Vec<FieldElement<F>> {
        input.iter().map(|x| self.evaluate(x)).collect()
    }

    /// Returns the degree of a polynomial, which corresponds to the highest power of x^d
    /// with non-zero coefficient
    pub fn degree(&self) -> usize {
        if self.coefficients.is_empty() {
            0
        } else {
            self.coefficients.len() - 1
        }
    }

    /// Returns the coefficient accompanying x^degree
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

    /// Returns the length of the vector of coefficients
    pub fn coeff_len(&self) -> usize {
        self.coefficients().len()
    }

    /// Returns true if the polynomial is zero (all coefficients are zero or empty).
    ///
    /// This is more robust than comparing with `Polynomial::zero()` because
    /// some operations (like `scale_coeffs` with zero) may produce polynomials
    /// with non-empty but all-zero coefficient vectors.
    pub fn is_zero(&self) -> bool {
        self.coefficients.iter().all(|c| *c == FieldElement::zero())
    }

    /// Returns the derivative of the polynomial with respect to x.
    pub fn differentiate(&self) -> Self {
        let degree = self.degree();
        if degree == 0 {
            return Polynomial::zero();
        }
        let mut derivative = Vec::with_capacity(degree);
        for (i, coeff) in self.coefficients().iter().enumerate().skip(1) {
            derivative.push(FieldElement::<F>::from(i as u64) * coeff);
        }
        Polynomial::new(&derivative)
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

    /// Computes the quotient of the division of P(x) with x - b using Ruffini's rule
    pub fn ruffini_division<L>(&self, b: &FieldElement<L>) -> Polynomial<FieldElement<L>>
    where
        L: IsField,
        F: IsSubFieldOf<L>,
    {
        if let Some(c) = self.coefficients.last() {
            let mut c = c.clone().to_extension();
            let mut coefficients = Vec::with_capacity(self.degree());
            for coeff in self.coefficients.iter().rev().skip(1) {
                coefficients.push(c.clone());
                c = coeff + c * b;
            }
            coefficients = coefficients.into_iter().rev().collect();
            Polynomial::new(&coefficients)
        } else {
            Polynomial::zero()
        }
    }

    /// Computes quotient and remainder of polynomial division.
    ///
    /// Output: `Ok((quotient, remainder))`
    ///
    /// # Errors
    /// Returns `PolynomialError::DivisionByZero` if the divisor is the zero polynomial.
    pub fn long_division_with_remainder(
        self,
        divisor: &Self,
    ) -> Result<(Self, Self), PolynomialError> {
        if divisor.is_zero() {
            return Err(PolynomialError::DivisionByZero);
        }
        if divisor.degree() > self.degree() {
            Ok((Polynomial::zero(), self))
        } else {
            let mut n = self;
            let mut q: Vec<FieldElement<F>> = vec![FieldElement::zero(); n.degree() + 1];
            let denominator = divisor
                .leading_coefficient()
                .inv()
                .expect("Leading coefficient should be non-zero for non-zero polynomial");
            while !n.is_zero() && n.degree() >= divisor.degree() {
                let new_coefficient = n.leading_coefficient() * &denominator;
                q[n.degree() - divisor.degree()] = new_coefficient.clone();
                let d = divisor.mul_with_ref(&Polynomial::new_monomial(
                    new_coefficient,
                    n.degree() - divisor.degree(),
                ));
                n = n - d;
            }
            Ok((Polynomial::new(&q), n))
        }
    }

    /// Extended Euclidean Algorithm for polynomials.
    ///
    /// This method computes the extended greatest common divisor (GCD) of two polynomials `self` and `y`.
    /// It returns a tuple of three elements: `(a, b, g)` such that `a * self + b * y = g`, where `g` is the
    /// greatest common divisor of `self` and `y`.
    ///
    /// # Errors
    /// Returns `PolynomialError::XgcdBothZero` if both polynomials are zero, as gcd(0, 0) is undefined.
    pub fn xgcd(&self, y: &Self) -> Result<(Self, Self, Self), PolynomialError> {
        // Handle special cases where one or both polynomials are zero
        if self.is_zero() && y.is_zero() {
            return Err(PolynomialError::XgcdBothZero);
        }

        let (mut old_r, mut r) = (self.clone(), y.clone());
        let (mut old_s, mut s) = (
            Polynomial::new(&[FieldElement::one()]),
            Polynomial::zero(),
        );
        let (mut old_t, mut t) = (
            Polynomial::zero(),
            Polynomial::new(&[FieldElement::one()]),
        );

        while !r.is_zero() {
            // Division is safe: the loop condition guarantees r is non-zero
            let quotient = old_r
                .clone()
                .div_with_ref(&r)
                .expect("divisor is non-zero by loop invariant");

            // old_r = old_r - quotient * r, then swap
            let qr = quotient.mul_with_ref(&r);
            old_r.sub_assign(&qr);
            core::mem::swap(&mut old_r, &mut r);

            // old_s = old_s - quotient * s, then swap
            let qs = quotient.mul_with_ref(&s);
            old_s.sub_assign(&qs);
            core::mem::swap(&mut old_s, &mut s);

            // old_t = old_t - quotient * t, then swap
            let qt = quotient.mul_with_ref(&t);
            old_t.sub_assign(&qt);
            core::mem::swap(&mut old_t, &mut t);
        }

        let lcinv = old_r
            .leading_coefficient()
            .inv()
            .expect("GCD should be non-zero when at least one input is non-zero");

        // Scale results in-place
        old_s.scale_coeffs_mut(&lcinv);
        old_t.scale_coeffs_mut(&lcinv);
        old_r.scale_coeffs_mut(&lcinv);

        Ok((old_s, old_t, old_r))
    }

    /// Divides this polynomial by the divisor, returning only the quotient.
    ///
    /// # Errors
    /// Returns `PolynomialError::DivisionByZero` if the divisor is the zero polynomial.
    pub fn div_with_ref(self, divisor: &Self) -> Result<Self, PolynomialError> {
        let (quotient, _remainder) = self.long_division_with_remainder(divisor)?;
        Ok(quotient)
    }

    pub fn mul_with_ref(&self, factor: &Self) -> Self {
        let degree = self.degree() + factor.degree();
        let mut coefficients = vec![FieldElement::zero(); degree + 1];

        if self.coefficients.is_empty() || factor.coefficients.is_empty() {
            Polynomial::new(&[FieldElement::zero()])
        } else {
            for i in 0..=factor.degree() {
                if factor.coefficients[i] != FieldElement::zero() {
                    for j in 0..=self.degree() {
                        if self.coefficients[j] != FieldElement::zero() {
                            coefficients[i + j] += &factor.coefficients[i] * &self.coefficients[j];
                        }
                    }
                }
            }
            Polynomial::new(&coefficients)
        }
    }

    /// Scales the coefficients of a polynomial P by a factor
    /// Returns P(factor * x)
    pub fn scale<S: IsSubFieldOf<F>>(&self, factor: &FieldElement<S>) -> Self {
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

    /// Multiplies all coefficients by a factor
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

    /// Multiplies all coefficients by a factor in-place
    pub fn scale_coeffs_mut(&mut self, factor: &FieldElement<F>) {
        for coeff in self.coefficients.iter_mut() {
            *coeff = &*coeff * factor;
        }
    }

    /// Adds another polynomial to this one in-place.
    /// If the other polynomial has higher degree, this polynomial is extended.
    pub fn add_assign(&mut self, other: &Self) {
        if other.coefficients.len() > self.coefficients.len() {
            self.coefficients
                .resize(other.coefficients.len(), FieldElement::zero());
        }
        for (a, b) in self.coefficients.iter_mut().zip(other.coefficients.iter()) {
            *a = &*a + b;
        }
        // Remove trailing zeros
        while self.coefficients.last().map_or(false, |c| *c == FieldElement::zero()) {
            self.coefficients.pop();
        }
    }

    /// Subtracts another polynomial from this one in-place.
    /// If the other polynomial has higher degree, this polynomial is extended.
    pub fn sub_assign(&mut self, other: &Self) {
        if other.coefficients.len() > self.coefficients.len() {
            self.coefficients
                .resize(other.coefficients.len(), FieldElement::zero());
        }
        for (a, b) in self.coefficients.iter_mut().zip(other.coefficients.iter()) {
            *a = &*a - b;
        }
        // Remove trailing zeros
        while self.coefficients.last().map_or(false, |c| *c == FieldElement::zero()) {
            self.coefficients.pop();
        }
    }

    /// Negates all coefficients in-place
    pub fn neg_mut(&mut self) {
        for coeff in self.coefficients.iter_mut() {
            *coeff = -&*coeff;
        }
    }

    /// Returns a vector of polynomials [p₀, p₁, ..., p_{d-1}], where d is `number_of_parts`, such that `self` equals
    /// p₀(Xᵈ) + Xp₁(Xᵈ) + ... + X^(d-1)p_{d-1}(Xᵈ).
    ///
    /// Example: if d = 2 and `self` is 3 X^3 + X^2 + 2X + 1, then `poly.break_in_parts(2)`
    /// returns a vector with two polynomials `(p₀, p₁)`, where p₀ = X + 1 and p₁ = 3X + 2.
    pub fn break_in_parts(&self, number_of_parts: usize) -> Vec<Self> {
        let coef = self.coefficients();
        let mut parts: Vec<Self> = Vec::with_capacity(number_of_parts);
        for i in 0..number_of_parts {
            let coeffs: Vec<_> = coef
                .iter()
                .skip(i)
                .step_by(number_of_parts)
                .cloned()
                .collect();
            parts.push(Polynomial::new(&coeffs));
        }
        parts
    }

    /// Embeds the coefficients of a polynomial into an extension field
    /// For example, given a polynomial with coefficients in F_p, returns the same
    /// polynomial with its coefficients as elements in F_{p^2}
    pub fn to_extension<L: IsField>(self) -> Polynomial<FieldElement<L>>
    where
        F: IsSubFieldOf<L>,
    {
        Polynomial {
            coefficients: self
                .coefficients
                .into_iter()
                .map(|x| x.to_extension::<L>())
                .collect(),
        }
    }

    pub fn truncate(&self, k: usize) -> Self {
        if k == 0 {
            Self::zero()
        } else {
            Self::new(&self.coefficients[0..k.min(self.coefficients.len())])
        }
    }
    pub fn reverse(&self, d: usize) -> Self {
        let mut coeffs = self.coefficients.clone();
        coeffs.resize(d + 1, FieldElement::zero());
        coeffs.reverse();
        Self::new(&coeffs)
    }
}

impl<F: IsPrimeField> Polynomial<FieldElement<F>> {
    // Print the polynomial as a string ready to be used in SageMath, or just for pretty printing.
    pub fn print_as_sage_poly(&self, var_name: Option<char>) -> String {
        let var_name = var_name.unwrap_or('x');
        if self.coefficients.is_empty()
            || self.coefficients.len() == 1 && self.coefficients[0] == FieldElement::zero()
        {
            return String::new();
        }

        let mut string = String::new();
        let zero = FieldElement::<F>::zero();

        for (i, coeff) in self.coefficients.iter().rev().enumerate() {
            if *coeff == zero {
                continue;
            }

            let coeff_str = coeff.representative().to_string();

            if i == self.coefficients.len() - 1 {
                string.push_str(&coeff_str);
            } else if i == self.coefficients.len() - 2 {
                string.push_str(&format!("{coeff_str}*{var_name} + "));
            } else {
                string.push_str(&format!(
                    "{}*{}^{} + ",
                    coeff_str,
                    var_name,
                    self.coefficients.len() - 1 - i
                ));
            }
        }

        string
    }
}

/// Pads a polynomial with zeros until the desired length
/// This function can be useful when evaluating polynomials with the FFT
pub fn pad_with_zero_coefficients_to_length<F: IsField>(
    pa: &mut Polynomial<FieldElement<F>>,
    n: usize,
) {
    pa.coefficients.resize(n, FieldElement::zero());
}

/// Pads polynomial representations with minimum number of zeros to match lengths.
pub fn pad_with_zero_coefficients<L: IsField, F: IsSubFieldOf<L>>(
    pa: &Polynomial<FieldElement<F>>,
    pb: &Polynomial<FieldElement<L>>,
) -> (Polynomial<FieldElement<F>>, Polynomial<FieldElement<L>>) {
    let mut pa = pa.clone();
    let mut pb = pb.clone();

    if pa.coefficients.len() > pb.coefficients.len() {
        pad_with_zero_coefficients_to_length(&mut pb, pa.coefficients.len());
    } else {
        pad_with_zero_coefficients_to_length(&mut pa, pb.coefficients.len());
    }
    (pa, pb)
}

/// Computes the composition of polynomials `P1(t)` and `P2(t)`, that is `P1(P2(t))`
/// It uses interpolation to determine the evaluation at points `x_i` and evaluates
/// `P1(P2(x[i]))`. The interpolation theorem ensures that we can reconstruct the polynomial
/// uniquely by interpolation over a suitable number of points.
/// This is an inefficient version, for something more efficient, use FFT for evaluation,
/// provided the field satisfies the necessary traits.
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

// impl Add
impl<F, L> ops::Add<&Polynomial<FieldElement<L>>> for &Polynomial<FieldElement<F>>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = Polynomial<FieldElement<L>>;

    fn add(self, other: &Polynomial<FieldElement<L>>) -> Self::Output {
        let max_len = self.coefficients.len().max(other.coefficients.len());
        let mut result = Vec::with_capacity(max_len);

        // Add coefficients where both polynomials have terms
        let common_len = self.coefficients.len().min(other.coefficients.len());
        for i in 0..common_len {
            result.push(self.coefficients[i].clone().to_extension::<L>() + &other.coefficients[i]);
        }

        // Handle remaining coefficients from self (F -> L extension)
        for coeff in self.coefficients.iter().skip(common_len) {
            result.push(coeff.clone().to_extension::<L>());
        }

        // Handle remaining coefficients from other (already in L)
        for coeff in other.coefficients.iter().skip(common_len) {
            result.push(coeff.clone());
        }

        Polynomial::new(&result)
    }
}

impl<F, L> ops::Add<Polynomial<FieldElement<L>>> for Polynomial<FieldElement<F>>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = Polynomial<FieldElement<L>>;

    fn add(self, a_polynomial: Polynomial<FieldElement<L>>) -> Polynomial<FieldElement<L>> {
        &self + &a_polynomial
    }
}

impl<F, L> ops::Add<&Polynomial<FieldElement<L>>> for Polynomial<FieldElement<F>>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = Polynomial<FieldElement<L>>;

    fn add(self, a_polynomial: &Polynomial<FieldElement<L>>) -> Polynomial<FieldElement<L>> {
        &self + a_polynomial
    }
}

impl<F, L> ops::Add<Polynomial<FieldElement<L>>> for &Polynomial<FieldElement<F>>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = Polynomial<FieldElement<L>>;

    fn add(self, a_polynomial: Polynomial<FieldElement<L>>) -> Polynomial<FieldElement<L>> {
        self + &a_polynomial
    }
}

// impl neg, that is, additive inverse for polynomials P(t) + Q(t) = 0
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

// impl Sub
impl<F, L> ops::Sub<&Polynomial<FieldElement<L>>> for &Polynomial<FieldElement<F>>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = Polynomial<FieldElement<L>>;

    fn sub(self, other: &Polynomial<FieldElement<L>>) -> Polynomial<FieldElement<L>> {
        let max_len = self.coefficients.len().max(other.coefficients.len());
        let mut result = Vec::with_capacity(max_len);

        // Subtract coefficients where both polynomials have terms
        let common_len = self.coefficients.len().min(other.coefficients.len());
        for i in 0..common_len {
            result.push(self.coefficients[i].clone().to_extension::<L>() - &other.coefficients[i]);
        }

        // Handle remaining coefficients from self (F -> L extension)
        for coeff in self.coefficients.iter().skip(common_len) {
            result.push(coeff.clone().to_extension::<L>());
        }

        // Handle remaining coefficients from other (negate them)
        for coeff in other.coefficients.iter().skip(common_len) {
            result.push(-coeff);
        }

        Polynomial::new(&result)
    }
}

impl<F, L> ops::Sub<Polynomial<FieldElement<L>>> for Polynomial<FieldElement<F>>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = Polynomial<FieldElement<L>>;

    fn sub(self, substrahend: Polynomial<FieldElement<L>>) -> Polynomial<FieldElement<L>> {
        &self - &substrahend
    }
}

impl<F, L> ops::Sub<&Polynomial<FieldElement<L>>> for Polynomial<FieldElement<F>>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = Polynomial<FieldElement<L>>;

    fn sub(self, substrahend: &Polynomial<FieldElement<L>>) -> Polynomial<FieldElement<L>> {
        &self - substrahend
    }
}

impl<F, L> ops::Sub<Polynomial<FieldElement<L>>> for &Polynomial<FieldElement<F>>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = Polynomial<FieldElement<L>>;

    fn sub(self, substrahend: Polynomial<FieldElement<L>>) -> Polynomial<FieldElement<L>> {
        self - &substrahend
    }
}

impl<F> ops::Div<Polynomial<FieldElement<F>>> for Polynomial<FieldElement<F>>
where
    F: IsField,
{
    type Output = Polynomial<FieldElement<F>>;

    /// Divides this polynomial by the divisor.
    ///
    /// # Panics
    /// Panics if the divisor is the zero polynomial. For error handling,
    /// use `div_with_ref` or `long_division_with_remainder` instead.
    fn div(self, divisor: Polynomial<FieldElement<F>>) -> Polynomial<FieldElement<F>> {
        self.div_with_ref(&divisor)
            .expect("Cannot divide by the zero polynomial")
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
impl<F, L> ops::Mul<FieldElement<F>> for Polynomial<FieldElement<L>>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = Polynomial<FieldElement<L>>;

    fn mul(self, multiplicand: FieldElement<F>) -> Polynomial<FieldElement<L>> {
        let new_coefficients = self
            .coefficients
            .iter()
            .map(|value| &multiplicand * value)
            .collect();
        Polynomial {
            coefficients: new_coefficients,
        }
    }
}

impl<F, L> ops::Mul<&FieldElement<F>> for &Polynomial<FieldElement<L>>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = Polynomial<FieldElement<L>>;

    fn mul(self, multiplicand: &FieldElement<F>) -> Polynomial<FieldElement<L>> {
        let new_coefficients = self
            .coefficients
            .iter()
            .map(|value| multiplicand * value)
            .collect();
        Polynomial {
            coefficients: new_coefficients,
        }
    }
}

impl<F, L> ops::Mul<FieldElement<F>> for &Polynomial<FieldElement<L>>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = Polynomial<FieldElement<L>>;

    fn mul(self, multiplicand: FieldElement<F>) -> Polynomial<FieldElement<L>> {
        self * &multiplicand
    }
}

impl<F, L> ops::Mul<&FieldElement<F>> for Polynomial<FieldElement<L>>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = Polynomial<FieldElement<L>>;

    fn mul(self, multiplicand: &FieldElement<F>) -> Polynomial<FieldElement<L>> {
        &self * multiplicand
    }
}

/* Multiplication field element at right */
impl<F, L> ops::Mul<&Polynomial<FieldElement<L>>> for &FieldElement<F>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = Polynomial<FieldElement<L>>;

    fn mul(self, multiplicand: &Polynomial<FieldElement<L>>) -> Polynomial<FieldElement<L>> {
        multiplicand * self
    }
}

impl<F, L> ops::Mul<Polynomial<FieldElement<L>>> for &FieldElement<F>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = Polynomial<FieldElement<L>>;

    fn mul(self, multiplicand: Polynomial<FieldElement<L>>) -> Polynomial<FieldElement<L>> {
        &multiplicand * self
    }
}

impl<F, L> ops::Mul<&Polynomial<FieldElement<L>>> for FieldElement<F>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = Polynomial<FieldElement<L>>;

    fn mul(self, multiplicand: &Polynomial<FieldElement<L>>) -> Polynomial<FieldElement<L>> {
        multiplicand * self
    }
}

impl<F, L> ops::Mul<Polynomial<FieldElement<L>>> for FieldElement<F>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = Polynomial<FieldElement<L>>;

    fn mul(self, multiplicand: Polynomial<FieldElement<L>>) -> Polynomial<FieldElement<L>> {
        &multiplicand * &self
    }
}

/* Addition field element at left */
impl<F, L> ops::Add<&FieldElement<F>> for &Polynomial<FieldElement<L>>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = Polynomial<FieldElement<L>>;

    fn add(self, other: &FieldElement<F>) -> Polynomial<FieldElement<L>> {
        Polynomial::new_monomial(other.clone(), 0) + self
    }
}

impl<F, L> ops::Add<FieldElement<F>> for Polynomial<FieldElement<L>>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = Polynomial<FieldElement<L>>;

    fn add(self, other: FieldElement<F>) -> Polynomial<FieldElement<L>> {
        &self + &other
    }
}

impl<F, L> ops::Add<FieldElement<F>> for &Polynomial<FieldElement<L>>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = Polynomial<FieldElement<L>>;

    fn add(self, other: FieldElement<F>) -> Polynomial<FieldElement<L>> {
        self + &other
    }
}

impl<F, L> ops::Add<&FieldElement<F>> for Polynomial<FieldElement<L>>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = Polynomial<FieldElement<L>>;

    fn add(self, other: &FieldElement<F>) -> Polynomial<FieldElement<L>> {
        &self + other
    }
}

/* Addition field element at right */
impl<F, L> ops::Add<&Polynomial<FieldElement<L>>> for &FieldElement<F>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = Polynomial<FieldElement<L>>;

    fn add(self, other: &Polynomial<FieldElement<L>>) -> Polynomial<FieldElement<L>> {
        Polynomial::new_monomial(self.clone(), 0) + other
    }
}

impl<F, L> ops::Add<Polynomial<FieldElement<L>>> for FieldElement<F>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = Polynomial<FieldElement<L>>;

    fn add(self, other: Polynomial<FieldElement<L>>) -> Polynomial<FieldElement<L>> {
        &self + &other
    }
}

impl<F, L> ops::Add<Polynomial<FieldElement<L>>> for &FieldElement<F>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = Polynomial<FieldElement<L>>;

    fn add(self, other: Polynomial<FieldElement<L>>) -> Polynomial<FieldElement<L>> {
        self + &other
    }
}

impl<F, L> ops::Add<&Polynomial<FieldElement<L>>> for FieldElement<F>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = Polynomial<FieldElement<L>>;

    fn add(self, other: &Polynomial<FieldElement<L>>) -> Polynomial<FieldElement<L>> {
        &self + other
    }
}

/* Substraction field element at left */
impl<F, L> ops::Sub<&FieldElement<F>> for &Polynomial<FieldElement<L>>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = Polynomial<FieldElement<L>>;

    fn sub(self, other: &FieldElement<F>) -> Polynomial<FieldElement<L>> {
        -Polynomial::new_monomial(other.clone(), 0) + self
    }
}

impl<F, L> ops::Sub<FieldElement<F>> for Polynomial<FieldElement<L>>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = Polynomial<FieldElement<L>>;

    fn sub(self, other: FieldElement<F>) -> Polynomial<FieldElement<L>> {
        &self - &other
    }
}

impl<F, L> ops::Sub<FieldElement<F>> for &Polynomial<FieldElement<L>>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = Polynomial<FieldElement<L>>;

    fn sub(self, other: FieldElement<F>) -> Polynomial<FieldElement<L>> {
        self - &other
    }
}

impl<F, L> ops::Sub<&FieldElement<F>> for Polynomial<FieldElement<L>>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = Polynomial<FieldElement<L>>;

    fn sub(self, other: &FieldElement<F>) -> Polynomial<FieldElement<L>> {
        &self - other
    }
}

/* Substraction field element at right */
impl<F, L> ops::Sub<&Polynomial<FieldElement<L>>> for &FieldElement<F>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = Polynomial<FieldElement<L>>;

    fn sub(self, other: &Polynomial<FieldElement<L>>) -> Polynomial<FieldElement<L>> {
        Polynomial::new_monomial(self.clone(), 0) - other
    }
}

impl<F, L> ops::Sub<Polynomial<FieldElement<L>>> for FieldElement<F>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = Polynomial<FieldElement<L>>;

    fn sub(self, other: Polynomial<FieldElement<L>>) -> Polynomial<FieldElement<L>> {
        &self - &other
    }
}

impl<F, L> ops::Sub<Polynomial<FieldElement<L>>> for &FieldElement<F>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = Polynomial<FieldElement<L>>;

    fn sub(self, other: Polynomial<FieldElement<L>>) -> Polynomial<FieldElement<L>> {
        self - &other
    }
}

impl<F, L> ops::Sub<&Polynomial<FieldElement<L>>> for FieldElement<F>
where
    L: IsField,
    F: IsSubFieldOf<L>,
{
    type Output = Polynomial<FieldElement<L>>;

    fn sub(self, other: &Polynomial<FieldElement<L>>) -> Polynomial<FieldElement<L>> {
        &self - other
    }
}

// In-place assignment operators
impl<F: IsField> AddAssign<&Polynomial<FieldElement<F>>> for Polynomial<FieldElement<F>> {
    fn add_assign(&mut self, other: &Polynomial<FieldElement<F>>) {
        self.add_assign(other);
    }
}

impl<F: IsField> AddAssign<Polynomial<FieldElement<F>>> for Polynomial<FieldElement<F>> {
    fn add_assign(&mut self, other: Polynomial<FieldElement<F>>) {
        self.add_assign(&other);
    }
}

impl<F: IsField> SubAssign<&Polynomial<FieldElement<F>>> for Polynomial<FieldElement<F>> {
    fn sub_assign(&mut self, other: &Polynomial<FieldElement<F>>) {
        Polynomial::sub_assign(self, other);
    }
}

impl<F: IsField> SubAssign<Polynomial<FieldElement<F>>> for Polynomial<FieldElement<F>> {
    fn sub_assign(&mut self, other: Polynomial<FieldElement<F>>) {
        Polynomial::sub_assign(self, &other);
    }
}

impl<F: IsField> MulAssign<&FieldElement<F>> for Polynomial<FieldElement<F>> {
    fn mul_assign(&mut self, scalar: &FieldElement<F>) {
        self.scale_coeffs_mut(scalar);
    }
}

impl<F: IsField> MulAssign<FieldElement<F>> for Polynomial<FieldElement<F>> {
    fn mul_assign(&mut self, scalar: FieldElement<F>) {
        self.scale_coeffs_mut(&scalar);
    }
}

#[derive(Debug)]
pub enum InterpolateError {
    UnequalLengths(usize, usize),
    NonUniqueXs,
}

impl Display for InterpolateError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            InterpolateError::UnequalLengths(x, y) => {
                write!(f, "xs and ys must be the same length. Got: {x} != {y}")
            }
            InterpolateError::NonUniqueXs => write!(f, "xs values should be unique."),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for InterpolateError {}

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
        let (pp1, pp2) = pad_with_zero_coefficients(&p1, &p2);
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
        let denominator = Polynomial::new(&[FE::new(1) * (FE::new(2) - FE::new(4)).inv().unwrap()]);
        let numerator = Polynomial::new(&[-FE::new(4), FE::new(1)]);
        let interpolating = numerator * denominator;
        assert_eq!(
            (FE::new(2) - FE::new(4)) * (FE::new(1) * (FE::new(2) - FE::new(4)).inv().unwrap()),
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

    #[test]
    fn break_in_parts() {
        // p = 3 X^3 + X^2 + 2X + 1
        let p = Polynomial::new(&[FE::new(1), FE::new(2), FE::new(1), FE::new(3)]);
        let p0_expected = Polynomial::new(&[FE::new(1), FE::new(1)]);
        let p1_expected = Polynomial::new(&[FE::new(2), FE::new(3)]);
        let parts = p.break_in_parts(2);
        assert_eq!(parts.len(), 2);
        let p0 = &parts[0];
        let p1 = &parts[1];
        assert_eq!(p0, &p0_expected);
        assert_eq!(p1, &p1_expected);
    }

    use proptest::prelude::*;
    proptest! {
        #[test]
        fn ruffini_inplace_equals_division(p in any::<Vec<u64>>(), b in any::<u64>()) {
            let p: Vec<_> = p.into_iter().map(FE::from).collect();
            let mut p = Polynomial::new(&p);
            let b = FE::from(b);

            let p_ref = p.clone();
            let m = Polynomial::new_monomial(FE::one(), 1) - b;

            p.ruffini_division_inplace(&b);
            prop_assert_eq!(p, p_ref / m);
        }
    }

    proptest! {
        #[test]
        fn ruffini_inplace_equals_ruffini(p in any::<Vec<u64>>(), b in any::<u64>()) {
            let p: Vec<_> = p.into_iter().map(FE::from).collect();
            let mut p = Polynomial::new(&p);
            let b = FE::from(b);
            let q = p.ruffini_division(&b);
            p.ruffini_division_inplace(&b);
            prop_assert_eq!(q, p);
        }
    }
    #[test]
    fn test_xgcd() {
        // Case 1: Simple polynomials
        let p1 = Polynomial::new(&[FE::new(1), FE::new(0), FE::new(1)]); // x^2 + 1
        let p2 = Polynomial::new(&[FE::new(1), FE::new(1)]); // x + 1
        let (a, b, g) = p1.xgcd(&p2).unwrap();
        // Check that a * p1 + b * p2 = g
        let lhs = a.mul_with_ref(&p1) + b.mul_with_ref(&p2);
        assert_eq!(a, Polynomial::new(&[FE::new(12)]));
        assert_eq!(b, Polynomial::new(&[FE::new(12), FE::new(11)]));
        assert_eq!(lhs, g);
        assert_eq!(g, Polynomial::new(&[FE::new(1)]));

        // x^2-1 :
        let p3 = Polynomial::new(&[FE::new(ORDER - 1), FE::new(0), FE::new(1)]);
        // x^3-x = x(x^2-1)
        let p4 = Polynomial::new(&[FE::new(0), FE::new(ORDER - 1), FE::new(0), FE::new(1)]);
        let (a, b, g) = p3.xgcd(&p4).unwrap();

        let lhs = a.mul_with_ref(&p3) + b.mul_with_ref(&p4);
        assert_eq!(a, Polynomial::new(&[FE::new(1)]));
        assert_eq!(b, Polynomial::zero());
        assert_eq!(lhs, g);
        assert_eq!(g, p3);
    }

    #[test]
    fn test_differentiate() {
        // 3x^2 + 2x + 42
        let px = Polynomial::new(&[FE::new(42), FE::new(2), FE::new(3)]);
        // 6x + 2
        let dpdx = px.differentiate();
        assert_eq!(dpdx, Polynomial::new(&[FE::new(2), FE::new(6)]));

        // 128
        let px = Polynomial::new(&[FE::new(128)]);
        // 0
        let dpdx = px.differentiate();
        assert_eq!(dpdx, Polynomial::new(&[FE::new(0)]));
    }

    #[test]
    fn test_reverse() {
        let p = Polynomial::new(&[FE::new(3), FE::new(2), FE::new(1)]);
        assert_eq!(
            p.reverse(3),
            Polynomial::new(&[FE::new(0), FE::new(1), FE::new(2), FE::new(3)])
        );
    }

    #[test]
    fn test_truncate() {
        let p = Polynomial::new(&[FE::new(3), FE::new(2), FE::new(1)]);
        assert_eq!(p.truncate(2), Polynomial::new(&[FE::new(3), FE::new(2)]));
    }

    #[test]
    fn test_print_as_sage_poly() {
        let p = Polynomial::new(&[FE::new(1), FE::new(2), FE::new(3)]);
        assert_eq!(p.print_as_sage_poly(None), "3*x^2 + 2*x + 1");
    }

    #[test]
    fn test_is_zero() {
        // Canonical zero polynomial
        assert!(Polynomial::<FE>::zero().is_zero());

        // Empty coefficients
        assert!(Polynomial::<FE>::new(&[]).is_zero());

        // Single zero coefficient
        assert!(Polynomial::new(&[FE::new(0)]).is_zero());

        // Non-zero polynomial
        assert!(!Polynomial::new(&[FE::new(1)]).is_zero());
        assert!(!Polynomial::new(&[FE::new(0), FE::new(1)]).is_zero());

        // Non-canonical zero (created via scale_coeffs with zero)
        let p = Polynomial::new(&[FE::new(1), FE::new(2)]);
        let scaled = p.scale_coeffs(&FE::new(0));
        assert!(scaled.is_zero());
    }

    #[test]
    fn test_division_by_zero_returns_error() {
        let p = Polynomial::new(&[FE::new(1), FE::new(2)]);
        let zero = Polynomial::<FE>::zero();
        let result = p.long_division_with_remainder(&zero);
        assert_eq!(result, Err(super::PolynomialError::DivisionByZero));
    }

    #[test]
    fn test_division_by_non_canonical_zero_returns_error() {
        let p = Polynomial::new(&[FE::new(1), FE::new(2)]);
        // Create a non-canonical zero polynomial via scale_coeffs
        let non_canonical_zero = Polynomial::new(&[FE::new(1)]).scale_coeffs(&FE::new(0));
        let result = p.long_division_with_remainder(&non_canonical_zero);
        assert_eq!(result, Err(super::PolynomialError::DivisionByZero));
    }

    #[test]
    fn test_xgcd_both_zero_returns_error() {
        let zero = Polynomial::<FE>::zero();
        let result = zero.xgcd(&zero);
        assert_eq!(result, Err(super::PolynomialError::XgcdBothZero));
    }

    #[test]
    fn test_xgcd_non_canonical_zeros_returns_error() {
        // Create non-canonical zero polynomials
        let zero1 = Polynomial::new(&[FE::new(1)]).scale_coeffs(&FE::new(0));
        let zero2 = Polynomial::new(&[FE::new(2), FE::new(3)]).scale_coeffs(&FE::new(0));
        let result = zero1.xgcd(&zero2);
        assert_eq!(result, Err(super::PolynomialError::XgcdBothZero));
    }

    // Tests for in-place operations
    #[test]
    fn test_add_assign() {
        let mut p = polynomial_a();
        p.add_assign(&polynomial_b());
        assert_eq!(p, polynomial_a_plus_b());
    }

    #[test]
    fn test_add_assign_different_lengths() {
        // Longer polynomial on the left
        let mut p = Polynomial::new(&[FE::new(1), FE::new(2), FE::new(3)]);
        let q = Polynomial::new(&[FE::new(4), FE::new(5)]);
        p.add_assign(&q);
        assert_eq!(p, Polynomial::new(&[FE::new(5), FE::new(7), FE::new(3)]));

        // Shorter polynomial on the left
        let mut p = Polynomial::new(&[FE::new(1), FE::new(2)]);
        let q = Polynomial::new(&[FE::new(3), FE::new(4), FE::new(5)]);
        p.add_assign(&q);
        assert_eq!(p, Polynomial::new(&[FE::new(4), FE::new(6), FE::new(5)]));
    }

    #[test]
    fn test_add_assign_trims_zeros() {
        let mut p = Polynomial::new(&[FE::new(1), FE::new(2), FE::new(3)]);
        let q = Polynomial::new(&[FE::new(0), FE::new(0), FE::new(ORDER - 3)]); // -3 in the field
        p.add_assign(&q);
        assert_eq!(p, Polynomial::new(&[FE::new(1), FE::new(2)]));
    }

    #[test]
    fn test_sub_assign() {
        let mut p = polynomial_b();
        Polynomial::sub_assign(&mut p, &polynomial_a());
        assert_eq!(p, polynomial_b_minus_a());
    }

    #[test]
    fn test_sub_assign_different_lengths() {
        // Longer polynomial on the left
        let mut p = Polynomial::new(&[FE::new(5), FE::new(6), FE::new(3)]);
        let q = Polynomial::new(&[FE::new(1), FE::new(2)]);
        Polynomial::sub_assign(&mut p, &q);
        assert_eq!(p, Polynomial::new(&[FE::new(4), FE::new(4), FE::new(3)]));

        // Shorter polynomial on the left
        let mut p = Polynomial::new(&[FE::new(5), FE::new(6)]);
        let q = Polynomial::new(&[FE::new(1), FE::new(2), FE::new(3)]);
        Polynomial::sub_assign(&mut p, &q);
        assert_eq!(
            p,
            Polynomial::new(&[FE::new(4), FE::new(4), FE::new(ORDER - 3)])
        );
    }

    #[test]
    fn test_sub_assign_trims_zeros() {
        let mut p = Polynomial::new(&[FE::new(1), FE::new(2), FE::new(3)]);
        let q = Polynomial::new(&[FE::new(0), FE::new(0), FE::new(3)]);
        Polynomial::sub_assign(&mut p, &q);
        assert_eq!(p, Polynomial::new(&[FE::new(1), FE::new(2)]));
    }

    #[test]
    fn test_scale_coeffs_mut() {
        let mut p = Polynomial::new(&[FE::new(1), FE::new(2), FE::new(3)]);
        p.scale_coeffs_mut(&FE::new(2));
        assert_eq!(p, Polynomial::new(&[FE::new(2), FE::new(4), FE::new(6)]));
    }

    #[test]
    fn test_scale_coeffs_mut_by_zero() {
        let mut p = Polynomial::new(&[FE::new(1), FE::new(2), FE::new(3)]);
        p.scale_coeffs_mut(&FE::new(0));
        // Note: scale_coeffs_mut doesn't trim, but is_zero handles non-canonical zeros
        assert!(p.is_zero());
    }

    #[test]
    fn test_neg_mut() {
        let mut p = polynomial_a();
        p.neg_mut();
        assert_eq!(p, polynomial_minus_a());
    }

    #[test]
    fn test_neg_mut_double_negation() {
        let mut p = polynomial_a();
        p.neg_mut();
        p.neg_mut();
        assert_eq!(p, polynomial_a());
    }

    #[test]
    fn test_add_assign_trait() {
        let mut p = polynomial_a();
        p += &polynomial_b();
        assert_eq!(p, polynomial_a_plus_b());
    }

    #[test]
    fn test_add_assign_trait_owned() {
        let mut p = polynomial_a();
        p += polynomial_b();
        assert_eq!(p, polynomial_a_plus_b());
    }

    #[test]
    fn test_sub_assign_trait() {
        let mut p = polynomial_b();
        p -= &polynomial_a();
        assert_eq!(p, polynomial_b_minus_a());
    }

    #[test]
    fn test_sub_assign_trait_owned() {
        let mut p = polynomial_b();
        p -= polynomial_a();
        assert_eq!(p, polynomial_b_minus_a());
    }

    #[test]
    fn test_mul_assign_trait() {
        let mut p = Polynomial::new(&[FE::new(1), FE::new(2), FE::new(3)]);
        p *= &FE::new(2);
        assert_eq!(p, Polynomial::new(&[FE::new(2), FE::new(4), FE::new(6)]));
    }

    #[test]
    fn test_mul_assign_trait_owned() {
        let mut p = Polynomial::new(&[FE::new(1), FE::new(2), FE::new(3)]);
        p *= FE::new(2);
        assert_eq!(p, Polynomial::new(&[FE::new(2), FE::new(4), FE::new(6)]));
    }
}
