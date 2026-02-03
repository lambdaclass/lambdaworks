//! Sparse polynomial representation for memory-efficient storage of polynomials
//! with many zero coefficients.
//!
//! A sparse polynomial is represented as a map from degree (exponent) to coefficient,
//! storing only non-zero terms. This is particularly useful for:
//!
//! - R1CS constraint systems where polynomials have few non-zero terms
//! - Vanishing polynomials like X^n - 1
//! - Custom gates in PLONK-like systems
//!
//! # Mathematical Definition
//!
//! A sparse polynomial P(X) is represented as:
//! P(X) = sum_{i in S} c_i * X^i
//!
//! where S is the set of indices with non-zero coefficients, and c_i are the coefficients.
//!
//! # Performance Characteristics
//!
//! For a polynomial of degree d with k non-zero terms:
//! - Memory: O(k) instead of O(d) for dense representation
//! - Addition/Subtraction: O(k1 + k2) where k1, k2 are non-zero term counts
//! - Multiplication: O(k1 * k2)
//! - Evaluation: O(k * log(d)) using efficient exponentiation
//!
//! # Example
//!
//! ```ignore
//! use lambdaworks_math::polynomial::sparse::SparsePolynomial;
//! use lambdaworks_math::field::element::FieldElement;
//! use lambdaworks_math::field::fields::u64_prime_field::U64PrimeField;
//!
//! type F = U64PrimeField<17>;
//! type FE = FieldElement<F>;
//!
//! // Create polynomial 3*X^100 + 2*X^50 + 1
//! let poly = SparsePolynomial::from_coefficients(vec![
//!     (0, FE::from(1)),
//!     (50, FE::from(2)),
//!     (100, FE::from(3)),
//! ]);
//!
//! assert_eq!(poly.degree(), 100);
//! assert_eq!(poly.num_terms(), 3);
//! ```

use crate::field::element::FieldElement;
use crate::field::traits::IsField;
use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use core::ops::{Add, Mul, Neg, Sub};

use super::Polynomial;

/// A sparse polynomial represented as a map from degree to coefficient.
///
/// Only non-zero coefficients are stored. The polynomial
/// `c_0 + c_1*X + c_2*X^2 + ... + c_n*X^n`
/// is stored as a map `{i -> c_i}` for all i where c_i != 0.
///
/// # Type Parameters
///
/// * `F` - The field over which the polynomial is defined
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SparsePolynomial<F: IsField> {
    /// Map from degree (exponent) to coefficient.
    /// Invariant: All values in the map are non-zero.
    coefficients: BTreeMap<usize, FieldElement<F>>,
}

impl<F: IsField> Default for SparsePolynomial<F> {
    fn default() -> Self {
        Self::zero()
    }
}

impl<F: IsField> SparsePolynomial<F> {
    /// Creates a new empty sparse polynomial (the zero polynomial).
    ///
    /// # Example
    ///
    /// ```ignore
    /// let zero_poly: SparsePolynomial<F> = SparsePolynomial::new();
    /// assert!(zero_poly.is_zero());
    /// ```
    #[inline]
    pub fn new() -> Self {
        Self {
            coefficients: BTreeMap::new(),
        }
    }

    /// Creates the zero polynomial.
    ///
    /// Alias for `new()` for consistency with the dense polynomial API.
    #[inline]
    pub fn zero() -> Self {
        Self::new()
    }

    /// Creates a sparse polynomial from a vector of (degree, coefficient) pairs.
    ///
    /// Zero coefficients are automatically filtered out to maintain the invariant
    /// that only non-zero terms are stored.
    ///
    /// # Arguments
    ///
    /// * `coeffs` - Vector of (degree, coefficient) pairs
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Create polynomial 5*X^3 + 2*X + 1
    /// let poly = SparsePolynomial::from_coefficients(vec![
    ///     (0, FE::from(1)),
    ///     (1, FE::from(2)),
    ///     (3, FE::from(5)),
    /// ]);
    /// ```
    pub fn from_coefficients(coeffs: Vec<(usize, FieldElement<F>)>) -> Self {
        let mut map = BTreeMap::new();
        let zero = FieldElement::zero();
        for (degree, coeff) in coeffs {
            if coeff != zero {
                // If the same degree appears multiple times, the last value wins
                // (alternatively, we could add them)
                map.insert(degree, coeff);
            }
        }
        Self { coefficients: map }
    }

    /// Creates a monomial c * X^d.
    ///
    /// # Arguments
    ///
    /// * `coefficient` - The coefficient c
    /// * `degree` - The degree d
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Create 7*X^5
    /// let mono = SparsePolynomial::new_monomial(FE::from(7), 5);
    /// ```
    pub fn new_monomial(coefficient: FieldElement<F>, degree: usize) -> Self {
        if coefficient == FieldElement::zero() {
            Self::new()
        } else {
            let mut map = BTreeMap::new();
            map.insert(degree, coefficient);
            Self { coefficients: map }
        }
    }

    /// Returns the degree of the polynomial.
    ///
    /// The degree is the highest exponent with a non-zero coefficient.
    /// By convention, the zero polynomial has degree 0.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let poly = SparsePolynomial::from_coefficients(vec![
    ///     (0, FE::from(1)),
    ///     (100, FE::from(3)),
    /// ]);
    /// assert_eq!(poly.degree(), 100);
    /// ```
    #[inline]
    pub fn degree(&self) -> usize {
        // BTreeMap keys are ordered, so last_key_value gives the maximum
        self.coefficients
            .last_key_value()
            .map(|(deg, _)| *deg)
            .unwrap_or(0)
    }

    /// Returns true if the polynomial is the zero polynomial.
    ///
    /// # Example
    ///
    /// ```ignore
    /// assert!(SparsePolynomial::<F>::zero().is_zero());
    /// ```
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.coefficients.is_empty()
    }

    /// Returns the number of non-zero terms in the polynomial.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let poly = SparsePolynomial::from_coefficients(vec![
    ///     (0, FE::from(1)),
    ///     (50, FE::from(2)),
    ///     (100, FE::from(3)),
    /// ]);
    /// assert_eq!(poly.num_terms(), 3);
    /// ```
    #[inline]
    pub fn num_terms(&self) -> usize {
        self.coefficients.len()
    }

    /// Returns the coefficient at the given degree.
    ///
    /// Returns zero if no term exists at that degree.
    ///
    /// # Arguments
    ///
    /// * `degree` - The degree to query
    #[inline]
    pub fn get_coefficient(&self, degree: usize) -> FieldElement<F> {
        self.coefficients
            .get(&degree)
            .cloned()
            .unwrap_or_else(FieldElement::zero)
    }

    /// Returns the leading coefficient (coefficient of the highest degree term).
    ///
    /// Returns zero for the zero polynomial.
    #[inline]
    pub fn leading_coefficient(&self) -> FieldElement<F> {
        self.coefficients
            .last_key_value()
            .map(|(_, coeff)| coeff.clone())
            .unwrap_or_else(FieldElement::zero)
    }

    /// Evaluates the polynomial at a point x.
    ///
    /// Uses an efficient evaluation strategy that takes advantage of sparsity.
    /// For each non-zero term c_i * X^i, we compute x^i efficiently using
    /// binary exponentiation.
    ///
    /// # Arguments
    ///
    /// * `x` - The point at which to evaluate
    ///
    /// # Complexity
    ///
    /// O(k * log(d)) where k is the number of non-zero terms and d is the degree.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let poly = SparsePolynomial::from_coefficients(vec![
    ///     (0, FE::from(1)),
    ///     (2, FE::from(3)),
    /// ]);
    /// // P(X) = 3*X^2 + 1
    /// // P(2) = 3*4 + 1 = 13
    /// assert_eq!(poly.evaluate(&FE::from(2)), FE::from(13));
    /// ```
    pub fn evaluate(&self, x: &FieldElement<F>) -> FieldElement<F> {
        if self.is_zero() {
            return FieldElement::zero();
        }

        // For sparse polynomials, we compute each term separately
        // using efficient exponentiation
        let mut result = FieldElement::zero();

        for (&degree, coeff) in &self.coefficients {
            if degree == 0 {
                result = &result + coeff;
            } else {
                // x^degree * coeff
                let power = x.pow(degree as u64);
                result = &result + &(&power * coeff);
            }
        }

        result
    }

    /// Adds another sparse polynomial to this one.
    ///
    /// # Complexity
    ///
    /// O(k1 + k2) where k1, k2 are the number of non-zero terms.
    pub fn add(&self, other: &Self) -> Self {
        let mut result = self.coefficients.clone();
        let zero = FieldElement::zero();

        for (degree, coeff) in &other.coefficients {
            let entry = result.entry(*degree).or_insert_with(FieldElement::zero);
            *entry = &*entry + coeff;
            // Remove if it became zero
            if *entry == zero {
                result.remove(degree);
            }
        }

        Self {
            coefficients: result,
        }
    }

    /// Subtracts another sparse polynomial from this one.
    ///
    /// # Complexity
    ///
    /// O(k1 + k2) where k1, k2 are the number of non-zero terms.
    pub fn sub(&self, other: &Self) -> Self {
        let mut result = self.coefficients.clone();
        let zero = FieldElement::zero();

        for (degree, coeff) in &other.coefficients {
            let entry = result.entry(*degree).or_insert_with(FieldElement::zero);
            *entry = &*entry - coeff;
            // Remove if it became zero
            if *entry == zero {
                result.remove(degree);
            }
        }

        Self {
            coefficients: result,
        }
    }

    /// Multiplies this polynomial by another sparse polynomial.
    ///
    /// # Complexity
    ///
    /// O(k1 * k2) where k1, k2 are the number of non-zero terms.
    /// This is much better than the dense O(d1 * d2) when sparsity is high.
    pub fn mul(&self, other: &Self) -> Self {
        if self.is_zero() || other.is_zero() {
            return Self::zero();
        }

        let mut result: BTreeMap<usize, FieldElement<F>> = BTreeMap::new();
        let zero = FieldElement::zero();

        for (&deg1, coeff1) in &self.coefficients {
            for (&deg2, coeff2) in &other.coefficients {
                let new_degree = deg1 + deg2;
                let product = coeff1 * coeff2;
                let entry = result.entry(new_degree).or_insert_with(FieldElement::zero);
                *entry = &*entry + &product;
            }
        }

        // Remove any zero entries that resulted from cancellation
        result.retain(|_, v| *v != zero);

        Self {
            coefficients: result,
        }
    }

    /// Multiplies the polynomial by a scalar.
    ///
    /// # Arguments
    ///
    /// * `scalar` - The scalar to multiply by
    ///
    /// # Complexity
    ///
    /// O(k) where k is the number of non-zero terms.
    #[inline]
    pub fn mul_by_scalar(&self, scalar: &FieldElement<F>) -> Self {
        if *scalar == FieldElement::zero() {
            return Self::zero();
        }

        let coefficients = self
            .coefficients
            .iter()
            .map(|(&deg, coeff)| (deg, scalar * coeff))
            .collect();

        Self { coefficients }
    }

    /// Returns the negation of the polynomial.
    ///
    /// # Complexity
    ///
    /// O(k) where k is the number of non-zero terms.
    pub fn neg(&self) -> Self {
        let coefficients = self
            .coefficients
            .iter()
            .map(|(&deg, coeff)| (deg, -coeff))
            .collect();

        Self { coefficients }
    }

    /// Converts the sparse polynomial to a dense representation.
    ///
    /// # Warning
    ///
    /// This can use significant memory if the polynomial has high degree
    /// but few terms. Use only when necessary.
    ///
    /// # Complexity
    ///
    /// O(d) where d is the degree of the polynomial.
    pub fn to_dense(&self) -> Polynomial<FieldElement<F>> {
        if self.is_zero() {
            return Polynomial::zero();
        }

        let degree = self.degree();
        let mut coeffs = alloc::vec![FieldElement::zero(); degree + 1];

        for (&deg, coeff) in &self.coefficients {
            coeffs[deg] = coeff.clone();
        }

        Polynomial::new(&coeffs)
    }

    /// Creates a sparse polynomial from a dense polynomial.
    ///
    /// Zero coefficients are automatically skipped.
    ///
    /// # Arguments
    ///
    /// * `poly` - The dense polynomial to convert
    ///
    /// # Complexity
    ///
    /// O(d) where d is the degree of the polynomial.
    pub fn from_dense(poly: &Polynomial<FieldElement<F>>) -> Self {
        let zero = FieldElement::zero();
        let mut map = BTreeMap::new();

        for (degree, coeff) in poly.coefficients().iter().enumerate() {
            if *coeff != zero {
                map.insert(degree, coeff.clone());
            }
        }

        Self { coefficients: map }
    }

    /// Returns the sparsity of the polynomial as a fraction.
    ///
    /// Sparsity is defined as the fraction of zero coefficients,
    /// i.e., `1 - (num_terms / (degree + 1))`.
    ///
    /// A sparsity of 1.0 means the polynomial is zero.
    /// A sparsity of 0.0 means all coefficients are non-zero.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Polynomial with 3 non-zero terms out of 101 possible (degree 100)
    /// let poly = SparsePolynomial::from_coefficients(vec![
    ///     (0, FE::from(1)),
    ///     (50, FE::from(2)),
    ///     (100, FE::from(3)),
    /// ]);
    /// // sparsity = 1 - 3/101 ≈ 0.97
    /// ```
    pub fn sparsity(&self) -> f64 {
        if self.is_zero() {
            return 1.0;
        }

        let degree = self.degree();
        let num_terms = self.num_terms();

        1.0 - (num_terms as f64) / ((degree + 1) as f64)
    }

    /// Returns an iterator over the non-zero terms as (degree, coefficient) pairs.
    ///
    /// The iterator yields terms in ascending order of degree.
    pub fn iter(&self) -> impl Iterator<Item = (&usize, &FieldElement<F>)> {
        self.coefficients.iter()
    }

    /// Returns the underlying coefficient map.
    pub fn coefficients(&self) -> &BTreeMap<usize, FieldElement<F>> {
        &self.coefficients
    }
}

// Implement operator traits for ergonomic usage

impl<F: IsField> Add for SparsePolynomial<F> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        SparsePolynomial::add(&self, &rhs)
    }
}

impl<F: IsField> Add for &SparsePolynomial<F> {
    type Output = SparsePolynomial<F>;

    fn add(self, rhs: Self) -> Self::Output {
        SparsePolynomial::add(self, rhs)
    }
}

impl<F: IsField> Add<&SparsePolynomial<F>> for SparsePolynomial<F> {
    type Output = Self;

    fn add(self, rhs: &Self) -> Self::Output {
        SparsePolynomial::add(&self, rhs)
    }
}

impl<F: IsField> Add<SparsePolynomial<F>> for &SparsePolynomial<F> {
    type Output = SparsePolynomial<F>;

    fn add(self, rhs: SparsePolynomial<F>) -> Self::Output {
        SparsePolynomial::add(self, &rhs)
    }
}

impl<F: IsField> Sub for SparsePolynomial<F> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        SparsePolynomial::sub(&self, &rhs)
    }
}

impl<F: IsField> Sub for &SparsePolynomial<F> {
    type Output = SparsePolynomial<F>;

    fn sub(self, rhs: Self) -> Self::Output {
        SparsePolynomial::sub(self, rhs)
    }
}

impl<F: IsField> Sub<&SparsePolynomial<F>> for SparsePolynomial<F> {
    type Output = Self;

    fn sub(self, rhs: &Self) -> Self::Output {
        SparsePolynomial::sub(&self, rhs)
    }
}

impl<F: IsField> Sub<SparsePolynomial<F>> for &SparsePolynomial<F> {
    type Output = SparsePolynomial<F>;

    fn sub(self, rhs: SparsePolynomial<F>) -> Self::Output {
        SparsePolynomial::sub(self, &rhs)
    }
}

impl<F: IsField> Mul for SparsePolynomial<F> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        SparsePolynomial::mul(&self, &rhs)
    }
}

impl<F: IsField> Mul for &SparsePolynomial<F> {
    type Output = SparsePolynomial<F>;

    fn mul(self, rhs: Self) -> Self::Output {
        SparsePolynomial::mul(self, rhs)
    }
}

impl<F: IsField> Mul<&SparsePolynomial<F>> for SparsePolynomial<F> {
    type Output = Self;

    fn mul(self, rhs: &Self) -> Self::Output {
        SparsePolynomial::mul(&self, rhs)
    }
}

impl<F: IsField> Mul<SparsePolynomial<F>> for &SparsePolynomial<F> {
    type Output = SparsePolynomial<F>;

    fn mul(self, rhs: SparsePolynomial<F>) -> Self::Output {
        SparsePolynomial::mul(self, &rhs)
    }
}

impl<F: IsField> Neg for SparsePolynomial<F> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        SparsePolynomial::neg(&self)
    }
}

impl<F: IsField> Neg for &SparsePolynomial<F> {
    type Output = SparsePolynomial<F>;

    fn neg(self) -> Self::Output {
        SparsePolynomial::neg(self)
    }
}

// Scalar multiplication
impl<F: IsField> Mul<FieldElement<F>> for SparsePolynomial<F> {
    type Output = Self;

    fn mul(self, rhs: FieldElement<F>) -> Self::Output {
        self.mul_by_scalar(&rhs)
    }
}

impl<F: IsField> Mul<&FieldElement<F>> for &SparsePolynomial<F> {
    type Output = SparsePolynomial<F>;

    fn mul(self, rhs: &FieldElement<F>) -> Self::Output {
        self.mul_by_scalar(rhs)
    }
}

impl<F: IsField> Mul<SparsePolynomial<F>> for FieldElement<F> {
    type Output = SparsePolynomial<F>;

    fn mul(self, rhs: SparsePolynomial<F>) -> Self::Output {
        rhs.mul_by_scalar(&self)
    }
}

impl<F: IsField> Mul<&SparsePolynomial<F>> for &FieldElement<F> {
    type Output = SparsePolynomial<F>;

    fn mul(self, rhs: &SparsePolynomial<F>) -> Self::Output {
        rhs.mul_by_scalar(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::fields::u64_prime_field::U64PrimeField;

    const ORDER: u64 = 23;
    type F = U64PrimeField<ORDER>;
    type FE = FieldElement<F>;

    // ==================== Basic Construction Tests ====================

    #[test]
    fn test_new_creates_zero_polynomial() {
        let poly: SparsePolynomial<F> = SparsePolynomial::new();
        assert!(poly.is_zero());
        assert_eq!(poly.degree(), 0);
        assert_eq!(poly.num_terms(), 0);
    }

    #[test]
    fn test_zero_creates_zero_polynomial() {
        let poly: SparsePolynomial<F> = SparsePolynomial::zero();
        assert!(poly.is_zero());
    }

    #[test]
    fn test_default_creates_zero_polynomial() {
        let poly: SparsePolynomial<F> = SparsePolynomial::default();
        assert!(poly.is_zero());
    }

    #[test]
    fn test_from_coefficients_basic() {
        let poly = SparsePolynomial::from_coefficients(vec![
            (0, FE::from(1)),
            (2, FE::from(3)),
            (5, FE::from(7)),
        ]);
        assert_eq!(poly.degree(), 5);
        assert_eq!(poly.num_terms(), 3);
        assert_eq!(poly.get_coefficient(0), FE::from(1));
        assert_eq!(poly.get_coefficient(1), FE::zero());
        assert_eq!(poly.get_coefficient(2), FE::from(3));
        assert_eq!(poly.get_coefficient(5), FE::from(7));
    }

    #[test]
    fn test_from_coefficients_filters_zeros() {
        let poly = SparsePolynomial::from_coefficients(vec![
            (0, FE::from(1)),
            (2, FE::zero()),
            (5, FE::from(7)),
        ]);
        assert_eq!(poly.num_terms(), 2);
        assert!(!poly.coefficients().contains_key(&2));
    }

    #[test]
    fn test_new_monomial() {
        let mono = SparsePolynomial::new_monomial(FE::from(5), 3);
        assert_eq!(mono.degree(), 3);
        assert_eq!(mono.num_terms(), 1);
        assert_eq!(mono.get_coefficient(3), FE::from(5));
    }

    #[test]
    fn test_new_monomial_zero_coefficient() {
        let mono = SparsePolynomial::new_monomial(FE::zero(), 3);
        assert!(mono.is_zero());
    }

    // ==================== Degree and Leading Coefficient Tests ====================

    #[test]
    fn test_degree_of_zero_is_zero() {
        let poly: SparsePolynomial<F> = SparsePolynomial::zero();
        assert_eq!(poly.degree(), 0);
    }

    #[test]
    fn test_degree_of_constant() {
        let poly = SparsePolynomial::from_coefficients(vec![(0, FE::from(5))]);
        assert_eq!(poly.degree(), 0);
    }

    #[test]
    fn test_leading_coefficient() {
        let poly = SparsePolynomial::from_coefficients(vec![
            (0, FE::from(1)),
            (3, FE::from(5)),
            (7, FE::from(11)),
        ]);
        assert_eq!(poly.leading_coefficient(), FE::from(11));
    }

    #[test]
    fn test_leading_coefficient_of_zero() {
        let poly: SparsePolynomial<F> = SparsePolynomial::zero();
        assert_eq!(poly.leading_coefficient(), FE::zero());
    }

    // ==================== Evaluation Tests ====================

    #[test]
    fn test_evaluate_zero_polynomial() {
        let poly: SparsePolynomial<F> = SparsePolynomial::zero();
        assert_eq!(poly.evaluate(&FE::from(5)), FE::zero());
    }

    #[test]
    fn test_evaluate_constant_polynomial() {
        let poly = SparsePolynomial::from_coefficients(vec![(0, FE::from(7))]);
        assert_eq!(poly.evaluate(&FE::from(5)), FE::from(7));
        assert_eq!(poly.evaluate(&FE::from(0)), FE::from(7));
    }

    #[test]
    fn test_evaluate_linear_polynomial() {
        // P(X) = 2*X + 3
        let poly = SparsePolynomial::from_coefficients(vec![(0, FE::from(3)), (1, FE::from(2))]);
        // P(4) = 2*4 + 3 = 11
        assert_eq!(poly.evaluate(&FE::from(4)), FE::from(11));
    }

    #[test]
    fn test_evaluate_quadratic_polynomial() {
        // P(X) = 3*X^2 + 2*X + 1
        let poly = SparsePolynomial::from_coefficients(vec![
            (0, FE::from(1)),
            (1, FE::from(2)),
            (2, FE::from(3)),
        ]);
        // P(2) = 3*4 + 2*2 + 1 = 12 + 4 + 1 = 17
        assert_eq!(poly.evaluate(&FE::from(2)), FE::from(17));
    }

    #[test]
    fn test_evaluate_sparse_polynomial() {
        // P(X) = X^100 + 1 (very sparse!)
        let poly = SparsePolynomial::from_coefficients(vec![(0, FE::from(1)), (100, FE::from(1))]);
        // P(1) = 1 + 1 = 2
        assert_eq!(poly.evaluate(&FE::from(1)), FE::from(2));
    }

    #[test]
    fn test_evaluate_at_zero() {
        // P(X) = 5*X^3 + 3*X + 7
        let poly = SparsePolynomial::from_coefficients(vec![
            (0, FE::from(7)),
            (1, FE::from(3)),
            (3, FE::from(5)),
        ]);
        // P(0) = 7
        assert_eq!(poly.evaluate(&FE::from(0)), FE::from(7));
    }

    // ==================== Addition Tests ====================

    #[test]
    fn test_add_zero_polynomials() {
        let zero: SparsePolynomial<F> = SparsePolynomial::zero();
        let result = &zero + &zero;
        assert!(result.is_zero());
    }

    #[test]
    fn test_add_to_zero() {
        let poly = SparsePolynomial::from_coefficients(vec![(0, FE::from(5)), (2, FE::from(3))]);
        let zero: SparsePolynomial<F> = SparsePolynomial::zero();

        let result1 = &poly + &zero;
        let result2 = &zero + &poly;

        assert_eq!(result1, poly);
        assert_eq!(result2, poly);
    }

    #[test]
    fn test_add_basic() {
        // P(X) = 2*X + 1
        let p1 = SparsePolynomial::from_coefficients(vec![(0, FE::from(1)), (1, FE::from(2))]);
        // Q(X) = 3*X + 4
        let p2 = SparsePolynomial::from_coefficients(vec![(0, FE::from(4)), (1, FE::from(3))]);
        // P + Q = 5*X + 5
        let result = &p1 + &p2;

        assert_eq!(result.get_coefficient(0), FE::from(5));
        assert_eq!(result.get_coefficient(1), FE::from(5));
    }

    #[test]
    fn test_add_with_cancellation() {
        // P(X) = X + 1
        let p1 = SparsePolynomial::from_coefficients(vec![(0, FE::from(1)), (1, FE::from(1))]);
        // Q(X) = -X + 2 = (ORDER-1)*X + 2
        let p2 =
            SparsePolynomial::from_coefficients(vec![(0, FE::from(2)), (1, FE::from(ORDER - 1))]);
        // P + Q = 3 (X terms cancel)
        let result = &p1 + &p2;

        assert_eq!(result.num_terms(), 1);
        assert_eq!(result.get_coefficient(0), FE::from(3));
        assert_eq!(result.get_coefficient(1), FE::zero());
    }

    #[test]
    fn test_add_different_degrees() {
        // P(X) = X^2 + 1
        let p1 = SparsePolynomial::from_coefficients(vec![(0, FE::from(1)), (2, FE::from(1))]);
        // Q(X) = X^5 + X
        let p2 = SparsePolynomial::from_coefficients(vec![(1, FE::from(1)), (5, FE::from(1))]);
        // P + Q = X^5 + X^2 + X + 1
        let result = &p1 + &p2;

        assert_eq!(result.degree(), 5);
        assert_eq!(result.num_terms(), 4);
    }

    // ==================== Subtraction Tests ====================

    #[test]
    fn test_sub_self_gives_zero() {
        let poly = SparsePolynomial::from_coefficients(vec![
            (0, FE::from(1)),
            (3, FE::from(5)),
            (7, FE::from(11)),
        ]);
        let result = &poly - &poly;
        assert!(result.is_zero());
    }

    #[test]
    fn test_sub_zero() {
        let poly = SparsePolynomial::from_coefficients(vec![(0, FE::from(5)), (2, FE::from(3))]);
        let zero: SparsePolynomial<F> = SparsePolynomial::zero();

        let result = &poly - &zero;
        assert_eq!(result, poly);
    }

    #[test]
    fn test_sub_basic() {
        // P(X) = 5*X + 7
        let p1 = SparsePolynomial::from_coefficients(vec![(0, FE::from(7)), (1, FE::from(5))]);
        // Q(X) = 2*X + 3
        let p2 = SparsePolynomial::from_coefficients(vec![(0, FE::from(3)), (1, FE::from(2))]);
        // P - Q = 3*X + 4
        let result = &p1 - &p2;

        assert_eq!(result.get_coefficient(0), FE::from(4));
        assert_eq!(result.get_coefficient(1), FE::from(3));
    }

    // ==================== Multiplication Tests ====================

    #[test]
    fn test_mul_by_zero() {
        let poly = SparsePolynomial::from_coefficients(vec![(0, FE::from(5)), (2, FE::from(3))]);
        let zero: SparsePolynomial<F> = SparsePolynomial::zero();

        let result1 = &poly * &zero;
        let result2 = &zero * &poly;

        assert!(result1.is_zero());
        assert!(result2.is_zero());
    }

    #[test]
    fn test_mul_by_one() {
        let poly = SparsePolynomial::from_coefficients(vec![(0, FE::from(5)), (2, FE::from(3))]);
        let one = SparsePolynomial::from_coefficients(vec![(0, FE::from(1))]);

        let result = &poly * &one;
        assert_eq!(result, poly);
    }

    #[test]
    fn test_mul_monomials() {
        // (3*X^2) * (5*X^3) = 15*X^5
        let m1 = SparsePolynomial::new_monomial(FE::from(3), 2);
        let m2 = SparsePolynomial::new_monomial(FE::from(5), 3);
        let result = &m1 * &m2;

        assert_eq!(result.num_terms(), 1);
        assert_eq!(result.degree(), 5);
        assert_eq!(result.get_coefficient(5), FE::from(15));
    }

    #[test]
    fn test_mul_basic() {
        // (X + 1) * (X + 2) = X^2 + 3*X + 2
        let p1 = SparsePolynomial::from_coefficients(vec![(0, FE::from(1)), (1, FE::from(1))]);
        let p2 = SparsePolynomial::from_coefficients(vec![(0, FE::from(2)), (1, FE::from(1))]);
        let result = &p1 * &p2;

        assert_eq!(result.get_coefficient(0), FE::from(2));
        assert_eq!(result.get_coefficient(1), FE::from(3));
        assert_eq!(result.get_coefficient(2), FE::from(1));
    }

    #[test]
    fn test_mul_sparse() {
        // (X^10 + 1) * (X^20 + 1) = X^30 + X^20 + X^10 + 1
        let p1 = SparsePolynomial::from_coefficients(vec![(0, FE::from(1)), (10, FE::from(1))]);
        let p2 = SparsePolynomial::from_coefficients(vec![(0, FE::from(1)), (20, FE::from(1))]);
        let result = &p1 * &p2;

        assert_eq!(result.degree(), 30);
        assert_eq!(result.num_terms(), 4);
        assert_eq!(result.get_coefficient(0), FE::from(1));
        assert_eq!(result.get_coefficient(10), FE::from(1));
        assert_eq!(result.get_coefficient(20), FE::from(1));
        assert_eq!(result.get_coefficient(30), FE::from(1));
    }

    // ==================== Scalar Multiplication Tests ====================

    #[test]
    fn test_mul_by_scalar_zero() {
        let poly = SparsePolynomial::from_coefficients(vec![(0, FE::from(5)), (2, FE::from(3))]);
        let result = poly.mul_by_scalar(&FE::zero());
        assert!(result.is_zero());
    }

    #[test]
    fn test_mul_by_scalar_one() {
        let poly = SparsePolynomial::from_coefficients(vec![(0, FE::from(5)), (2, FE::from(3))]);
        let result = poly.mul_by_scalar(&FE::from(1));
        assert_eq!(result, poly);
    }

    #[test]
    fn test_mul_by_scalar_basic() {
        // 2 * (3*X + 5) = 6*X + 10
        let poly = SparsePolynomial::from_coefficients(vec![(0, FE::from(5)), (1, FE::from(3))]);
        let result = poly.mul_by_scalar(&FE::from(2));

        assert_eq!(result.get_coefficient(0), FE::from(10));
        assert_eq!(result.get_coefficient(1), FE::from(6));
    }

    // ==================== Negation Tests ====================

    #[test]
    fn test_neg_zero() {
        let zero: SparsePolynomial<F> = SparsePolynomial::zero();
        let result = -&zero;
        assert!(result.is_zero());
    }

    #[test]
    fn test_neg_basic() {
        let poly = SparsePolynomial::from_coefficients(vec![(0, FE::from(5)), (1, FE::from(3))]);
        let neg_poly = -&poly;

        assert_eq!(neg_poly.get_coefficient(0), FE::from(ORDER - 5));
        assert_eq!(neg_poly.get_coefficient(1), FE::from(ORDER - 3));
    }

    #[test]
    fn test_double_negation() {
        let poly = SparsePolynomial::from_coefficients(vec![(0, FE::from(5)), (1, FE::from(3))]);
        let result = -(-&poly);
        assert_eq!(result, poly);
    }

    // ==================== Conversion Tests ====================

    #[test]
    fn test_to_dense_zero() {
        let sparse: SparsePolynomial<F> = SparsePolynomial::zero();
        let dense = sparse.to_dense();
        assert!(dense.is_zero());
    }

    #[test]
    fn test_to_dense_basic() {
        // 3*X^2 + 2*X + 1
        let sparse = SparsePolynomial::from_coefficients(vec![
            (0, FE::from(1)),
            (1, FE::from(2)),
            (2, FE::from(3)),
        ]);
        let dense = sparse.to_dense();

        assert_eq!(dense.coefficients().len(), 3);
        assert_eq!(dense.coefficients()[0], FE::from(1));
        assert_eq!(dense.coefficients()[1], FE::from(2));
        assert_eq!(dense.coefficients()[2], FE::from(3));
    }

    #[test]
    fn test_from_dense_basic() {
        let dense = Polynomial::new(&[FE::from(1), FE::from(2), FE::from(3)]);
        let sparse = SparsePolynomial::from_dense(&dense);

        assert_eq!(sparse.num_terms(), 3);
        assert_eq!(sparse.get_coefficient(0), FE::from(1));
        assert_eq!(sparse.get_coefficient(1), FE::from(2));
        assert_eq!(sparse.get_coefficient(2), FE::from(3));
    }

    #[test]
    fn test_from_dense_with_zeros() {
        // Dense polynomial with internal zeros: 5*X^3 + 0*X^2 + 0*X + 1
        let dense = Polynomial::new(&[FE::from(1), FE::zero(), FE::zero(), FE::from(5)]);
        let sparse = SparsePolynomial::from_dense(&dense);

        assert_eq!(sparse.num_terms(), 2);
        assert!(sparse.coefficients().contains_key(&0));
        assert!(sparse.coefficients().contains_key(&3));
        assert!(!sparse.coefficients().contains_key(&1));
        assert!(!sparse.coefficients().contains_key(&2));
    }

    #[test]
    fn test_roundtrip_sparse_to_dense_to_sparse() {
        let original = SparsePolynomial::from_coefficients(vec![
            (0, FE::from(1)),
            (5, FE::from(7)),
            (10, FE::from(13)),
        ]);

        let dense = original.to_dense();
        let back_to_sparse = SparsePolynomial::from_dense(&dense);

        assert_eq!(back_to_sparse, original);
    }

    #[test]
    fn test_roundtrip_dense_to_sparse_to_dense() {
        let original = Polynomial::new(&[FE::from(1), FE::from(2), FE::from(3)]);
        let sparse = SparsePolynomial::from_dense(&original);
        let back_to_dense = sparse.to_dense();

        assert_eq!(back_to_dense, original);
    }

    // ==================== Sparsity Tests ====================

    #[test]
    fn test_sparsity_zero_polynomial() {
        let poly: SparsePolynomial<F> = SparsePolynomial::zero();
        assert_eq!(poly.sparsity(), 1.0);
    }

    #[test]
    fn test_sparsity_dense_polynomial() {
        // All coefficients non-zero: 1 + X + X^2
        let poly = SparsePolynomial::from_coefficients(vec![
            (0, FE::from(1)),
            (1, FE::from(1)),
            (2, FE::from(1)),
        ]);
        // sparsity = 1 - 3/3 = 0.0
        assert_eq!(poly.sparsity(), 0.0);
    }

    #[test]
    fn test_sparsity_sparse_polynomial() {
        // Only 2 non-zero terms out of 101: 1 + X^100
        let poly = SparsePolynomial::from_coefficients(vec![(0, FE::from(1)), (100, FE::from(1))]);
        // sparsity = 1 - 2/101 ≈ 0.98
        let sparsity = poly.sparsity();
        assert!(sparsity > 0.98);
        assert!(sparsity < 0.99);
    }

    // ==================== Evaluation Consistency Tests ====================

    #[test]
    fn test_evaluation_matches_dense() {
        let sparse = SparsePolynomial::from_coefficients(vec![
            (0, FE::from(3)),
            (2, FE::from(7)),
            (5, FE::from(11)),
        ]);
        let dense = sparse.to_dense();

        // Test at multiple points
        for i in 0..10 {
            let x = FE::from(i);
            assert_eq!(sparse.evaluate(&x), dense.evaluate(&x));
        }
    }

    // ==================== Operator Trait Tests ====================

    #[test]
    fn test_add_operator_owned() {
        let p1 = SparsePolynomial::from_coefficients(vec![(0, FE::from(1))]);
        let p2 = SparsePolynomial::from_coefficients(vec![(0, FE::from(2))]);
        let result = p1 + p2;
        assert_eq!(result.get_coefficient(0), FE::from(3));
    }

    #[test]
    fn test_sub_operator_owned() {
        let p1 = SparsePolynomial::from_coefficients(vec![(0, FE::from(5))]);
        let p2 = SparsePolynomial::from_coefficients(vec![(0, FE::from(2))]);
        let result = p1 - p2;
        assert_eq!(result.get_coefficient(0), FE::from(3));
    }

    #[test]
    fn test_mul_operator_owned() {
        let p1 = SparsePolynomial::from_coefficients(vec![(0, FE::from(2))]);
        let p2 = SparsePolynomial::from_coefficients(vec![(0, FE::from(3))]);
        let result = p1 * p2;
        assert_eq!(result.get_coefficient(0), FE::from(6));
    }

    #[test]
    fn test_neg_operator_owned() {
        let poly = SparsePolynomial::from_coefficients(vec![(0, FE::from(5))]);
        let result = -poly;
        assert_eq!(result.get_coefficient(0), FE::from(ORDER - 5));
    }

    #[test]
    fn test_scalar_mul_operator() {
        let poly = SparsePolynomial::from_coefficients(vec![(0, FE::from(5))]);
        let scalar = FE::from(3);

        let result1 = poly.clone() * scalar;
        let result2 = scalar * poly;

        assert_eq!(result1.get_coefficient(0), FE::from(15));
        assert_eq!(result2.get_coefficient(0), FE::from(15));
    }

    // ==================== Edge Cases ====================

    #[test]
    fn test_single_term() {
        let poly = SparsePolynomial::new_monomial(FE::from(7), 0);
        assert_eq!(poly.degree(), 0);
        assert_eq!(poly.num_terms(), 1);
        assert_eq!(poly.evaluate(&FE::from(100)), FE::from(7));
    }

    #[test]
    fn test_high_degree_single_term() {
        let poly = SparsePolynomial::new_monomial(FE::from(1), 1000);
        assert_eq!(poly.degree(), 1000);
        assert_eq!(poly.num_terms(), 1);
        // 1^1000 = 1
        assert_eq!(poly.evaluate(&FE::from(1)), FE::from(1));
    }

    #[test]
    fn test_iterator() {
        let poly = SparsePolynomial::from_coefficients(vec![
            (5, FE::from(7)),
            (0, FE::from(1)),
            (2, FE::from(3)),
        ]);

        let terms: Vec<_> = poly.iter().collect();
        // Should be in ascending order of degree
        assert_eq!(terms.len(), 3);
        assert_eq!(*terms[0].0, 0);
        assert_eq!(*terms[1].0, 2);
        assert_eq!(*terms[2].0, 5);
    }
}
