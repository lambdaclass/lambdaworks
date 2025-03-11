//! Binary field and tower field implementations.
//!
//! This module provides implementations for:
//! 1. Basic binary fields GF(2ⁿ) with irreducible polynomial reduction
//! 2. Tower field extensions that support splitting and joining operations

use crate::field::{element::FieldElement, errors::FieldError, traits::IsField};
use core::fmt;
use core::marker::PhantomData;
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// Represents a binary field configuration for GF(2ⁿ).
///
/// This trait defines the parameters needed for a binary field:
/// - The degree `n` of the field extension (GF(2ⁿ))
/// - The primitive polynomial that defines the field structure
pub trait BinaryFieldConfig: Clone + Copy + fmt::Debug {
    /// Degree of the field extension (n in GF(2ⁿ))
    const DEGREE: u32;

    /// Primitive polynomial represented as bits
    /// Example: x³ + x + 1 is represented as 0b1011
    const PRIMITIVE_POLY: u128;
}

/// Binary field structure parametrized by a configuration.
#[derive(Clone, Debug)]
pub struct BinaryField<C: BinaryFieldConfig>(PhantomData<C>);

impl<C: BinaryFieldConfig> IsField for BinaryField<C> {
    type BaseType = u64;

    fn add(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        a ^ b
    }

    fn mul(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        let mut result = 0;
        let mut a_val = *a;
        let mut b_val = *b;
        while b_val != 0 {
            if b_val & 1 != 0 {
                result ^= a_val;
            }
            b_val >>= 1;
            a_val <<= 1;
            // When a overflows DEGREE bits, reduce it using the irreducible polynomial.
            if a_val & (1 << C::DEGREE) != 0 {
                a_val ^= C::PRIMITIVE_POLY as u64;
            }
        }
        result
    }

    fn sub(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        // In characteristic 2, subtraction equals addition
        Self::add(a, b)
    }

    fn neg(a: &Self::BaseType) -> Self::BaseType {
        // In characteristic 2, negation is identity
        *a
    }

    fn inv(a: &Self::BaseType) -> Result<Self::BaseType, FieldError> {
        if *a == 0 {
            return Err(FieldError::InvZeroError);
        }
        let mut t = 0u64;
        let mut newt = 1u64;
        let mut r = C::PRIMITIVE_POLY as u64;
        let mut newr = *a;

        while newr != 0 {
            let deg_r = 63 - r.leading_zeros();
            let deg_newr = 63 - newr.leading_zeros();
            if deg_r < deg_newr {
                core::mem::swap(&mut t, &mut newt);
                core::mem::swap(&mut r, &mut newr);
                continue;
            }
            if let Some(shift) = deg_r.checked_sub(deg_newr) {
                r ^= newr << shift;
                t ^= newt << shift;
            } else {
                core::mem::swap(&mut t, &mut newt);
                core::mem::swap(&mut r, &mut newr);
            }
        }

        if r != 1 {
            return Err(FieldError::InvZeroError);
        }

        Ok(t)
    }

    fn div(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        let b_inv = Self::inv(b).expect("Division by zero");
        Self::mul(a, &b_inv)
    }

    fn eq(a: &Self::BaseType, b: &Self::BaseType) -> bool {
        a == b
    }

    fn zero() -> Self::BaseType {
        0
    }

    fn one() -> Self::BaseType {
        1
    }

    fn from_u64(x: u64) -> Self::BaseType {
        x
    }

    fn from_base_type(x: Self::BaseType) -> Self::BaseType {
        x % (1 << C::DEGREE) as u64
    }
}

/// A binary field element in GF(2ⁿ).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BinaryFieldElement<F: BinaryFieldConfig> {
    /// The underlying value, only lower DEGREE bits are significant
    value: u128,
    /// Phantom data to keep track of the field configuration
    _phantom: PhantomData<F>,
}

impl<F: BinaryFieldConfig> Default for BinaryFieldElement<F> {
    fn default() -> Self {
        Self::new(0)
    }
}

impl<F: BinaryFieldConfig> BinaryFieldElement<F> {
    /// Creates a new binary field element
    #[inline]
    pub fn new(value: u128) -> Self {
        let mask = (1 << F::DEGREE) - 1;
        Self {
            value: value & mask as u128,
            _phantom: PhantomData,
        }
    }

    /// Returns true if this element is zero
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.value == 0
    }

    /// Returns true if this element is one
    #[inline]
    pub fn is_one(&self) -> bool {
        self.value == 1
    }

    /// Returns the underlying value
    #[inline]
    pub fn value(&self) -> u128 {
        self.value
    }

    /// Lookup table for multiplicative inverses in GF(2⁴)
    const F2_4_INVERSE: [u128; 15] = [1, 9, 14, 13, 11, 7, 6, 15, 2, 12, 5, 4, 10, 3, 8];

    // Multiplication table for GF(2²)
    const F2_2_MUL: [[u128; 4]; 4] = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]];

    // Multiplication table for GF(2³)
    const F2_3_MUL: [[u128; 8]; 8] = [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 2, 3, 4, 5, 6, 7],
        [0, 2, 4, 6, 3, 1, 7, 5],
        [0, 3, 6, 5, 7, 4, 1, 2],
        [0, 4, 3, 7, 6, 2, 5, 1],
        [0, 5, 1, 4, 2, 7, 3, 6],
        [0, 6, 7, 1, 5, 3, 2, 4],
        [0, 7, 5, 2, 1, 6, 4, 3],
    ];

    // Multiplication table for GF(2⁴)
    const F2_4_MUL: [[u128; 16]; 16] = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        [0, 2, 4, 6, 8, 10, 12, 14, 3, 1, 7, 5, 11, 9, 15, 13],
        [0, 3, 6, 5, 12, 15, 10, 9, 11, 8, 13, 14, 7, 4, 1, 2],
        [0, 4, 8, 12, 3, 7, 11, 15, 6, 2, 14, 10, 5, 1, 13, 9],
        [0, 5, 10, 15, 7, 2, 13, 8, 14, 11, 4, 1, 9, 12, 3, 6],
        [0, 6, 12, 10, 11, 13, 7, 1, 5, 3, 9, 15, 14, 8, 2, 4],
        [0, 7, 14, 9, 15, 8, 1, 6, 13, 10, 3, 4, 2, 5, 12, 11],
        [0, 8, 3, 11, 6, 14, 5, 13, 12, 4, 15, 7, 10, 2, 9, 1],
        [0, 9, 1, 8, 2, 11, 3, 10, 4, 13, 5, 12, 6, 15, 7, 14],
        [0, 10, 7, 13, 14, 4, 9, 3, 15, 5, 8, 2, 1, 11, 6, 12],
        [0, 11, 5, 14, 10, 1, 15, 4, 7, 12, 2, 9, 13, 6, 8, 3],
        [0, 12, 11, 7, 5, 9, 14, 2, 10, 6, 1, 13, 15, 3, 4, 8],
        [0, 13, 9, 4, 1, 12, 8, 5, 2, 15, 11, 6, 3, 14, 10, 7],
        [0, 14, 15, 1, 13, 3, 2, 12, 9, 7, 6, 8, 4, 10, 11, 5],
        [0, 15, 13, 2, 9, 6, 4, 11, 1, 14, 12, 3, 8, 7, 5, 10],
    ];

    /// Performs multiplication using lookup tables for small fields (≤ 4 bits)
    /// Falls back to polynomial multiplication for larger fields
    #[inline]
    fn mul_small(&self, other: &Self) -> Self {
        match F::DEGREE {
            2 => Self::new(Self::F2_2_MUL[self.value as usize][other.value as usize]),
            3 => Self::new(Self::F2_3_MUL[self.value as usize][other.value as usize]),
            4 => Self::new(Self::F2_4_MUL[self.value as usize][other.value as usize]),
            _ => {
                let result = Self::mul_internal(self.value, other.value);
                Self::new(result)
            }
        }
    }

    /// Computes the multiplicative inverse of this element
    /// Returns None if the element is zero (as zero has no multiplicative inverse)
    #[inline]
    pub fn inverse(&self) -> Option<Self> {
        if self.is_zero() {
            return None;
        }

        // Use Fermat's little theorem:
        // In GF(2^n), x^(2^n-1) = 1 for any non-zero x
        // Therefore x^(2^n-2) is the multiplicative inverse
        Some(self.pow((1 << F::DEGREE) - 2))
    }

    /// Splits the element into high and low parts
    /// Used in the recursive inverse computation
    /// Returns (high_part, low_part) where each part has half the bits
    fn split(&self) -> (Self, Self) {
        let half_degree = F::DEGREE / 2;
        let mask = (1 << half_degree) - 1;
        let lo = self.value & mask;
        let hi = (self.value >> half_degree) & mask;
        (Self::new(hi), Self::new(lo))
    }

    /// Joins high and low parts into a single element
    /// Used in the recursive inverse computation
    /// The high part becomes the most significant bits
    fn join(&self, low: &Self) -> Self {
        let half_degree = F::DEGREE / 2;
        Self::new((self.value << half_degree) | low.value)
    }

    /// Raises this element to the power of exp
    #[inline]
    pub fn pow(&self, exp: u32) -> Self {
        if exp == 0 {
            return Self::new(1);
        }
        if self.is_zero() {
            return self.clone();
        }

        let mut result = Self::new(1);
        let mut base = self.clone();
        let mut exp_val = exp;

        while exp_val > 0 {
            if exp_val & 1 == 1 {
                result = result * base.clone();
            }
            base = base.clone() * base;
            exp_val >>= 1;
        }

        result
    }

    /// Performs carry-less multiplication with reduction
    #[inline]
    fn mul_internal(mut a: u128, mut b: u128) -> u128 {
        let mut result = 0;
        let mask = (1 << F::DEGREE) - 1;

        // Ensure inputs are within field size
        a &= mask;
        b &= mask;

        while b != 0 {
            if b & 1 != 0 {
                result ^= a;
            }
            b >>= 1;
            a <<= 1;
            // When a overflows DEGREE bits, reduce it using the primitive polynomial
            if a & (1 << F::DEGREE) != 0 {
                a ^= F::PRIMITIVE_POLY;
            }
            // Keep a within field size
            a &= mask;
        }
        // Final reduction to ensure result is in field
        result & mask
    }
}

impl<F: BinaryFieldConfig> Add for BinaryFieldElement<F> {
    type Output = Self;

    #[inline]
    fn add(self, other: Self) -> Self {
        Self::new(self.value ^ other.value)
    }
}

impl<F: BinaryFieldConfig> AddAssign for BinaryFieldElement<F> {
    #[inline]
    fn add_assign(&mut self, other: Self) {
        self.value ^= other.value;
        // Ensure the value stays within range (though XOR won't increase bit length)
        let mask = (1 << F::DEGREE) - 1;
        self.value &= mask as u128;
    }
}

impl<F: BinaryFieldConfig> Sub for BinaryFieldElement<F> {
    type Output = Self;

    #[inline]
    fn sub(self, other: Self) -> Self {
        // In characteristic 2, subtraction equals addition
        self + other
    }
}

impl<F: BinaryFieldConfig> SubAssign for BinaryFieldElement<F> {
    #[inline]
    fn sub_assign(&mut self, other: Self) {
        // In characteristic 2, subtraction equals addition
        *self += other;
    }
}

impl<F: BinaryFieldConfig> Mul for BinaryFieldElement<F> {
    type Output = Self;

    #[inline]
    fn mul(self, other: Self) -> Self {
        // Always use polynomial multiplication for GF(2³)
        Self::new(Self::mul_internal(self.value, other.value))
    }
}

impl<F: BinaryFieldConfig> MulAssign for BinaryFieldElement<F> {
    #[inline]
    fn mul_assign(&mut self, other: Self) {
        *self = self.clone() * other;
    }
}

impl<F: BinaryFieldConfig> Neg for BinaryFieldElement<F> {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        // In characteristic 2, negation is identity
        self
    }
}

// Implement From traits for common integer types
impl<F: BinaryFieldConfig> From<u64> for BinaryFieldElement<F> {
    fn from(value: u64) -> Self {
        Self::new(value as u128)
    }
}

impl<F: BinaryFieldConfig> From<u32> for BinaryFieldElement<F> {
    fn from(value: u32) -> Self {
        Self::new(value as u128)
    }
}

impl<F: BinaryFieldConfig> From<u16> for BinaryFieldElement<F> {
    fn from(value: u16) -> Self {
        Self::new(value as u128)
    }
}

impl<F: BinaryFieldConfig> From<u8> for BinaryFieldElement<F> {
    fn from(value: u8) -> Self {
        Self::new(value as u128)
    }
}

// Example configuration for GF(2³)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GF2_3;
impl BinaryFieldConfig for GF2_3 {
    const DEGREE: u32 = 3;
    const PRIMITIVE_POLY: u128 = 0b1011; // x³ + x + 1
}

pub type GF2_3Field = BinaryField<GF2_3>;
pub type GF2_3Element = FieldElement<GF2_3Field>;

/// Tower Field Extension
///
/// The TowerField trait extends the binary field concept by adding metadata
/// about how many "levels" the element has (which determines its bit‐length),
/// and provides functions to split the element into two halves and to join them back.
///
/// This trait is particularly useful for implementing field extensions and
/// efficient arithmetic operations in larger fields.
pub trait TowerField: Sized + Copy + Clone + PartialEq + fmt::Debug {
    /// Create a new tower field element from a u128 value and an optional number of levels.
    /// If num_levels is None, a default of 3 levels (8 bits) is used.
    fn new(val: u128, num_levels: Option<usize>) -> Self;

    /// Returns the underlying value.
    fn get_val(&self) -> u128;

    /// Returns the number of levels.
    /// The number of bits is 2^num_levels.
    fn num_levels(&self) -> usize;

    /// Returns the total number of bits (typically, 1 << num_levels).
    fn num_bits(&self) -> usize;

    /// Returns a zero-padded binary string representation.
    /// The string length is equal to num_bits.
    fn bin(&self) -> String;

    /// Splits the element into two halves (high and low), each with half the bits.
    ///
    /// Example:
    /// For an 8-bit element (3 levels), this returns two 4-bit elements (2 levels).
    /// The high part contains the most significant 4 bits, and the low part
    /// contains the least significant 4 bits.
    fn split(&self) -> (Self, Self);

    /// Joins a high part and a low part into one element, increasing the level by one.
    ///
    /// Example:
    /// Given two 4-bit elements (2 levels), this returns an 8-bit element (3 levels)
    /// where the high part is placed in the most significant 4 bits and the low part
    /// in the least significant 4 bits.
    fn join(&self, low: &Self) -> Self;

    /// Basic addition (XOR).
    /// This operation is commutative and associative.
    fn add(&self, other: &Self) -> Self;

    /// Basic multiplication (naïve carry-less multiplication with reduction).
    /// The reduction is performed modulo x^(num_bits) + 1.
    /// This operation is commutative and associative.
    fn mul(&self, other: &Self) -> Self;
}

/// A tower field element with hierarchical structure.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct TowerFieldElement {
    /// The underlying value
    value: u128,
    /// Number of levels in the tower
    num_levels: usize,
    /// Number of bits (2^num_levels)
    num_bits: usize,
}

impl Default for TowerFieldElement {
    fn default() -> Self {
        Self::new(0, None)
    }
}

impl TowerFieldElement {
    /// Creates a new tower field element
    pub fn new(value: u128, num_levels: Option<usize>) -> Self {
        let levels = num_levels.unwrap_or(3);
        let bits = 1 << levels;
        let mask = if bits >= 128 {
            u128::MAX
        } else {
            (1 << bits) - 1
        };
        Self {
            value: value & mask,
            num_levels: levels,
            num_bits: bits,
        }
    }

    /// Returns true if this element is zero
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.value == 0
    }

    /// Returns true if this element is one
    #[inline]
    pub fn is_one(&self) -> bool {
        self.value == 1
    }

    /// Returns the underlying value
    #[inline]
    pub fn value(&self) -> u128 {
        self.value
    }

    /// Returns number of levels in the tower
    #[inline]
    pub fn num_levels(&self) -> usize {
        self.num_levels
    }

    /// Returns number of bits (2^num_levels)
    #[inline]
    pub fn num_bits(&self) -> usize {
        self.num_bits
    }

    /// Returns binary string representation
    pub fn to_binary_string(&self) -> String {
        format!("{:0width$b}", self.value, width = self.num_bits)
    }

    /// Splits element into high and low parts
    pub fn split(&self) -> (Self, Self) {
        let half_bits = self.num_bits / 2;
        let mask = (1 << half_bits) - 1;
        let lo = self.value & mask;
        let hi = (self.value >> half_bits) & mask;

        let new_levels = self.num_levels - 1;
        (
            Self::new(hi, Some(new_levels)),
            Self::new(lo, Some(new_levels)),
        )
    }

    /// Joins with another element as the low part
    pub fn join(&self, low: &Self) -> Self {
        let new_bits = self.num_bits * 2;
        let joined = (self.value << self.num_bits) | low.value;
        Self {
            value: joined,
            num_levels: self.num_levels + 1,
            num_bits: new_bits,
        }
    }

    /// Computes the multiplicative inverse of this element
    pub fn inverse(&self) -> Option<Self> {
        if self.is_zero() {
            return None;
        }

        // Use Fermat's little theorem:
        // In GF(2^n), x^(2^n-1) = 1 for any non-zero x
        // Therefore x^(2^n-2) is the multiplicative inverse
        Some(self.pow((1 << self.num_bits) - 2))
    }

    /// Raises this element to the power of exp
    pub fn pow(&self, exp: u32) -> Self {
        if exp == 0 {
            return Self::new(1, Some(self.num_levels));
        }
        if self.is_zero() {
            return self.clone();
        }

        let mut result = Self::new(1, Some(self.num_levels));
        let mut base = self.clone();
        let mut exp_val = exp;

        while exp_val > 0 {
            if exp_val & 1 == 1 {
                result = result * base.clone();
            }
            base = base.clone() * base;
            exp_val >>= 1;
        }

        result
    }

    /// Performs carry-less multiplication with reduction
    fn mul_internal(&self, other: &Self) -> Self {
        let mut a = self.value;
        let mut b = other.value;
        let mut result = 0;
        let mask = (1 << self.num_bits) - 1;

        // For 4-bit field, use x⁴ + x + 1 as primitive polynomial
        let primitive_poly = if self.num_bits == 4 {
            0b10011 // x⁴ + x + 1
        } else {
            (1 << self.num_bits) | 0b11 // Default to x^n + x + 1
        };

        // Keep inputs within field size
        a &= mask;
        b &= mask;

        while b != 0 {
            if b & 1 != 0 {
                result ^= a;
            }
            b >>= 1;
            a <<= 1;
            if a & (1 << self.num_bits) != 0 {
                a ^= primitive_poly;
            }
        }

        // Final reduction
        result &= mask;
        Self::new(result, Some(self.num_levels))
    }

    fn with_level(self, level: usize) -> Self {
        Self::new(self.value, Some(level))
    }
}

// Implement TowerField trait for TowerFieldElement
impl TowerField for TowerFieldElement {
    fn new(val: u128, num_levels: Option<usize>) -> Self {
        TowerFieldElement::new(val, num_levels)
    }

    fn get_val(&self) -> u128 {
        self.value()
    }

    fn num_levels(&self) -> usize {
        self.num_levels
    }

    fn num_bits(&self) -> usize {
        self.num_bits
    }

    fn bin(&self) -> String {
        self.to_binary_string()
    }

    fn split(&self) -> (Self, Self) {
        self.split()
    }

    fn join(&self, low: &Self) -> Self {
        self.join(low)
    }

    fn add(&self, other: &Self) -> Self {
        *self + *other
    }

    fn mul(&self, other: &Self) -> Self {
        let max_level = self.num_levels.max(other.num_levels);
        self.mul_internal(other).with_level(max_level)
    }
}

impl Add for TowerFieldElement {
    type Output = Self;

    #[inline]
    fn add(self, other: Self) -> Self {
        // Use the maximum level of the two operands
        let max_level = self.num_levels.max(other.num_levels);
        Self::new(self.value ^ other.value, Some(max_level))
    }
}

impl AddAssign for TowerFieldElement {
    #[inline]
    fn add_assign(&mut self, other: Self) {
        self.value ^= other.value;
        // Ensure the value stays within range
        let mask = if self.num_bits >= 128 {
            u128::MAX
        } else {
            (1 << self.num_bits) - 1
        };
        self.value &= mask;
    }
}

impl Sub for TowerFieldElement {
    type Output = Self;

    #[inline]
    fn sub(self, other: Self) -> Self {
        // In characteristic 2, subtraction equals addition
        self + other
    }
}

impl SubAssign for TowerFieldElement {
    #[inline]
    fn sub_assign(&mut self, other: Self) {
        // In characteristic 2, subtraction equals addition
        *self += other;
    }
}

impl Mul for TowerFieldElement {
    type Output = Self;

    #[inline]
    fn mul(self, other: Self) -> Self {
        let max_level = self.num_levels.max(other.num_levels);
        self.mul_internal(&other).with_level(max_level)
    }
}

impl MulAssign for TowerFieldElement {
    #[inline]
    fn mul_assign(&mut self, other: Self) {
        *self = self.mul_internal(&other);
    }
}

impl Neg for TowerFieldElement {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        // In characteristic 2, negation is identity
        self
    }
}

// Implement From traits for common integer types
impl From<u64> for TowerFieldElement {
    fn from(value: u64) -> Self {
        Self::new(value as u128, None)
    }
}

impl From<u32> for TowerFieldElement {
    fn from(value: u32) -> Self {
        Self::new(value as u128, None)
    }
}

impl From<u16> for TowerFieldElement {
    fn from(value: u16) -> Self {
        Self::new(value as u128, None)
    }
}

impl From<u8> for TowerFieldElement {
    fn from(value: u8) -> Self {
        Self::new(value as u128, None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== Binary Field Tests ====================

    /// Tests basic addition in the binary field GF(2³)
    #[test]
    fn test_basic_binary_field_addition() {
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        struct GF2_3;
        impl BinaryFieldConfig for GF2_3 {
            const DEGREE: u32 = 3;
            const PRIMITIVE_POLY: u128 = 0b1011; // x³ + x + 1
        }
        type Elem = BinaryFieldElement<GF2_3>;

        let a = Elem::new(0b011);
        let b = Elem::new(0b100);
        let sum = a + b;
        assert_eq!(sum.value(), 0b111); // Check if the sum is correct
    }

    /// Tests basic multiplication in the binary field GF(2³)
    #[test]
    fn test_basic_binary_field_multiplication() {
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        struct GF2_3;
        impl BinaryFieldConfig for GF2_3 {
            const DEGREE: u32 = 3;
            const PRIMITIVE_POLY: u128 = 0b1011; // x³ + x + 1
        }
        type Elem = BinaryFieldElement<GF2_3>;

        let a = Elem::new(0b011);
        let prod = a.clone() * a;
        assert_eq!(prod.value(), 0b101); // Check if the product is correct
    }

    /// Tests field properties for BinaryFieldElement
    #[test]
    fn test_binary_field_properties() {
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        struct GF2_3;
        impl BinaryFieldConfig for GF2_3 {
            const DEGREE: u32 = 3;
            const PRIMITIVE_POLY: u128 = 0b1011; // x³ + x + 1
        }
        type Elem = BinaryFieldElement<GF2_3>;

        // Test zero and one
        let zero = Elem::new(0);
        let one = Elem::new(1);
        assert!(zero.is_zero());
        assert!(!one.is_zero());
        assert!(!zero.is_one());
        assert!(one.is_one());

        // Test addition with zero
        let a = Elem::new(0b101);
        assert_eq!(a.clone() + zero, a);

        // Test multiplication with one
        assert_eq!(a.clone() * one, a);

        // Test negation (identity in characteristic 2)
        assert_eq!(-a.clone(), a);

        // Test subtraction equals addition
        let b = Elem::new(0b110);
        assert_eq!(a.clone() - b.clone(), a.clone() + b);
    }

    /// Tests boundary conditions for BinaryFieldElement
    #[test]
    fn test_binary_field_boundaries() {
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        struct GF2_3;
        impl BinaryFieldConfig for GF2_3 {
            const DEGREE: u32 = 3;
            const PRIMITIVE_POLY: u128 = 0b1011; // x³ + x + 1
        }
        type Elem = BinaryFieldElement<GF2_3>;

        // Test value masking on creation
        let a = Elem::new(0b1111); // This exceeds 3 bits
        assert_eq!(a.value(), 0b111); // Should be masked to 3 bits

        // Test large value
        let b = Elem::new(0xFFFFFFFF);
        assert_eq!(b.value(), 0b111); // Should be masked to 3 bits
    }

    /// Tests inverse computation for BinaryFieldElement
    #[test]
    fn test_binary_field_inverse() {
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        struct GF2_3;
        impl BinaryFieldConfig for GF2_3 {
            const DEGREE: u32 = 3;
            const PRIMITIVE_POLY: u128 = 0b1011; // x³ + x + 1
        }
        type Elem = BinaryFieldElement<GF2_3>;

        // Test inverse of zero
        let zero = Elem::new(0);
        assert!(zero.inverse().is_none());

        // Test inverse of one
        let one = Elem::new(1);
        assert_eq!(one.inverse().unwrap(), one);

        // Test inverse of other elements
        let a = Elem::new(0b010); // Element 2
        let a_inv = a.inverse().unwrap();
        assert_eq!((a * a_inv).value(), 1); // a * a^-1 = 1

        let b = Elem::new(0b011); // Element 3
        let b_inv = b.inverse().unwrap();
        assert_eq!((b * b_inv).value(), 1); // b * b^-1 = 1
    }

    /// Tests power operation for BinaryFieldElement
    #[test]
    fn test_binary_field_pow() {
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        struct GF2_3;
        impl BinaryFieldConfig for GF2_3 {
            const DEGREE: u32 = 3;
            const PRIMITIVE_POLY: u128 = 0b1011; // x³ + x + 1
        }
        type Elem = BinaryFieldElement<GF2_3>;

        // Test x^0 = 1
        let a = Elem::new(0b010); // Element 2
        assert_eq!(a.pow(0).value(), 1);

        // Test x^1 = x
        assert_eq!(a.pow(1), a);

        // Test x^2
        assert_eq!(a.pow(2), a.clone() * a.clone());

        // Test x^3
        assert_eq!(a.pow(3), a.clone() * a.clone() * a.clone());

        // Test x^7 = 1 in GF(2³) with primitive polynomial x³ + x + 1
        // This is because the multiplicative order of GF(2³)* is 2³-1 = 7
        assert_eq!(a.pow(7).value(), 1);
    }

    // ==================== Tower Field Tests ====================

    /// Tests creation of a tower field element
    #[test]
    fn test_tower_new_and_bin() {
        let elem = TowerFieldElement::new(5, Some(3)); // 3 levels => 8 bits
        assert_eq!(elem.num_bits(), 8); // Check the number of bits
        assert_eq!(elem.to_binary_string(), "00000101"); // Check binary representation
    }

    /// Tests addition in the tower field
    #[test]
    fn test_tower_addition() {
        let a = TowerFieldElement::new(0b011, Some(3)); // 8-bit representation: 00000011
        let b = TowerFieldElement::new(0b100, Some(3)); // 00000100
        let sum = a + b; // 00000011 XOR 00000100 = 00000111
        assert_eq!(sum.value(), 0b111);
        assert_eq!(sum.num_levels(), 3);
    }

    /// Tests multiplication in the tower field
    #[test]
    fn test_tower_multiplication() {
        let a = TowerFieldElement::new(0b011, Some(2)); // 4-bit representation: 0011
        let b = TowerFieldElement::new(0b010, Some(2)); // 0010
        let prod = a * b;
        assert_eq!(prod.value(), 0b110); // Check multiplication result
        assert_eq!(prod.num_levels(), 2);
    }

    /// Tests split and join operations
    #[test]
    fn test_tower_split_join() {
        let elem = TowerFieldElement::new(0b11001010, Some(3)); // 8 bits
        let (hi, lo) = elem.split();

        // Check that split produces correct high and low parts
        assert_eq!(hi.value(), 0b1100);
        assert_eq!(lo.value(), 0b1010);
        assert_eq!(hi.num_levels(), 2);
        assert_eq!(lo.num_levels(), 2);

        // Check that joining them produces the original element
        let joined = hi.join(&lo);
        assert_eq!(joined.value(), elem.value());
        assert_eq!(joined.num_levels(), elem.num_levels());
    }

    /// Tests tower field properties
    #[test]
    fn test_tower_field_properties() {
        let zero = TowerFieldElement::new(0, Some(3));
        let one = TowerFieldElement::new(1, Some(3));
        let a = TowerFieldElement::new(0b101, Some(3));
        let b = TowerFieldElement::new(0b011, Some(3));

        // Additive identity
        assert_eq!((a.clone() + zero.clone()).value(), a.value());

        // Multiplicative identity
        assert_eq!((a.clone() * one.clone()).value(), a.value());

        // Commutativity of addition
        assert_eq!(a.clone() + b.clone(), b.clone() + a.clone());

        // Commutativity of multiplication
        assert_eq!(a.clone() * b.clone(), b.clone() * a.clone());

        // Associativity of addition
        let c = TowerFieldElement::new(0b110, Some(3));
        assert_eq!(
            (a.clone() + b.clone()) + c.clone(),
            a.clone() + (b.clone() + c.clone())
        );

        // Distributivity
        assert_eq!(
            a.clone() * (b.clone() + c.clone()),
            (a.clone() * b.clone()) + (a.clone() * c.clone())
        );
    }

    /// Tests power operation in tower field
    #[test]
    fn test_tower_pow() {
        let a = TowerFieldElement::new(0b010, Some(2)); // 4-bit element

        // Test x^0 = 1
        assert_eq!(a.pow(0).value(), 1);

        // Test x^1 = x
        assert_eq!(a.pow(1), a);

        // Test x^2
        let squared = a.clone() * a.clone();
        assert_eq!(a.pow(2), squared);

        // Test x^3
        assert_eq!(a.pow(3), squared * a);
    }

    /// Tests operations between different tower levels
    #[test]
    fn test_tower_mixed_levels() {
        let a = TowerFieldElement::new(0b11, Some(2)); // 4-bit element
        let b = TowerFieldElement::new(0b1101, Some(3)); // 8-bit element

        // Addition should use the maximum level
        let sum = a.clone() + b.clone();
        assert_eq!(sum.num_levels(), 3);
        assert_eq!(sum.value(), 0b1101 ^ 0b11);

        // Multiplication should use the maximum level
        let prod = a * b;
        assert_eq!(prod.num_levels(), 3);
    }

    /// Tests boundary conditions and edge cases
    #[test]
    fn test_tower_boundaries() {
        // Test with maximum value for 3 levels (8 bits)
        let max_val = TowerFieldElement::new(0xFF, Some(3));
        assert_eq!(max_val.num_bits(), 8);
        assert_eq!(max_val.value(), 0xFF);

        // Test with value exceeding bit length
        let overflow = TowerFieldElement::new(0x1FF, Some(3));
        assert_eq!(overflow.value(), 0xFF); // Should be masked to 8 bits

        // Test with minimum level
        let min_level = TowerFieldElement::new(0b11, Some(1));
        assert_eq!(min_level.num_bits(), 2);
        assert_eq!(min_level.value(), 0b11);
    }

    /// Tests inverse computation
    #[test]
    fn test_tower_inverse() {
        let a = TowerFieldElement::new(0b011, Some(2)); // 4-bit element

        // Test inverse of zero
        let zero = TowerFieldElement::new(0, Some(2));
        assert!(zero.inverse().is_none());

        // Test inverse of one
        let one = TowerFieldElement::new(1, Some(2));
        assert_eq!(one.inverse().unwrap(), one);

        // Test inverse of non-zero element
        if let Some(a_inv) = a.inverse() {
            assert_eq!((a * a_inv).value(), 1);
        }
    }
}
