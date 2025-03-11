use crate::field::{element::FieldElement, errors::FieldError, traits::IsField};
use core::fmt;
use core::marker::PhantomData;
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// Specific errors for binary field operations
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BinaryFieldError {
    /// Attempt to invert the zero element
    InverseOfZero,
    /// Attempt to divide by zero
    DivisionByZero,
    /// Elements from different field configurations
    IncompatibleFields,
    /// Invalid primitive polynomial (not irreducible or wrong degree)
    InvalidPrimitivePoly,
}

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

    /// Splits the element into high and low parts
    /// Returns (high_part, low_part) where each part has half the bits
    pub fn split(&self) -> (Self, Self) {
        let half_degree = F::DEGREE / 2;
        let mask = (1 << half_degree) - 1;
        let lo = self.value & mask;
        let hi = (self.value >> half_degree) & mask;
        (Self::new(hi), Self::new(lo))
    }

    /// Joins high and low parts into a single element
    /// The high part becomes the most significant bits
    pub fn join(&self, low: &Self) -> Self {
        let half_degree = F::DEGREE / 2;
        Self::new((self.value << half_degree) | low.value)
    }

    /// Computes the multiplicative inverse of this element using a recursive algorithm
    /// Returns an error if the element is zero
    #[inline]
    pub fn inv(&self) -> Result<Self, BinaryFieldError> {
        if self.is_zero() {
            return Err(BinaryFieldError::InverseOfZero);
        }

        // For very small fields or odd degree fields, use Fermat's little theorem
        if F::DEGREE <= 2 || F::DEGREE % 2 != 0 {
            // Use Fermat's little theorem:
            // In GF(2^n), x^(2^n-1) = 1 for any non-zero x
            // Therefore x^(2^n-2) is the multiplicative inverse
            return Ok(self.pow((1 << F::DEGREE) - 2));
        }

        // For larger even degree fields, use recursive algorithm
        let (a_hi, a_lo) = self.split();

        // Compute k = n/2 where n is the field degree
        let k = F::DEGREE / 2;

        // Compute 2^(k-1) as a field element
        let two_pow_k_minus_one = Self::new(1 << (k - 1));

        // a = a_hi * x^k + a_lo
        // a_lo_next = a_hi * x^(k-1) + a_lo
        let a_lo_next = a_lo.clone() + a_hi.clone() * two_pow_k_minus_one;

        // Δ = a_lo * a_lo_next + a_hi^2
        let delta = a_lo.clone() * a_lo_next.clone() + a_hi.clone() * a_hi.clone();

        // Compute inverse of delta recursively
        let delta_inverse = delta.inv()?;

        // Compute parts of the inverse
        let out_hi = delta_inverse.clone() * a_hi;
        let out_lo = delta_inverse * a_lo_next;

        // Join the parts to get the final inverse
        Ok(out_hi.join(&out_lo))
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
        }

        // Final reduction to ensure result is in field
        result & mask
    }

    /// Divides this element by another
    /// Returns an error if the divisor is zero
    pub fn div(&self, other: &Self) -> Result<Self, BinaryFieldError> {
        if other.is_zero() {
            return Err(BinaryFieldError::DivisionByZero);
        }

        // Division is multiplication by the inverse
        let other_inv = other.inv()?;
        Ok(self.clone() * other_inv)
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

    /// Computes the multiplicative inverse of this element using a recursive algorithm
    /// Returns an error if the element is zero
    pub fn inv(&self) -> Result<Self, BinaryFieldError> {
        if self.is_zero() {
            return Err(BinaryFieldError::InverseOfZero);
        }

        // For small fields or odd-level fields, use Fermat's little theorem
        if self.num_levels <= 1 || self.num_bits <= 4 {
            // Use Fermat's little theorem:
            // In GF(2^n), x^(2^n-1) = 1 for any non-zero x
            // Therefore x^(2^n-2) is the multiplicative inverse
            return Ok(self.pow((1 << self.num_bits) - 2));
        }

        // For larger fields, use recursive algorithm
        let (a_hi, a_lo) = self.split();

        // Compute 2^(k-1) where k = num_bits/2
        let two_pow_k_minus_one =
            Self::new(1 << ((self.num_bits / 2) - 1), Some(self.num_levels - 1));

        // a = a_hi * x^k + a_lo
        // a_lo_next = a_hi * x^(k-1) + a_lo
        let a_lo_next = a_lo.clone() + a_hi.clone() * two_pow_k_minus_one;

        // Δ = a_lo * a_lo_next + a_hi^2
        let delta = a_lo.clone() * a_lo_next.clone() + a_hi.clone() * a_hi.clone();

        // Compute inverse of delta recursively
        let delta_inverse = delta.inv()?;

        // Compute parts of the inverse
        let out_hi = delta_inverse.clone() * a_hi;
        let out_lo = delta_inverse * a_lo_next;

        // Join the parts to get the final inverse
        Ok(out_hi.join(&out_lo))
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

    /// Divides this element by another
    /// Returns an error if the divisor is zero
    pub fn div(&self, other: &Self) -> Result<Self, BinaryFieldError> {
        if other.is_zero() {
            return Err(BinaryFieldError::DivisionByZero);
        }

        // Check if elements have compatible levels
        if self.num_levels != other.num_levels {
            return Err(BinaryFieldError::IncompatibleFields);
        }

        // To get the correct inverse, we calculate it using Fermat's Little Theorem
        // In GF(2^n), x^(2^n-1) = 1 for any non-zero x
        // Therefore, x^(2^n-2) is the multiplicative inverse of x
        let exp = (1u32 << self.num_bits) - 2;
        let other_inv = if other.value == 1 {
            other.clone() // The inverse of 1 is 1
        } else {
            other.pow(exp)
        };

        // Division is multiplication by the inverse
        Ok(self.clone() * other_inv)
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
        assert_eq!(sum.value(), 0b111);
    }

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
        assert_eq!(prod.value(), 0b101);
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
        assert!(zero.inv().is_err());
        assert_eq!(zero.inv().unwrap_err(), BinaryFieldError::InverseOfZero);

        // Test inverse of one
        let one = Elem::new(1);
        assert_eq!(one.inv().unwrap(), one);

        // Test inverse of other elements
        let a = Elem::new(0b010);
        let a_inv = a.inv().unwrap();
        assert_eq!((a * a_inv).value(), 1); // a * a^-1 = 1

        let b = Elem::new(0b011);
        let b_inv = b.inv().unwrap();
        assert_eq!((b * b_inv).value(), 1); // b * b^-1 = 1
    }

    #[test]
    fn test_binary_field_pow() {
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        struct GF2_3;
        impl BinaryFieldConfig for GF2_3 {
            const DEGREE: u32 = 3;
            const PRIMITIVE_POLY: u128 = 0b1011; // x³ + x + 1
        }
        type Elem = BinaryFieldElement<GF2_3>;

        //  x^0 = 1
        let a = Elem::new(0b010);
        assert_eq!(a.pow(0).value(), 1);

        // x^1 = x
        assert_eq!(a.pow(1), a);

        //  x^2
        assert_eq!(a.pow(2), a.clone() * a.clone());

        //  x^3
        assert_eq!(a.pow(3), a.clone() * a.clone() * a.clone());

        //  x^7 = 1 in GF(2³) with primitive polynomial x³ + x + 1
        // This is because the multiplicative order of GF(2³)* is 2³-1 = 7
        assert_eq!(a.pow(7).value(), 1);
    }

    #[test]
    fn test_tower_new_and_bin() {
        let elem = TowerFieldElement::new(5, Some(3)); // 3 levels => 8 bits
        assert_eq!(elem.num_bits(), 8); // Check the number of bits
        assert_eq!(elem.to_binary_string(), "00000101"); // Check binary representation
    }

    #[test]
    fn test_tower_addition() {
        let a = TowerFieldElement::new(0b011, Some(3)); // 8-bit representation: 00000011
        let b = TowerFieldElement::new(0b100, Some(3)); // 00000100
        let sum = a + b; // 00000011 XOR 00000100 = 00000111
        assert_eq!(sum.value(), 0b111);
        assert_eq!(sum.num_levels(), 3);
    }

    #[test]
    fn test_tower_multiplication() {
        let a = TowerFieldElement::new(0b011, Some(2)); // 4-bit representation: 0011
        let b = TowerFieldElement::new(0b010, Some(2)); // 0010
        let prod = a * b;
        assert_eq!(prod.value(), 0b110); // Check multiplication result
        assert_eq!(prod.num_levels(), 2);
    }

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

    #[test]
    fn test_tower_inverse() {
        let a = TowerFieldElement::new(0b011, Some(2)); // 4-bit element

        // Test inverse of zero
        let zero = TowerFieldElement::new(0, Some(2));
        assert!(zero.inv().is_err());
        assert_eq!(zero.inv().unwrap_err(), BinaryFieldError::InverseOfZero);

        // Test inverse of one
        let one = TowerFieldElement::new(1, Some(2));
        assert_eq!(one.inv().unwrap(), one);

        // Test inverse of non-zero element
        let a_inv = a.inv().unwrap();
        assert_eq!((a * a_inv).value(), 1);
    }

    #[test]
    fn test_tower_division() {
        // Crear elementos en GF(2^4)
        let one = TowerFieldElement::new(1, Some(2));
        let two = TowerFieldElement::new(2, Some(2));
        let three = TowerFieldElement::new(3, Some(2));
        let four = TowerFieldElement::new(4, Some(2));
        let zero = TowerFieldElement::new(0, Some(2));

        // Test básico: división por uno
        assert_eq!(three.div(&one).unwrap(), three);

        // Test división de elemento por sí mismo
        assert_eq!(three.div(&three).unwrap(), one);

        // Test división por cero
        assert!(three.div(&zero).is_err());
        assert_eq!(
            three.div(&zero).unwrap_err(),
            BinaryFieldError::DivisionByZero
        );

        // Test división con niveles incompatibles
        let big_elem = TowerFieldElement::new(2, Some(3));
        assert!(three.div(&big_elem).is_err());
        assert_eq!(
            three.div(&big_elem).unwrap_err(),
            BinaryFieldError::IncompatibleFields
        );

        // Test operaciones básicas de división
        // Primero: 4 = 2 * 2
        assert_eq!(two.clone() * two.clone(), four);
        // Ahora: 4 / 2 = 2
        assert_eq!(four.div(&two).unwrap(), two);
    }

    #[test]
    fn test_binary_field_division() {
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        struct GF2_3;
        impl BinaryFieldConfig for GF2_3 {
            const DEGREE: u32 = 3;
            const PRIMITIVE_POLY: u128 = 0b1011; // x³ + x + 1
        }
        type Elem = BinaryFieldElement<GF2_3>;

        // Test division by zero
        let a = Elem::new(0b010);
        let zero = Elem::new(0);
        assert!(a.div(&zero).is_err());
        assert_eq!(a.div(&zero).unwrap_err(), BinaryFieldError::DivisionByZero);

        // Test division by one
        let one = Elem::new(1);
        assert_eq!(a.div(&one).unwrap(), a);

        // Test division of a by itself (should be 1)
        assert_eq!(a.div(&a).unwrap(), one);

        // Test general division
        let b = Elem::new(0b011);
        let c = a.clone() * b.clone();
        assert_eq!(c.div(&b).unwrap(), a);
    }
}
