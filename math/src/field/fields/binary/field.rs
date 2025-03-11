/// Tower Field Extension
use crate::field::{element::FieldElement, errors::FieldError, traits::IsField};
use core::fmt;
use core::marker::PhantomData;
use core::ops::{Add, Mul, Neg, Sub};

/// Implementation of binary fields (GF(2ⁿ)) and their tower field extensions.
///
/// This module provides implementations for:
/// 1. Basic binary fields GF(2ⁿ) with irreducible polynomial reduction
/// 2. Tower field extensions that support splitting and joining operations

/// Represents a binary field configuration for GF(2ⁿ).
///
/// This trait defines the parameters needed for a binary field:
/// - The degree `n` of the field extension (GF(2ⁿ))
/// - The primitive polynomial that defines the field structure
pub trait BinaryFieldConfig: Clone + fmt::Debug {
    /// Degree of the field extension (n in GF(2ⁿ))
    const DEGREE: u32;

    /// Primitive polynomial represented as bits
    /// Example: x³ + x + 1 is represented as 0b1011
    const PRIMITIVE_POLY: u128;
}

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
        // Extended Euclidean algorithm in GF(2^n)
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
            }
            let shift = deg_r.saturating_sub(deg_newr);
            r ^= newr << shift;
            t ^= newt << shift;
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

// Example configuration for GF(2³)
#[derive(Clone, Debug)]
pub struct GF2_3;
impl BinaryFieldConfig for GF2_3 {
    const DEGREE: u32 = 3;
    const PRIMITIVE_POLY: u128 = 0b1011; // x³ + x + 1
}

pub type GF2_3Field = BinaryField<GF2_3>;
pub type GF2_3Element = FieldElement<GF2_3Field>;

/// A binary field element in GF(2ⁿ).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
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

    /// Returns the underlying value
    #[inline]
    pub fn value(&self) -> u128 {
        self.value
    }

    /// Performs carry-less multiplication with reduction
    #[inline]
    fn mul_internal(mut a: u128, mut b: u128) -> u128 {
        let mut result = 0;
        while b != 0 {
            if b & 1 != 0 {
                result ^= a;
            }
            b >>= 1;
            a <<= 1;
            if a & (1 << F::DEGREE) != 0 {
                a ^= F::PRIMITIVE_POLY;
            }
        }
        result
    }
}

impl<F: BinaryFieldConfig> Add for BinaryFieldElement<F> {
    type Output = Self;

    #[inline]
    fn add(self, other: Self) -> Self {
        Self::new(self.value ^ other.value)
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

impl<F: BinaryFieldConfig> Mul for BinaryFieldElement<F> {
    type Output = Self;

    #[inline]
    fn mul(self, other: Self) -> Self {
        Self::new(Self::mul_internal(self.value, other.value))
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

/// Tower Field Extension
///
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
    value: u128,
    num_levels: usize,
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

    /// Performs carry-less multiplication with reduction
    fn mul_internal(&self, other: &Self) -> Self {
        let mut a = self.value;
        let mut b = other.value;
        let mut result = 0;

        while b != 0 {
            if b & 1 != 0 {
                result ^= a;
            }
            b >>= 1;
            a <<= 1;
            if a & (1 << self.num_bits) != 0 {
                a ^= (1 << self.num_bits) ^ 1;
            }
        }

        Self::new(result, Some(self.num_levels))
    }
}

impl Add for TowerFieldElement {
    type Output = Self;

    #[inline]
    fn add(self, other: Self) -> Self {
        Self::new(self.value ^ other.value, Some(self.num_levels))
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

impl Mul for TowerFieldElement {
    type Output = Self;

    #[inline]
    fn mul(self, other: Self) -> Self {
        self.mul_internal(&other)
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
#[cfg(test)]
mod tests {
    use super::*;

    // Test basic addition in the binary field GF(2³)
    #[test]
    fn test_basic_binary_field_addition() {
        #[derive(Clone, Debug)]
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

    // Test basic multiplication in the binary field GF(2³)
    #[test]
    fn test_basic_binary_field_multiplication() {
        #[derive(Clone, Debug)]
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

    // Test creation of a tower field element
    #[test]
    fn test_tower_new_and_bin() {
        let elem = TowerFieldElement::new(5, Some(3)); // 3 levels => 8 bits
        assert_eq!(elem.num_bits(), 8); // Check the number of bits
        assert_eq!(elem.to_binary_string(), "00000101"); // Check binary representation
    }

    // Test addition in the tower field
    #[test]
    fn test_tower_addition() {
        let a = TowerFieldElement::new(0b011, Some(3)); // 8-bit representation: 00000011
        let b = TowerFieldElement::new(0b100, Some(3)); // 00000100
        let sum = a + b; // 00000011 XOR 00000100 = 00000111
        assert_eq!(sum.value(), 0b111); // Check if the sum is correct
    }

    // Test multiplication in the tower field
    #[test]
    fn test_tower_multiplication() {
        let a = TowerFieldElement::new(0b011, Some(3));
        let prod = a * a; // Expected: (x+1)² = x² + 1 (binary 101)
        assert_eq!(prod.value(), 0b101); // Check if the product is correct
    }

    // Test split and join operations in the tower field
    #[test]
    fn test_split_and_join() {
        let elem = TowerFieldElement::new(0xABCD, Some(4)); // 16 bits
        let (hi, lo) = elem.split();
        let joined = hi.join(&lo);
        assert_eq!(joined.value(), elem.value()); // Check if the joined value equals the original
        assert_eq!(joined.num_levels(), elem.num_levels()); // Check if the number of levels is preserved
    }

    // Test detailed split and join operations
    #[test]
    fn test_split_and_join_detailed() {
        let elem = TowerFieldElement::new(0b10110011, Some(3)); // 8 bits
        let (hi, lo) = elem.split();

        assert_eq!(hi.num_levels(), 2); // Check levels after split
        assert_eq!(lo.num_levels(), 2);
        assert_eq!(hi.num_bits(), 4); // Check bits after split
        assert_eq!(lo.num_bits(), 4);
        assert_eq!(hi.value(), 0b1011); // Check high part value
        assert_eq!(lo.value(), 0b0011); // Check low part value

        let joined = hi.join(&lo);
        assert_eq!(joined.num_levels(), 3); // Check levels after join
        assert_eq!(joined.num_bits(), 8); // Check bits after join
        assert_eq!(joined.value(), 0b10110011); // Check original value after join
        assert_eq!(joined.to_binary_string(), "10110011"); // Check binary representation
    }

    // Test commutativity of addition and multiplication
    #[test]
    fn test_commutativity() {
        let a = TowerFieldElement::new(0x1F, Some(4));
        let b = TowerFieldElement::new(0x2A, Some(4));
        assert_eq!(a + b, b + a); // Check addition commutativity
        assert_eq!(a * b, b * a); // Check multiplication commutativity
    }

    // Test associativity of addition and multiplication
    #[test]
    fn test_associativity() {
        let a = TowerFieldElement::new(0x1F, Some(4));
        let b = TowerFieldElement::new(0x2A, Some(4));
        let c = TowerFieldElement::new(0x35, Some(4));

        assert_eq!((a + b) + c, a + (b + c)); // Check addition associativity
        assert_eq!((a * b) * c, a * (b * c)); // Check multiplication associativity
    }

    // Test distributivity of multiplication over addition
    #[test]
    fn test_distributivity() {
        let a = TowerFieldElement::new(0x1F, Some(4));
        let b = TowerFieldElement::new(0x2A, Some(4));
        let c = TowerFieldElement::new(0x35, Some(4));

        assert_eq!(a * (b + c), a * b + a * c); // Check distributivity
    }

    // Test identity properties
    #[test]
    fn test_identity_properties() {
        let a = TowerFieldElement::new(0x1F, Some(4));
        assert_eq!(a + TowerFieldElement::new(0, Some(4)), a); // Check additive identity
        assert_eq!(a * TowerFieldElement::new(1, Some(4)), a); // Check multiplicative identity
    }

    // Test consistency of join and split operations
    #[test]
    fn test_join_and_split_consistency() {
        for i in 1..10 {
            let original = TowerFieldElement::new(i, Some(3));
            let (hi, lo) = original.split();
            let rejoined = hi.join(&lo);
            assert_eq!(rejoined.value(), original.value()); // Check if rejoined equals original
            assert_eq!(rejoined.num_levels(), original.num_levels()); // Check levels
            assert_eq!(rejoined.num_bits(), original.num_bits()); // Check bits
        }
    }

    // Test binary representation of tower field elements
    #[test]
    fn test_binary_representation() {
        let field = TowerFieldElement::new(0b1010, Some(3));
        assert_eq!(field.to_binary_string(), "00001010"); // Check binary representation for 8 bits

        let field = TowerFieldElement::new(0b1, Some(2));
        assert_eq!(field.to_binary_string(), "0001"); // Check binary representation for 4 bits
    }

    // Test negation in characteristic 2
    #[test]
    fn test_negation() {
        let field = TowerFieldElement::new(0x1F, Some(4));
        assert_eq!(-field, field); // Negation should be identity
    }

    // Test that subtraction equals addition in characteristic 2
    #[test]
    fn test_subtraction_equals_addition() {
        let a = TowerFieldElement::new(0x1F, Some(4));
        let b = TowerFieldElement::new(0x2A, Some(4));
        assert_eq!(a - b, a + b); // Check that subtraction equals addition
    }

    // Test operations between elements with different levels
    #[test]
    fn test_different_level_operations() {
        let a = TowerFieldElement::new(0x0F, Some(3)); // 3 levels (8 bits)
        let b = TowerFieldElement::new(0x03, Some(2)); // 2 levels (4 bits)

        let sum = a + b;
        assert_eq!(sum.num_levels(), 3); // Result should have the maximum level
        assert_eq!(sum.value(), 0x0F ^ 0x03); // Check value

        let product = a * b;
        assert_eq!(product.num_levels(), 3); // Result should have the maximum level
    }

    // Test complex split and join operations
    #[test]
    fn test_complex_split_join_operations() {
        let original = TowerFieldElement::new(0b10101010, Some(3)); // 8 bits
        let (hi, lo) = original.split();

        assert_eq!(hi.value(), 0b1010); // Check high part value
        assert_eq!(lo.value(), 0b1010); // Check low part value
        assert_eq!(hi.num_levels(), 2); // Check levels after split
        assert_eq!(lo.num_levels(), 2);

        let modified_hi = hi + TowerFieldElement::new(0b0001, Some(2));
        let modified_lo = lo + TowerFieldElement::new(0b0100, Some(2));

        let rejoined = modified_hi.join(&modified_lo);
        assert_eq!(rejoined.value(), 0b10111110); // Check rejoined value
        assert_eq!(rejoined.num_levels(), 3); // Check levels after join
        assert_eq!(rejoined.num_bits(), 8); // Check bits after join
    }

    // Test multiplication by zero
    #[test]
    fn test_tower_multiplication_special_cases() {
        let a = TowerFieldElement::new(0x1F, Some(4));
        let zero = TowerFieldElement::new(0, Some(4));
        assert_eq!(a * zero, zero); // Check multiplication by zero
        assert_eq!(zero * a, zero); // Check multiplication by zero
    }

    // Test multiplication by one
    #[test]
    fn test_tower_multiplication_identity() {
        let a = TowerFieldElement::new(0x1F, Some(4));
        let one = TowerFieldElement::new(1, Some(4));
        assert_eq!(a * one, a); // Check multiplication by one
        assert_eq!(one * a, a); // Check multiplication by one
    }

    // Test multiplication that causes overflow
    #[test]
    fn test_tower_multiplication_overflow() {
        let a = TowerFieldElement::new(0xFFFF, Some(4)); // 16 bits all ones
        let b = TowerFieldElement::new(0xFFFF, Some(4));
        let result = a * b;
        assert!(result.value() < (1 << 16)); // Verify the result is properly reduced
    }

    // Test multiplication with specific bit patterns
    #[test]
    fn test_tower_multiplication_patterns() {
        let patterns = [
            (0b1010, 0b0101), // Alternating bits
            (0b1111, 0b0000), // All ones and zeros
            (0b1100, 0b0011), // Split pattern
        ];

        for (a, b) in patterns {
            let elem_a = TowerFieldElement::new(a, Some(2));
            let elem_b = TowerFieldElement::new(b, Some(2));
            let result = elem_a * elem_b;
            assert!(result.value() < (1 << result.num_bits())); // Verify result is within bounds
        }
    }

    // Test that multiplication is consistent across different representations
    #[test]
    fn test_tower_multiplication_consistency() {
        let value = 0xABCD;
        let a = TowerFieldElement::new(value, Some(4));

        let (hi, lo) = a.split();
        let rejoined = hi.join(&lo);

        let multiplier = TowerFieldElement::new(0x1234, Some(4));
        let result1 = a * multiplier;
        let result2 = rejoined * multiplier;

        assert_eq!(result1.value(), result2.value()); // Check consistency
    }
}
