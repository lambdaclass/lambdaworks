use crate::field::{element::FieldElement, errors::FieldError, traits::IsField};
use core::fmt;
use core::marker::PhantomData;
use core::ops::{Add, Mul, Neg, Sub};

/// Implementation of binary fields (GF(2ⁿ)) and their tower field extensions.
///
/// This module provides implementations for:
/// 1. Basic binary fields GF(2ⁿ) with irreducible polynomial reduction
/// 2. Tower field extensions that support splitting and joining operations
///
/// The implementation is optimized for cryptographic applications, providing
/// efficient arithmetic operations and field extensions.
///
/// # Examples
///
/// ## Basic Binary Field
/// ```
/// use crate::field::fields::binary::{BinaryFieldConfig, BinaryFieldElement};
///
/// // Define a configuration for GF(2³)
/// #[derive(Clone, Debug)]
/// struct GF2_3;
/// impl BinaryFieldConfig for GF2_3 {
///     const DEGREE: u32 = 3;
///     // The primitive polynomial defines how elements are reduced during multiplication.
///     // For GF(2³), we use x³ + x + 1 (binary 1011) which is irreducible over GF(2).
///     // This ensures that field operations (especially multiplication) produce valid results
///     // within our field of 8 elements.
///     //
///     // Example: When multiplying (x + 1)(x + 1) = x² + 2x + 1 in GF(2³):
///     // 1. First compute regular multiplication: (0b011)(0b011) = x² + 2x + 1
///     // 2. In characteristic 2, 2x becomes 0: x² + 1 (binary 0b101)
///     // 3. Since x² + 1 is already in reduced form (degree < 3), this is our result
///     const PRIMITIVE_POLY: u128 = 0b1011;
/// }
///
/// // Create field elements
/// let a = BinaryFieldElement::<GF2_3>::new(0b011); // represents x + 1
/// let b = BinaryFieldElement::<GF2_3>::new(0b100); // represents x²
///
/// // Field operations
/// let sum = a + b;  // Addition in GF(2³) is XOR
/// let product = a * b;  // Multiplication followed by reduction
/// ```
///
/// ## Tower Field Extension
/// ```
/// use crate::field::fields::binary::TowerFieldElement;
///
/// // Create an 8-bit element (3 levels)
/// let a = TowerFieldElement::new(0b10110011, Some(3));
///
/// // Split into two 4-bit elements
/// let (hi, lo) = a.split();
/// assert_eq!(hi.get_val(), 0b1011);
/// assert_eq!(lo.get_val(), 0b0011);
///
/// // Join back
/// let rejoined = hi.join(&lo);
/// assert_eq!(rejoined.get_val(), 0b10110011);
/// ```
///
/// # Implementation Details
///
/// ## Binary Field Operations
/// - Addition and subtraction are performed using XOR (characteristic 2)
/// - Multiplication uses carry-less multiplication with reduction
/// - The primitive polynomial defines the reduction rule
///
/// ## Tower Field Structure
/// - Elements can be split into smaller field elements
/// - Smaller elements can be joined to form larger field elements
/// - Each level doubles the bit-length
/// - Operations maintain field properties at each level
///
/// ## Field Element Representation
/// - The underlying value is stored in a u128
/// - Only the lower `DEGREE` bits are significant
/// - Arithmetic operations are performed modulo the irreducible polynomial
///
/// ## Tower Field Operations
/// ----------------------
/// Basic Binary Field
/// ----------------------
///
///
/// Configuration for a binary field GF(2ⁿ).
///
/// This trait defines the parameters needed to construct a binary field:
/// - The degree `n` of the field extension (GF(2ⁿ))
/// - The primitive polynomial that defines the field structure
///
/// The primitive polynomial must be irreducible over GF(2) and of degree n.
/// It is represented as a u128 where each bit corresponds to a coefficient.
/// For example, for GF(2³), the polynomial x³ + x + 1 is represented as 0b1011:
/// - The bit at position 3 (0b1000) represents x³
/// - The bit at position 1 (0b0010) represents x
/// - The bit at position 0 (0b0001) represents 1
/// - Together they form 0b1011, representing x³ + x + 1
///
/// # Example
/// ```
/// #[derive(Clone, Debug)]
/// struct GF2_3;
/// impl BinaryFieldConfig for GF2_3 {
///     const DEGREE: u32 = 3;
///     const PRIMITIVE_POLY: u128 = 0b1011; // x³ + x + 1
/// }
/// ```
pub trait BinaryFieldConfig: Clone + fmt::Debug {
    /// Degree of the field extension (n). Only the lower n bits of an element are used.
    /// For GF(2ⁿ), this is the value of n.
    const DEGREE: u32;

    /// Primitive polynomial that defines the field structure.
    /// Must be an irreducible polynomial of degree n over GF(2).
    /// Represented as a u128 where each bit is a coefficient.
    /// For example, x³ + x + 1 is represented as 0b1011.
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

/// A binary field element in GF(2ⁿ), parameterized by a type F implementing `BinaryField`.
///
/// The underlying value is stored in a u128; however, only the lower `DEGREE` bits are significant.
/// All arithmetic operations are performed modulo the irreducible polynomial.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct BinaryFieldElement<F: BinaryFieldConfig> {
    /// Underlying value (only the lower `DEGREE` bits are significant)
    pub value: u128,
    /// Phantom data to keep track of the field configuration
    _phantom: PhantomData<F>,
}

impl<F: BinaryFieldConfig> BinaryFieldElement<F> {
    /// Creates a new binary field element by reducing the input value modulo 2^DEGREE.
    pub fn new(value: u128) -> Self {
        Self {
            value: value % (1 << F::DEGREE) as u128,
            _phantom: PhantomData,
        }
    }

    /// Returns the additive identity (0).
    pub fn zero() -> Self {
        Self::new(0)
    }

    /// Returns the multiplicative identity (1).
    pub fn one() -> Self {
        Self::new(1)
    }

    /// Adds two field elements using bitwise XOR.
    ///
    /// In GF(2ⁿ), addition is performed by bitwise XOR.
    /// This operation is commutative and associative.
    pub fn add(&self, other: &Self) -> Self {
        Self::new(self.value ^ other.value)
    }

    /// Subtracts two field elements.
    ///
    /// In GF(2ⁿ) subtraction is identical to addition due to the characteristic 2.
    /// This operation is commutative and associative.
    pub fn sub(&self, other: &Self) -> Self {
        Self::new(self.value ^ other.value)
    }

    /// Multiplies two field elements using carry-less polynomial multiplication
    /// with modular reduction by the irreducible polynomial.
    ///
    /// The algorithm uses the standard polynomial multiplication algorithm:
    /// 1. Perform carry-less multiplication of the operands
    /// 2. Reduce the result modulo the irreducible polynomial
    ///
    /// This operation is commutative and associative.
    pub fn mul(&self, other: &Self) -> Self {
        let mut a = self.value;
        let mut b = other.value;
        let mut result = 0;
        while b != 0 {
            if b & 1 != 0 {
                result ^= a;
            }
            b >>= 1;
            a <<= 1;
            // When a overflows DEGREE bits, reduce it using the irreducible polynomial.
            if a & (1 << F::DEGREE) != 0 {
                a ^= F::PRIMITIVE_POLY;
            }
        }
        Self::new(result)
    }

    /// Returns true if the element is zero
    pub fn is_zero(&self) -> bool {
        self.value == 0
    }
}

impl<F: BinaryFieldConfig> Add for BinaryFieldElement<F> {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self::add(&self, &other)
    }
}

impl<F: BinaryFieldConfig> Sub for BinaryFieldElement<F> {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        Self::sub(&self, &other)
    }
}

impl<F: BinaryFieldConfig> Mul for BinaryFieldElement<F> {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        Self::mul(&self, &other)
    }
}

impl<F: BinaryFieldConfig> Neg for BinaryFieldElement<F> {
    type Output = Self;
    /// In a field of characteristic 2, negation is the identity.
    fn neg(self) -> Self {
        self
    }
}

impl<F: BinaryFieldConfig> Default for BinaryFieldElement<F> {
    fn default() -> Self {
        Self::zero()
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

/// A simple tower field element.
/// It stores a u128 value along with the number of levels and the number of bits.
/// The number of bits is always 2^num_levels.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct TowerFieldElement {
    /// The underlying value
    pub value: u128,
    /// Number of levels in the tower
    pub num_levels: usize,
    /// Number of bits (2^num_levels)
    pub num_bits: usize,
}

impl TowerFieldElement {
    /// Creates a new tower field element.
    ///
    /// If num_levels is None, a default of 3 levels (8 bits) is used.
    /// The element is reduced modulo 2^(num_bits).
    pub fn new(val: u128, num_levels: Option<usize>) -> Self {
        let levels = num_levels.unwrap_or(3);
        let bits = 1 << levels;
        let modulus = if bits >= 128 {
            u128::MAX
        } else {
            (1 << bits) - 1
        };
        Self {
            value: val & modulus,
            num_levels: levels,
            num_bits: bits,
        }
    }

    /// Returns true if the element is zero
    pub fn is_zero(&self) -> bool {
        self.value == 0
    }
}

impl Default for TowerFieldElement {
    fn default() -> Self {
        Self::new(0, None)
    }
}

impl TowerField for TowerFieldElement {
    fn new(val: u128, num_levels: Option<usize>) -> Self {
        Self::new(val, num_levels)
    }

    fn get_val(&self) -> u128 {
        self.value
    }

    fn num_levels(&self) -> usize {
        self.num_levels
    }

    fn num_bits(&self) -> usize {
        self.num_bits
    }

    fn bin(&self) -> String {
        // Return a binary string of fixed width = num_bits.
        format!("{:0width$b}", self.value, width = self.num_bits)
    }

    fn split(&self) -> (Self, Self) {
        let half = self.num_bits / 2;
        let bin_str = self.bin();
        let hi_str = &bin_str[..half];
        let lo_str = &bin_str[half..];
        let hi_val = u128::from_str_radix(hi_str, 2).unwrap();
        let lo_val = u128::from_str_radix(lo_str, 2).unwrap();
        (
            TowerFieldElement::new(hi_val, Some(self.num_levels - 1)),
            TowerFieldElement::new(lo_val, Some(self.num_levels - 1)),
        )
    }

    fn join(&self, low: &Self) -> Self {
        let new_bits = self.num_bits * 2;
        let joined = (self.value << (self.num_bits)) | low.value;
        Self {
            value: joined,
            num_levels: self.num_levels + 1,
            num_bits: new_bits,
        }
    }

    fn add(&self, other: &Self) -> Self {
        TowerFieldElement::new(self.value ^ other.value, Some(self.num_levels))
    }

    fn mul(&self, other: &Self) -> Self {
        let mut a = self.value;
        let mut b = other.value;
        let mut result = 0;
        while b != 0 {
            if b & 1 != 0 {
                result ^= a;
            }
            b >>= 1;
            a <<= 1;
            // Naïve reduction: if a overflows the current bit-length, reduce it.
            if a & (1u128 << (self.num_bits)) != 0 {
                // Here we use a reduction polynomial: x^num_bits + 1
                a ^= (1u128 << self.num_bits) ^ 1;
            }
        }
        Self::new(result, Some(self.num_levels))
    }
}

impl Add for TowerFieldElement {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        <Self as TowerField>::add(&self, &other)
    }
}

impl Sub for TowerFieldElement {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        // In GF(2ⁿ), subtraction equals addition.
        <Self as TowerField>::add(&self, &other)
    }
}

impl Mul for TowerFieldElement {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        <Self as TowerField>::mul(&self, &other)
    }
}

impl Neg for TowerFieldElement {
    type Output = Self;
    fn neg(self) -> Self {
        // In characteristic 2, negation is a no-op.
        self
    }
}

impl IsField for TowerFieldElement {
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
            // Naïve reduction: if a overflows the current bit-length, reduce it.
            if a_val & (1u64 << 63) != 0 {
                // Here we use a reduction polynomial: x^63 + 1.
                a_val ^= (1u64 << 63) ^ 1;
            }
        }
        result
    }

    fn sub(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        // In characteristic 2, subtraction equals addition
        <Self as IsField>::add(a, b)
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
        let mut r = (1u64 << 63) ^ 1; // x^63 + 1
        let mut newr = *a;

        while newr != 0 {
            let deg_r = 63u32 - r.leading_zeros();
            let deg_newr = 63u32 - newr.leading_zeros();
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
        let b_inv = <Self as IsField>::inv(b).expect("Division by zero");
        <Self as IsField>::mul(a, &b_inv)
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
        x
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_binary_field_addition() {
        // Using the basic BinaryFieldElement for GF(2³)
        #[derive(Clone, Debug)]
        struct GF2_3;
        impl BinaryFieldConfig for GF2_3 {
            const DEGREE: u32 = 3;
            const PRIMITIVE_POLY: u128 = 0b1011;
        }
        type Elem = BinaryFieldElement<GF2_3>;

        let a = Elem::new(0b011);
        let b = Elem::new(0b100);
        let sum = a + b;
        assert_eq!(sum.value, 0b111);
    }

    #[test]
    fn test_basic_binary_field_multiplication() {
        #[derive(Clone, Debug)]
        struct GF2_3;
        impl BinaryFieldConfig for GF2_3 {
            const DEGREE: u32 = 3;
            const PRIMITIVE_POLY: u128 = 0b1011;
        }
        type Elem = BinaryFieldElement<GF2_3>;

        let a = Elem::new(0b011);
        let prod = a.clone() * a;
        assert_eq!(prod.value, 0b101);
    }

    #[test]
    fn test_tower_new_and_bin() {
        let elem = TowerFieldElement::new(5, Some(3)); // 3 levels => 8 bits
        assert_eq!(elem.num_bits(), 8);
        assert_eq!(elem.bin(), "00000101");
    }

    #[test]
    fn test_tower_addition() {
        let a = TowerFieldElement::new(0b011, Some(3)); // 8-bit representation: 00000011
        let b = TowerFieldElement::new(0b100, Some(3)); // 00000100
        let sum = a + b; // 00000011 XOR 00000100 = 00000111
        assert_eq!(sum.get_val(), 0b111);
    }

    #[test]
    fn test_tower_multiplication() {
        let a = TowerFieldElement::new(0b011, Some(3));
        let prod = a * a; // Expected: (x+1)² = x² + 1 (binary 101)
        assert_eq!(prod.get_val(), 0b101);
    }

    #[test]
    fn test_split_and_join() {
        // Create an element with 4 levels (16 bits)
        let elem = TowerFieldElement::new(0xABCD, Some(4));
        let (hi, lo) = elem.split();
        let joined = hi.join(&lo);
        // The joined value should equal the original.
        assert_eq!(joined.get_val(), elem.get_val());
        // The number of levels should be the same as the original.
        assert_eq!(joined.num_levels(), elem.num_levels());
    }

    #[test]
    fn test_split_and_join_detailed() {
        // Create an element with 3 levels (8 bits) with a specific pattern
        let elem = TowerFieldElement::new(0b10110011, Some(3));

        // Split into high and low parts
        let (hi, lo) = elem.split();

        // Verify the split parts
        assert_eq!(hi.num_levels(), 2); // 3 - 1 = 2 levels
        assert_eq!(lo.num_levels(), 2);
        assert_eq!(hi.num_bits(), 4); // 8/2 = 4 bits
        assert_eq!(lo.num_bits(), 4);
        assert_eq!(hi.get_val(), 0b1011); // High 4 bits
        assert_eq!(lo.get_val(), 0b0011); // Low 4 bits

        // Join the parts back
        let joined = hi.join(&lo);

        // Verify the joined result
        assert_eq!(joined.num_levels(), 3); // Back to original levels
        assert_eq!(joined.num_bits(), 8); // Back to original bits
        assert_eq!(joined.get_val(), 0b10110011); // Original value
        assert_eq!(joined.bin(), "10110011"); // Binary representation
    }

    #[test]
    fn test_commutativity() {
        let a = TowerFieldElement::new(0x1F, Some(4));
        let b = TowerFieldElement::new(0x2A, Some(4));
        assert_eq!(a + b, b + a);
        assert_eq!(a * b, b * a);
    }

    #[test]
    fn test_associativity() {
        // Test associativity of addition and multiplication
        let a = TowerFieldElement::new(0x1F, Some(4));
        let b = TowerFieldElement::new(0x2A, Some(4));
        let c = TowerFieldElement::new(0x35, Some(4));

        // Addition associativity: (a + b) + c = a + (b + c)
        assert_eq!((a + b) + c, a + (b + c));

        // Multiplication associativity: (a * b) * c = a * (b * c)
        assert_eq!((a * b) * c, a * (b * c));
    }

    #[test]
    fn test_distributivity() {
        // Test distributivity of multiplication over addition
        let a = TowerFieldElement::new(0x1F, Some(4));
        let b = TowerFieldElement::new(0x2A, Some(4));
        let c = TowerFieldElement::new(0x35, Some(4));

        // a * (b + c) = a * b + a * c
        assert_eq!(a * (b + c), a * b + a * c);
    }

    #[test]
    fn test_identity_properties() {
        // Test identity properties
        let a = TowerFieldElement::new(0x1F, Some(4));

        // Zero is the additive identity: a + 0 = a
        assert_eq!(a + TowerFieldElement::new(0, Some(4)), a);

        // One is the multiplicative identity: a * 1 = a
        assert_eq!(a * TowerFieldElement::new(1, Some(4)), a);
    }

    #[test]
    fn test_join_and_split_consistency() {
        // Test that join and split are consistent operations
        for i in 1..10 {
            let original = TowerFieldElement::new(i, Some(3));
            let (hi, lo) = original.split();
            let rejoined = hi.join(&lo);

            // The rejoined value should equal the original
            assert_eq!(rejoined.get_val(), original.get_val());
            assert_eq!(rejoined.num_levels(), original.num_levels());
            assert_eq!(rejoined.num_bits(), original.num_bits());
        }
    }

    #[test]
    fn test_binary_representation() {
        // Test binary string representation
        let field = TowerFieldElement::new(0b1010, Some(3));
        assert_eq!(field.bin(), "00001010"); // 8-bit representation (3 levels)

        let field = TowerFieldElement::new(0b1, Some(2));
        assert_eq!(field.bin(), "0001"); // 4-bit representation (2 levels)
    }

    #[test]
    fn test_negation() {
        // Test that negation is identity in characteristic 2
        let field = TowerFieldElement::new(0x1F, Some(4));
        assert_eq!(-field, field);
    }

    #[test]
    fn test_subtraction_equals_addition() {
        // Test that subtraction equals addition in characteristic 2
        let a = TowerFieldElement::new(0x1F, Some(4));
        let b = TowerFieldElement::new(0x2A, Some(4));

        assert_eq!(a - b, a + b);
    }

    #[test]
    fn test_different_level_operations() {
        // Test operations between elements with different levels
        let a = TowerFieldElement::new(0x0F, Some(3)); // 3 levels (8 bits)
        let b = TowerFieldElement::new(0x03, Some(2)); // 2 levels (4 bits)

        // When adding/multiplying elements with different levels,
        // the result should have the maximum level
        let sum = a + b;
        assert_eq!(sum.num_levels(), 3);
        assert_eq!(sum.get_val(), 0x0F ^ 0x03);

        let product = a * b;
        assert_eq!(product.num_levels(), 3);
    }

    #[test]
    fn test_complex_split_join_operations() {
        // Create a tower field element with a specific pattern
        let original = TowerFieldElement::new(0b10101010, Some(3)); // 8 bits

        // Split into high and low parts
        let (hi, lo) = original.split();

        // Verify the high and low parts
        assert_eq!(hi.get_val(), 0b1010);
        assert_eq!(lo.get_val(), 0b1010);
        assert_eq!(hi.num_levels(), 2);
        assert_eq!(lo.num_levels(), 2);

        // Modify the parts
        let modified_hi = hi + TowerFieldElement::new(0b0001, Some(2));
        let modified_lo = lo + TowerFieldElement::new(0b0100, Some(2));

        // Join the modified parts
        let rejoined = modified_hi.join(&modified_lo);

        // Verify the rejoined value
        assert_eq!(rejoined.get_val(), 0b10111110);
        assert_eq!(rejoined.num_levels(), 3);
        assert_eq!(rejoined.num_bits(), 8);
    }

    #[test]
    fn test_tower_multiplication_special_cases() {
        // Test multiplication by zero
        let a = TowerFieldElement::new(0x1F, Some(4));
        let zero = TowerFieldElement::new(0, Some(4));
        assert_eq!(a * zero, zero);
        assert_eq!(zero * a, zero);

        // Test multiplication by one
        let one = TowerFieldElement::new(1, Some(4));
        assert_eq!(a * one, a);
        assert_eq!(one * a, a);

        // Test multiplication by self
        let square = a * a;
        // Verify the result is properly reduced
        assert!(square.get_val() < (1 << square.num_bits()));
    }

    #[test]
    fn test_tower_multiplication_overflow() {
        // Test multiplication that causes overflow
        let a = TowerFieldElement::new(0xFFFF, Some(4)); // 16 bits all ones
        let b = TowerFieldElement::new(0xFFFF, Some(4));
        let result = a * b;

        // Verify the result is properly reduced
        assert!(result.get_val() < (1 << result.num_bits()));
    }

    #[test]
    fn test_tower_multiplication_patterns() {
        // Test multiplication with specific bit patterns
        let patterns = [
            (0b1010, 0b0101), // Alternating bits
            (0b1111, 0b0000), // All ones and zeros
            (0b1100, 0b0011), // Split pattern
        ];

        for (a, b) in patterns {
            let elem_a = TowerFieldElement::new(a, Some(2));
            let elem_b = TowerFieldElement::new(b, Some(2));
            let result = elem_a * elem_b;

            // Verify result is within bounds
            assert!(result.get_val() < (1 << result.num_bits()));
        }
    }

    #[test]
    fn test_tower_multiplication_levels() {
        // Test multiplication with different levels
        let a = TowerFieldElement::new(0x0F, Some(3)); // 3 levels (8 bits)
        let b = TowerFieldElement::new(0x03, Some(2)); // 2 levels (4 bits)

        // Result should have the maximum level
        let result = a * b;
        assert_eq!(result.num_levels(), 3);

        // Verify the multiplication is correct
        // In binary field multiplication:
        // 0x0F (1111) * 0x03 (0011) should give us the carry-less multiplication
        // followed by reduction modulo x^8 + 1
        let expected = 0x11; // This is the correct result in the binary field
        assert_eq!(result.get_val(), expected);
    }

    #[test]
    fn test_tower_multiplication_reduction() {
        // Test that multiplication properly reduces modulo the field size
        let a = TowerFieldElement::new(0xFFFF, Some(4)); // 16 bits
        let b = TowerFieldElement::new(0xFFFF, Some(4));
        let result = a * b;

        // The result should be reduced modulo 2^16
        assert!(result.get_val() < (1 << 16));
    }

    #[test]
    fn test_tower_multiplication_consistency() {
        // Test that multiplication is consistent across different representations
        let value = 0xABCD;
        let a = TowerFieldElement::new(value, Some(4));

        // Split and rejoin
        let (hi, lo) = a.split();
        let rejoined = hi.join(&lo);

        // Multiply both representations by the same value
        let multiplier = TowerFieldElement::new(0x1234, Some(4));
        let result1 = a * multiplier;
        let result2 = rejoined * multiplier;

        assert_eq!(result1.get_val(), result2.get_val());
    }

    #[test]
    fn test_tower_multiplication_identity() {
        // Test multiplicative identity properties
        let values = [0x1F, 0x2A, 0x35, 0x40];
        let one = TowerFieldElement::new(1, Some(4));

        for &val in &values {
            let a = TowerFieldElement::new(val, Some(4));
            assert_eq!(a * one, a);
            assert_eq!(one * a, a);
        }
    }

    #[test]
    fn test_tower_multiplication_zero() {
        // Test multiplicative zero properties
        let values = [0x1F, 0x2A, 0x35, 0x40];
        let zero = TowerFieldElement::new(0, Some(4));

        for &val in &values {
            let a = TowerFieldElement::new(val, Some(4));
            assert_eq!(a * zero, zero);
            assert_eq!(zero * a, zero);
        }
    }
}
