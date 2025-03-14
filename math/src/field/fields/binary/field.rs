use crate::field::{element::FieldElement, errors::FieldError, traits::IsField};
use crate::unsigned_integer::traits::IsUnsignedInteger;
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
/// The degree of the extension is equal to the degree of the primitive polynomial.
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

    // Shift and add algorithm.
    fn mul(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        let mut result = 0;
        let mut a_val = *a;
        let mut b_val = *b;
        while b_val != 0 {
            // If the LSB isn't 0:
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

    // Using Fermat's little therome we know that a^{-1} = a^{q-2} with q the order of the field.
    // https://planetmath.org/fermatslittletheorem
    // If the extension is of degree n, then the field has order 2^n.
    fn inv(a: &Self::BaseType) -> Result<Self::BaseType, FieldError> {
        // TODO: what if a == mod.
        if *a == 0 {
            return Err(FieldError::InvZeroError);
        }
        let exponent = (1 << C::DEGREE) - 2;
        Ok(Self::pow(a, exponent as u64))
    }

    fn div(a: &Self::BaseType, b: &Self::BaseType) -> Result<Self::BaseType, FieldError> {
        let b_inv = Self::inv(b).map_err(|_| FieldError::DivisionByZero)?;
        Ok(Self::mul(a, &b_inv))
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

    // The order of the field is 2^n with n the degree of the extension
    // (i.e the degree of the irreducible polynomial).
    fn from_base_type(x: Self::BaseType) -> Self::BaseType {
        x % (1 << C::DEGREE) as u64
    }
}

impl<C: BinaryFieldConfig> FieldElement<BinaryField<C>> {
    /// Splits the element into high and low parts
    /// Returns (high_part, low_part) where each part has half the bits
    pub fn split(&self) -> (Self, Self) {
        let half_degree = C::DEGREE / 2;
        let mask = (1 << half_degree) - 1;
        let lo = self.value() & mask;
        let hi = (self.value() >> half_degree) & mask;
        (Self::new(hi), Self::new(lo))
    }

    /// Joins high and low parts into a single element
    /// The high part becomes the most significant bits
    pub fn join(&self, low: &Self) -> Self {
        let half_degree = C::DEGREE / 2;
        Self::new((self.value() << half_degree) | low.value())
    }

    /// TODO: Is it usefull to have this function like this?
    /// Returns the total number of bits.
    pub fn num_bits(&self) -> usize {
        1 << C::DEGREE
    }

    /// Returns binary string representation.
    /// The string length is equal to num_bits. why???
    pub fn to_binary_string(&self) -> String {
        format!("{:0width$b}", self.value(), width = self.num_bits())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Example configuration for GF(2^2)
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct GF2_2;
    impl BinaryFieldConfig for GF2_2 {
        const DEGREE: u32 = 2;
        const PRIMITIVE_POLY: u128 = 0b111; // x^2 + x + 1
    }

    pub type F2 = BinaryField<GF2_2>;
    pub type F2E = FieldElement<F2>;

    // Example configuration for GF(2³)
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct GF2_3;
    impl BinaryFieldConfig for GF2_3 {
        const DEGREE: u32 = 3;
        const PRIMITIVE_POLY: u128 = 0b1011; // x³ + x + 1
    }

    pub type F3 = BinaryField<GF2_3>;
    pub type F3E = FieldElement<F3>;

    #[test]
    fn test_binary_field_properties() {
        // Test addition with zero
        let a = F3E::new(0b101);
        assert_eq!(a.clone() + F3E::zero(), a);

        // Test multiplication with one
        assert_eq!(a.clone() * F3E::one(), a);

        // Test negation (identity in characteristic 2)
        assert_eq!(-a.clone(), a);

        // Test subtraction equals addition
        let b = F3E::new(0b110);
        assert_eq!(a.clone() - b.clone(), a.clone() + b);
    }

    /// Tests boundary conditions for BinaryFieldElement
    #[test]
    fn test_binary_field_boundaries() {
        // Test value masking on creation
        let a = F3E::new(0b1111); // This exceeds 3 bits
        assert_eq!(*a.value(), 0b111); // Should be masked to 3 bits

        // Test large value
        let b = F3E::new(0xFFFFFFFF);
        assert_eq!(*b.value(), 0b111); // Should be masked to 3 bits
    }

    #[test]
    fn f2_add() {
        let a = F2E::new(0b10); // x
        let b = F2E::new(0b11); // x + 1
        assert_eq!(a + b, F2E::new(0b01));
    }

    #[test]
    fn f2_mul() {
        let a = F2E::new(0b00); // 0
        let b = F2E::new(0b01); // 1
        let c = F2E::new(0b10); // x
        let d = F2E::new(0b11); // x + 1
        assert_eq!(&a * &b, a);
        assert_eq!(&c * &d, b);
        assert_eq!(&c * &c, d);
        assert_eq!(&d * &d, c);
    }

    #[test]
    fn f2_pow() {
        let a = F2E::new(0b10);
        // Since the multiplicative group of GF(2²) is of order 2² - 1,
        // every element pow 3 is one.
        assert_eq!(a.pow(3u64), F2E::one());
    }

    #[test]
    fn f2_inv() {
        let a = F2E::new(0b10); // x
        let b = F2E::new(0b11); // x + 1
        assert_eq!(&a * a.inv().unwrap(), F2E::one());
        assert_eq!(a.inv().unwrap(), b);
    }

    #[test]
    fn f3_add() {
        let a = F3E::new(0b011);
        let b = F3E::new(0b100);
        let sum = a + b;
        assert_eq!(*sum.value(), 0b111);
    }

    #[test]
    fn f3_mul() {
        // a = x + 1
        let a = F3E::new(0b011);
        // prod = (x + 1)(x + 1) = x^2 + 2x + 1 = x^2 + 1
        let prod = a.clone() * a;
        assert_eq!(*prod.value(), 0b101);
    }

    #[test]
    fn f3_inv() {
        // Test inverse of zero is error.
        let zero = F3E::new(0);
        assert!(matches!(zero.inv(), Err(FieldError::InvZeroError)));

        // Test inverse of one is one.
        let one = F3E::one();
        assert_eq!(one.inv().unwrap(), one);

        // Test inverse of other elements
        let a = F3E::new(0b010);
        assert_eq!(&a * a.inv().unwrap(), F3E::one());
        let b = F3E::new(0b101);
        assert_eq!(&b * b.inv().unwrap(), F3E::one());
    }

    #[test]
    fn f3_pow() {
        //  x^0 = 1
        let a = F3E::new(0b010);
        assert_eq!(*a.pow(0 as u64).value(), 1);

        // x^1 = x
        assert_eq!(a.pow(1 as u64), a);

        //  x^2 = x * x
        assert_eq!(a.pow(2 as u64), a.clone() * a.clone());

        //  x^3 = x * x * x
        assert_eq!(a.pow(3 as u64), a.clone() * a.clone() * a.clone());

        //  x^7 = 1 in GF(2³) with primitive polynomial x³ + x + 1
        // This is because the multiplicative group of GF(2³) is of order is 2³-1 = 7.
        assert_eq!(*a.pow(7 as u64).value(), 1);
    }

    // Test not working. What should do to_binary_string()?
    #[test]
    fn test_to_binary_string() {
        let a = F3E::new(5);
        let b = F3E::new(2);
        assert_eq!(a.to_binary_string(), "101");
        //assert_eq!(b.to_binary_string(), "010");
    }
}
