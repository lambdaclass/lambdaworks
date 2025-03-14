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

    // TODO: Choose the mul algorithm.
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

    // TODO: Choose the inv algorithm.
    // Using Fermat's little therome we know that a^{-1} = a^{q-2} with q the order of the field.
    // https://planetmath.org/fermatslittletheorem
    // If the extension is of degree n, then the field has order 2^{2^n} or 2^n ????.
    fn inv(a: &Self::BaseType) -> Result<Self::BaseType, FieldError> {
        // TODO: what if a == mod.
        if *a == 0 {
            return Err(FieldError::InvZeroError);
        }
        let exponent = 1 << C::DEGREE - 2; //x^3 + x + 1
        Ok(Self::pow(a, exponent as u128))
        // if *a == 0 {
        //     return Err(FieldError::InvZeroError);
        // }
        // let mut t = 0u64;
        // let mut newt = 1u64;
        // let mut r = C::PRIMITIVE_POLY as u64;
        // let mut newr = *a;

        // while newr != 0 {
        //     let deg_r = 63 - r.leading_zeros();
        //     let deg_newr = 63 - newr.leading_zeros();
        //     if deg_r < deg_newr {
        //         core::mem::swap(&mut t, &mut newt);
        //         core::mem::swap(&mut r, &mut newr);
        //         continue;
        //     }
        //     if let Some(shift) = deg_r.checked_sub(deg_newr) {
        //         r ^= newr << shift;
        //         t ^= newt << shift;
        //     } else {
        //         core::mem::swap(&mut t, &mut newt);
        //         core::mem::swap(&mut r, &mut newr);
        //     }
        // }

        // if r != 1 {
        //     return Err(FieldError::InvZeroError);
        // }

        // Ok(t)
    }

    // fn inv(&self) -> Result<Self, BinaryFieldError> {
    //     if self.is_zero() {
    //         return Err(BinaryFieldError::InverseOfZero);
    //     }

    //     // For very small fields or odd degree fields, use Fermat's little theorem
    //     if F::DEGREE <= 2 || F::DEGREE % 2 != 0 {
    //         // Use Fermat's little theorem:
    //         // In GF(2^n), x^(2^n-1) = 1 for any non-zero x
    //         // Therefore x^(2^n-2) is the multiplicative inverse
    //         return Ok(self.pow((1 << F::DEGREE) - 2));
    //     }

    //     // For larger even degree fields, use recursive algorithm
    //     let (a_hi, a_lo) = self.split();

    //     // Compute k = n/2 where n is the field degree
    //     let k = F::DEGREE / 2;

    //     // Compute 2^(k-1) as a field element
    //     let two_pow_k_minus_one = Self::new(1 << (k - 1));

    //     // a = a_hi * x^k + a_lo
    //     // a_lo_next = a_hi * x^(k-1) + a_lo
    //     let a_lo_next = a_lo.clone() + a_hi.clone() * two_pow_k_minus_one;

    //     // Δ = a_lo * a_lo_next + a_hi^2
    //     let delta = a_lo.clone() * a_lo_next.clone() + a_hi.clone() * a_hi.clone();

    //     // Compute inverse of delta recursively
    //     let delta_inverse = delta.inv()?;

    //     // Compute parts of the inverse
    //     let out_hi = delta_inverse.clone() * a_hi;
    //     let out_lo = delta_inverse * a_lo_next;

    //     // Join the parts to get the final inverse
    //     Ok(out_hi.join(&out_lo))
    // }

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

    /// Returns the total number of bits.
    pub fn num_bits(&self) -> usize {
        1 << C::DEGREE
    }

    /// Returns binary string representation.
    /// The string length is equal to num_bits.
    pub fn to_binary_string(&self) -> String {
        format!("{:0width$b}", self.value(), width = self.num_bits())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Example configuration for GF(2³)
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct GF2_3;
    impl BinaryFieldConfig for GF2_3 {
        const DEGREE: u32 = 3;
        const PRIMITIVE_POLY: u128 = 0b1011; // x³ + x + 1
    }

    pub type F = BinaryField<GF2_3>;
    pub type FE = FieldElement<F>;

    #[test]
    fn test_basic_binary_field_addition() {
        let a = FE::new(0b011);
        let b = FE::new(0b100);
        let sum = a + b;
        assert_eq!(*sum.value(), 0b111);
    }

    #[test]
    fn test_basic_binary_field_multiplication() {
        let a = FE::new(0b011);
        let prod = a.clone() * a;
        assert_eq!(*prod.value(), 0b101);
    }

    /// Tests field properties for BinaryFieldElement
    #[test]
    fn test_binary_field_properties() {
        // Test addition with zero
        let a = FE::new(0b101);
        assert_eq!(a.clone() + FE::zero(), a);

        // Test multiplication with one
        assert_eq!(a.clone() * FE::one(), a);

        // Test negation (identity in characteristic 2)
        assert_eq!(-a.clone(), a);

        // Test subtraction equals addition
        let b = FE::new(0b110);
        assert_eq!(a.clone() - b.clone(), a.clone() + b);
    }

    /// Tests boundary conditions for BinaryFieldElement
    #[test]
    fn test_binary_field_boundaries() {
        // Test value masking on creation
        let a = FE::new(0b1111); // This exceeds 3 bits
        assert_eq!(*a.value(), 0b111); // Should be masked to 3 bits

        // Test large value
        let b = FE::new(0xFFFFFFFF);
        assert_eq!(*b.value(), 0b111); // Should be masked to 3 bits
    }

    #[test]
    fn test_binary_field_inverse() {
        // Test inverse of zero
        let zero = FE::new(0);
        //assert!(matches!(zero.inv(), Err(FieldError::InvZeroError)));

        // Test inverse of one
        let one = FE::one();
        //assert_eq!(one.inv().unwrap(), one);

        // Test inverse of other elements
        let a = FE::new(0b010);
        assert_eq!(&a * a.inv().unwrap(), FE::one());

        let b = FE::new(0b011);
        //assert_eq!(&b * b.inv().unwrap(), FE::one());
    }

    #[test]
    fn test_binary_field_pow() {
        //  x^0 = 1
        let a = FE::new(0b010);
        assert_eq!(*a.pow(0 as u64).value(), 1);

        // x^1 = x
        assert_eq!(a.pow(1 as u64), a);

        //  x^2
        assert_eq!(a.pow(2 as u64), a.clone() * a.clone());

        //  x^3
        assert_eq!(a.pow(3 as u64), a.clone() * a.clone() * a.clone());

        //  x^7 = 1 in GF(2³) with primitive polynomial x³ + x + 1
        // This is because the multiplicative order of GF(2³)* is 2³-1 = 7
        assert_eq!(*a.pow(7 as u64).value(), 1);
    }

    // #[test]
    // fn test_tower_new_and_bin() {
    //     let elem = FE::new(5, Some(3)); // 3 levels => 8 bits
    //     assert_eq!(elem.num_bits(), 8); // Check the number of bits
    //     assert_eq!(elem.to_binary_string(), "00000101"); // Check binary representation
    // }

    // #[test]
    // fn test_tower_addition() {
    //     let a = TowerFieldElement::new(0b011, Some(3)); // 8-bit representation: 00000011
    //     let b = TowerFieldElement::new(0b100, Some(3)); // 00000100
    //     let sum = a + b; // 00000011 XOR 00000100 = 00000111
    //     assert_eq!(sum.value(), 0b111);
    //     assert_eq!(sum.num_levels(), 3);
    // }

    // #[test]
    // fn test_tower_multiplication() {
    //     let a = TowerFieldElement::new(0b011, Some(2)); // 4-bit representation: 0011
    //     let b = TowerFieldElement::new(0b010, Some(2)); // 0010
    //     let prod = a * b;
    //     assert_eq!(prod.value(), 0b110); // Check multiplication result
    //     assert_eq!(prod.num_levels(), 2);
    // }

    // #[test]
    // fn test_tower_split_join() {
    //     let elem = TowerFieldElement::new(0b11001010, Some(3)); // 8 bits
    //     let (hi, lo) = elem.split();

    //     // Check that split produces correct high and low parts
    //     assert_eq!(hi.value(), 0b1100);
    //     assert_eq!(lo.value(), 0b1010);
    //     assert_eq!(hi.num_levels(), 2);
    //     assert_eq!(lo.num_levels(), 2);

    //     // Check that joining them produces the original element
    //     let joined = hi.join(&lo);
    //     assert_eq!(joined.value(), elem.value());
    //     assert_eq!(joined.num_levels(), elem.num_levels());
    // }

    // #[test]
    // fn test_tower_field_properties() {
    //     let zero = TowerFieldElement::new(0, Some(3));
    //     let one = TowerFieldElement::new(1, Some(3));
    //     let a = TowerFieldElement::new(0b101, Some(3));
    //     let b = TowerFieldElement::new(0b011, Some(3));

    //     // Additive identity
    //     assert_eq!((a.clone() + zero.clone()).value(), a.value());

    //     // Multiplicative identity
    //     assert_eq!((a.clone() * one.clone()).value(), a.value());

    //     // Commutativity of addition
    //     assert_eq!(a.clone() + b.clone(), b.clone() + a.clone());

    //     // Commutativity of multiplication
    //     assert_eq!(a.clone() * b.clone(), b.clone() * a.clone());

    //     // Associativity of addition
    //     let c = TowerFieldElement::new(0b110, Some(3));
    //     assert_eq!(
    //         (a.clone() + b.clone()) + c.clone(),
    //         a.clone() + (b.clone() + c.clone())
    //     );

    //     // Distributivity
    //     assert_eq!(
    //         a.clone() * (b.clone() + c.clone()),
    //         (a.clone() * b.clone()) + (a.clone() * c.clone())
    //     );
    // }

    // #[test]
    // fn test_tower_pow() {
    //     let a = TowerFieldElement::new(0b010, Some(2)); // 4-bit element

    //     // Test x^0 = 1
    //     assert_eq!(a.pow(0).value(), 1);

    //     // Test x^1 = x
    //     assert_eq!(a.pow(1), a);

    //     // Test x^2
    //     let squared = a.clone() * a.clone();
    //     assert_eq!(a.pow(2), squared);

    //     // Test x^3
    //     assert_eq!(a.pow(3), squared * a);
    // }

    // #[test]
    // fn test_tower_mixed_levels() {
    //     let a = TowerFieldElement::new(0b11, Some(2)); // 4-bit element
    //     let b = TowerFieldElement::new(0b1101, Some(3)); // 8-bit element

    //     // Addition should use the maximum level
    //     let sum = a.clone() + b.clone();
    //     assert_eq!(sum.num_levels(), 3);
    //     assert_eq!(sum.value(), 0b1101 ^ 0b11);

    //     // Multiplication should use the maximum level
    //     let prod = a * b;
    //     assert_eq!(prod.num_levels(), 3);
    // }

    // /// Tests boundary conditions and edge cases
    // #[test]
    // fn test_tower_boundaries() {
    //     // Test with maximum value for 3 levels (8 bits)
    //     let max_val = TowerFieldElement::new(0xFF, Some(3));
    //     assert_eq!(max_val.num_bits(), 8);
    //     assert_eq!(max_val.value(), 0xFF);

    //     // Test with value exceeding bit length
    //     let overflow = TowerFieldElement::new(0x1FF, Some(3));
    //     assert_eq!(overflow.value(), 0xFF); // Should be masked to 8 bits

    //     // Test with minimum level
    //     let min_level = TowerFieldElement::new(0b11, Some(1));
    //     assert_eq!(min_level.num_bits(), 2);
    //     assert_eq!(min_level.value(), 0b11);
    // }

    // #[test]
    // fn test_tower_inverse() {
    //     let a = TowerFieldElement::new(0b011, Some(2)); // 4-bit element

    //     // Test inverse of zero
    //     let zero = TowerFieldElement::new(0, Some(2));
    //     assert!(zero.inv().is_err());
    //     assert_eq!(zero.inv().unwrap_err(), BinaryFieldError::InverseOfZero);

    //     // Test inverse of one
    //     let one = TowerFieldElement::new(1, Some(2));
    //     assert_eq!(one.inv().unwrap(), one);

    //     // Test inverse of non-zero element
    //     let a_inv = a.inv().unwrap();
    //     assert_eq!((a * a_inv).value(), 1);
    // }

    // #[test]
    // fn test_tower_division() {
    //     // Crear elementos en GF(2^4)
    //     let one = TowerFieldElement::new(1, Some(2));
    //     let two = TowerFieldElement::new(2, Some(2));
    //     let three = TowerFieldElement::new(3, Some(2));
    //     let four = TowerFieldElement::new(4, Some(2));
    //     let zero = TowerFieldElement::new(0, Some(2));

    //     // Test básico: división por uno
    //     assert_eq!(three.div(&one).unwrap(), three);

    //     // Test división de elemento por sí mismo
    //     assert_eq!(three.div(&three).unwrap(), one);

    //     // Test división por cero
    //     assert!(three.div(&zero).is_err());
    //     assert_eq!(
    //         three.div(&zero).unwrap_err(),
    //         BinaryFieldError::DivisionByZero
    //     );

    //     // Test división con niveles incompatibles
    //     let big_elem = TowerFieldElement::new(2, Some(3));
    //     assert!(three.div(&big_elem).is_err());
    //     assert_eq!(
    //         three.div(&big_elem).unwrap_err(),
    //         BinaryFieldError::IncompatibleFields
    //     );

    //     // Test operaciones básicas de división
    //     // Primero: 4 = 2 * 2
    //     assert_eq!(two.clone() * two.clone(), four);
    //     // Ahora: 4 / 2 = 2
    //     assert_eq!(four.div(&two).unwrap(), two);
    // }

    // #[test]
    // fn test_binary_field_division() {
    //     #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    //     struct GF2_3;
    //     impl BinaryFieldConfig for GF2_3 {
    //         const DEGREE: u32 = 3;
    //         const PRIMITIVE_POLY: u128 = 0b1011; // x³ + x + 1
    //     }
    //     type Elem = BinaryFieldElement<GF2_3>;

    //     // Test division by zero
    //     let a = Elem::new(0b010);
    //     let zero = Elem::new(0);
    //     assert!(a.div(&zero).is_err());
    //     assert_eq!(a.div(&zero).unwrap_err(), BinaryFieldError::DivisionByZero);

    //     // Test division by one
    //     let one = Elem::new(1);
    //     assert_eq!(a.div(&one).unwrap(), a);

    //     // Test division of a by itself (should be 1)
    //     assert_eq!(a.div(&a).unwrap(), one);

    //     // Test general division
    //     let b = Elem::new(0b011);
    //     let c = a.clone() * b.clone();
    //     assert_eq!(c.div(&b).unwrap(), a);
    // }
}
