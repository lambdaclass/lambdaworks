use crate::cyclic_group::IsGroup;
use crate::errors::ByteConversionError::{FromBEBytesError, FromLEBytesError};
use crate::field::element::FieldElement;
use crate::field::traits::IsField;
use crate::traits::ByteConversion;

/// Type representing prime fields over unsigned 64-bit integers.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct U64PrimeField<const MODULUS: u64>;
pub type U64FieldElement<const MODULUS: u64> = FieldElement<U64PrimeField<MODULUS>>;

impl<const MODULUS: u64> IsField for U64PrimeField<MODULUS> {
    type BaseType = u64;

    fn add(a: &u64, b: &u64) -> u64 {
        ((*a as u128 + *b as u128) % MODULUS as u128) as u64
    }

    fn sub(a: &u64, b: &u64) -> u64 {
        (((*a as u128 + MODULUS as u128) - *b as u128) % MODULUS as u128) as u64
    }

    fn neg(a: &u64) -> u64 {
        MODULUS - a
    }

    fn mul(a: &u64, b: &u64) -> u64 {
        ((*a as u128 * *b as u128) % MODULUS as u128) as u64
    }

    fn div(a: &u64, b: &u64) -> u64 {
        Self::mul(a, &Self::inv(b))
    }

    fn inv(a: &u64) -> u64 {
        debug_assert_ne!(*a, 0, "Cannot invert zero element");
        Self::pow(a, MODULUS - 2)
    }

    fn eq(a: &u64, b: &u64) -> bool {
        Self::from_u64(*a) == Self::from_u64(*b)
    }

    fn zero() -> u64 {
        0
    }

    fn one() -> u64 {
        1
    }

    fn from_u64(x: u64) -> u64 {
        x % MODULUS
    }

    fn from_base_type(x: u64) -> u64 {
        Self::from_u64(x)
    }

    fn representative(x: u64) -> u64 {
        x
    }
}

impl<const MODULUS: u64> Copy for U64FieldElement<MODULUS> {}

/// Represents an element in Fp. (E.g: 0, 1, 2 are the elements of F3)
impl<const MODULUS: u64> IsGroup for U64FieldElement<MODULUS> {
    fn neutral_element() -> U64FieldElement<MODULUS> {
        U64FieldElement::zero()
    }

    fn operate_with(&self, other: &Self) -> Self {
        *self + *other
    }
}

impl<const MODULUS: u64> ByteConversion for U64FieldElement<MODULUS> {
    fn to_bytes_be(&self) -> Vec<u8> {
        u64::to_be_bytes(*self.value()).into()
    }

    fn to_bytes_le(&self) -> Vec<u8> {
        u64::to_le_bytes(*self.value()).into()
    }

    fn from_bytes_be(bytes: &[u8]) -> Result<Self, crate::errors::ByteConversionError> {
        let bytes: [u8; 8] = bytes.try_into().map_err(|_| FromBEBytesError)?;
        Ok(Self::from(u64::from_be_bytes(bytes)))
    }

    fn from_bytes_le(bytes: &[u8]) -> Result<Self, crate::errors::ByteConversionError> {
        let bytes: [u8; 8] = bytes.try_into().map_err(|_| FromLEBytesError)?;
        Ok(Self::from(u64::from_le_bytes(bytes)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const MODULUS: u64 = 13;
    type FE = FieldElement<U64PrimeField<MODULUS>>;

    #[test]
    fn two_plus_one_is_three() {
        assert_eq!(FE::new(2) + FE::new(1), FE::new(3));
    }

    #[test]
    fn max_order_plus_1_is_0() {
        assert_eq!(FE::new(MODULUS - 1) + FE::new(1), FE::new(0));
    }

    #[test]
    fn when_comparing_13_and_13_they_are_equal() {
        let a: FE = FE::new(13);
        let b: FE = FE::new(13);
        assert_eq!(a, b);
    }

    #[test]
    fn when_comparing_13_and_8_they_are_different() {
        let a: FE = FE::new(13);
        let b: FE = FE::new(8);
        assert_ne!(a, b);
    }

    #[test]
    fn mul_neutral_element() {
        let a: FE = FE::new(1);
        let b: FE = FE::new(2);
        assert_eq!(a * b, FE::new(2));
    }

    #[test]
    fn mul_2_3_is_6() {
        let a: FE = FE::new(2);
        let b: FE = FE::new(3);
        assert_eq!(a * b, FE::new(6));
    }

    #[test]
    fn mul_order_minus_1() {
        let a: FE = FE::new(MODULUS - 1);
        let b: FE = FE::new(MODULUS - 1);
        assert_eq!(a * b, FE::new(1));
    }

    #[test]
    #[should_panic]
    fn inv_0_error() {
        FE::new(0).inv();
    }

    #[test]
    fn inv_2() {
        let a: FE = FE::new(2);
        assert_eq!(a * a.inv(), FE::new(1));
    }

    #[test]
    fn pow_2_3() {
        assert_eq!(FE::new(2).pow(3_u64), FE::new(8))
    }

    #[test]
    fn pow_p_minus_1() {
        assert_eq!(FE::new(2).pow(MODULUS - 1), FE::new(1))
    }

    #[test]
    fn div_1() {
        assert_eq!(FE::new(2) / FE::new(1), FE::new(2))
    }

    #[test]
    fn div_4_2() {
        assert_eq!(FE::new(4) / FE::new(2), FE::new(2))
    }

    #[test]
    fn div_4_3() {
        assert_eq!(FE::new(4) / FE::new(3) * FE::new(3), FE::new(4))
    }

    #[test]
    fn two_plus_its_additive_inv_is_0() {
        let two = FE::new(2);

        assert_eq!(two + (-two), FE::new(0))
    }

    #[test]
    fn four_minus_three_is_1() {
        let four = FE::new(4);
        let three = FE::new(3);

        assert_eq!(four - three, FE::new(1))
    }

    #[test]
    fn zero_minus_1_is_order_minus_1() {
        let zero = FE::new(0);
        let one = FE::new(1);

        assert_eq!(zero - one, FE::new(MODULUS - 1))
    }

    #[test]
    fn neg_zero_is_zero() {
        let zero = FE::new(0);

        assert_eq!(-zero, zero);
    }

    #[test]
    fn zero_constructor_returns_zero() {
        assert_eq!(FE::new(0), FE::new(0));
    }

    #[test]
    fn field_element_as_group_element_multiplication_by_scalar_works_as_multiplication_in_finite_fields(
    ) {
        let a = FE::new(3);
        let b = FE::new(12);
        assert_eq!(a * b, a.operate_with_self(12_u16));
    }

    #[test]
    fn to_bytes_from_bytes_be_is_the_identity() {
        let x = FE::new(12345);
        assert_eq!(FE::from_bytes_be(&x.to_bytes_be()).unwrap(), x);
    }

    #[test]
    fn from_bytes_to_bytes_be_is_the_identity_for_one() {
        let bytes = vec![0, 0, 0, 0, 0, 0, 0, 1];
        assert_eq!(FE::from_bytes_be(&bytes).unwrap().to_bytes_be(), bytes);
    }

    #[test]
    fn to_bytes_from_bytes_le_is_the_identity() {
        let x = FE::new(12345);
        assert_eq!(FE::from_bytes_le(&x.to_bytes_le()).unwrap(), x);
    }

    #[test]
    fn from_bytes_to_bytes_le_is_the_identity_for_one() {
        let bytes = vec![1, 0, 0, 0, 0, 0, 0, 0];
        assert_eq!(FE::from_bytes_le(&bytes).unwrap().to_bytes_le(), bytes);
    }
}
