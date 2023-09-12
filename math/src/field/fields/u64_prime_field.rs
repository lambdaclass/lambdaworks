use crate::cyclic_group::IsGroup;
#[cfg(feature = "std")]
use crate::errors::ByteConversionError::{FromBEBytesError, FromLEBytesError};
use crate::errors::CreationError;
#[cfg(feature = "std")]
use crate::errors::DeserializationError;
use crate::field::element::FieldElement;
use crate::field::errors::FieldError;
use crate::field::traits::{IsFFTField, IsField, IsPrimeField};
#[cfg(feature = "std")]
use crate::traits::{ByteConversion, Deserializable, Serializable};

/// Type representing prime fields over unsigned 64-bit integers.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct U64PrimeField<const MODULUS: u64>;
pub type U64FieldElement<const MODULUS: u64> = FieldElement<U64PrimeField<MODULUS>>;

pub type F17 = U64PrimeField<17>;
pub type FE17 = U64FieldElement<17>;

impl IsFFTField for F17 {
    const TWO_ADICITY: u64 = 4;
    const TWO_ADIC_PRIMITVE_ROOT_OF_UNITY: u64 = 3;
}

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
        Self::mul(a, &Self::inv(b).unwrap())
    }

    fn inv(a: &u64) -> Result<u64, FieldError> {
        if *a == 0 {
            return Err(FieldError::InvZeroError);
        }
        Ok(Self::pow(a, MODULUS - 2))
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
}

impl<const MODULUS: u64> Copy for U64FieldElement<MODULUS> {}

impl<const MODULUS: u64> IsPrimeField for U64PrimeField<MODULUS> {
    type RepresentativeType = u64;

    fn representative(x: &u64) -> u64 {
        *x
    }

    /// Returns how many bits do you need to represent the biggest field element
    /// It expects the MODULUS to be a Prime
    fn field_bit_size() -> usize {
        ((MODULUS - 1).ilog2() + 1) as usize
    }

    fn from_hex(hex_string: &str) -> Result<Self::BaseType, CreationError> {
        let mut hex_string = hex_string;
        // Remove 0x if it's on the string
        let mut char_iterator = hex_string.chars();
        if hex_string.len() > 2
            && char_iterator.next().unwrap() == '0'
            && char_iterator.next().unwrap() == 'x'
        {
            hex_string = &hex_string[2..];
        }

        u64::from_str_radix(hex_string, 16).map_err(|_| CreationError::InvalidHexString)
    }
}

/// Represents an element in Fp. (E.g: 0, 1, 2 are the elements of F3)
impl<const MODULUS: u64> IsGroup for U64FieldElement<MODULUS> {
    fn neutral_element() -> U64FieldElement<MODULUS> {
        U64FieldElement::zero()
    }

    fn operate_with(&self, other: &Self) -> Self {
        *self + *other
    }

    fn neg(&self) -> Self {
        -self
    }
}

#[cfg(feature = "std")]
impl<const MODULUS: u64> ByteConversion for U64FieldElement<MODULUS> {
    #[cfg(feature = "std")]
    fn to_bytes_be(&self) -> Vec<u8> {
        u64::to_be_bytes(*self.value()).into()
    }

    #[cfg(feature = "std")]
    fn to_bytes_le(&self) -> Vec<u8> {
        u64::to_le_bytes(*self.value()).into()
    }

    fn from_bytes_be(bytes: &[u8]) -> Result<Self, crate::errors::ByteConversionError> {
        let bytes: [u8; 8] = bytes[0..8].try_into().map_err(|_| FromBEBytesError)?;
        Ok(Self::from(u64::from_be_bytes(bytes)))
    }

    fn from_bytes_le(bytes: &[u8]) -> Result<Self, crate::errors::ByteConversionError> {
        let bytes: [u8; 8] = bytes[0..8].try_into().map_err(|_| FromLEBytesError)?;
        Ok(Self::from(u64::from_le_bytes(bytes)))
    }
}

#[cfg(feature = "std")]
impl<const MODULUS: u64> Serializable for FieldElement<U64PrimeField<MODULUS>> {
    fn serialize(&self) -> Vec<u8> {
        self.to_bytes_be()
    }
}

#[cfg(feature = "std")]
impl<const MODULUS: u64> Deserializable for FieldElement<U64PrimeField<MODULUS>> {
    fn deserialize(bytes: &[u8]) -> Result<Self, DeserializationError>
    where
        Self: Sized,
    {
        Self::from_bytes_be(bytes).map_err(|x| x.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const MODULUS: u64 = 13;
    type F = U64PrimeField<MODULUS>;
    type FE = FieldElement<F>;

    #[test]
    fn from_hex_for_b_is_11() {
        assert_eq!(F::from_hex("B").unwrap(), 11);
    }

    #[test]
    fn from_hex_for_0x1_a_is_26() {
        assert_eq!(F::from_hex("0x1a").unwrap(), 26);
    }

    #[test]
    fn bit_size_of_mod_13_field_is_4() {
        assert_eq!(
            <U64PrimeField<MODULUS> as crate::field::traits::IsPrimeField>::field_bit_size(),
            4
        );
    }

    #[test]
    fn bit_size_of_big_mod_field_is_64() {
        const MODULUS: u64 = 10000000000000000000;
        assert_eq!(
            <U64PrimeField<MODULUS> as crate::field::traits::IsPrimeField>::field_bit_size(),
            64
        );
    }

    #[test]
    fn bit_size_of_63_bit_mod_field_is_63() {
        const MODULUS: u64 = 9000000000000000000;
        assert_eq!(
            <U64PrimeField<MODULUS> as crate::field::traits::IsPrimeField>::field_bit_size(),
            63
        );
    }

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
    fn inv_0_error() {
        let result = FE::new(0).inv();
        assert!(matches!(result, Err(FieldError::InvZeroError)));
    }

    #[test]
    fn inv_2() {
        let a: FE = FE::new(2);
        assert_eq!(a * a.inv().unwrap(), FE::new(1));
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
    #[cfg(feature = "std")]
    fn to_bytes_from_bytes_be_is_the_identity() {
        let x = FE::new(12345);
        assert_eq!(FE::from_bytes_be(&x.to_bytes_be()).unwrap(), x);
    }

    #[test]
    #[cfg(feature = "std")]
    fn from_bytes_to_bytes_be_is_the_identity_for_one() {
        let bytes = [0, 0, 0, 0, 0, 0, 0, 1];
        assert_eq!(FE::from_bytes_be(&bytes).unwrap().to_bytes_be(), bytes);
    }

    #[test]
    #[cfg(feature = "std")]
    fn to_bytes_from_bytes_le_is_the_identity() {
        let x = FE::new(12345);
        assert_eq!(FE::from_bytes_le(&x.to_bytes_le()).unwrap(), x);
    }

    #[test]
    #[cfg(feature = "std")]
    fn from_bytes_to_bytes_le_is_the_identity_for_one() {
        let bytes = [1, 0, 0, 0, 0, 0, 0, 0];
        assert_eq!(FE::from_bytes_le(&bytes).unwrap().to_bytes_le(), bytes);
    }

    #[test]
    fn creating_a_field_element_from_its_representative_returns_the_same_element_1() {
        let change = 1;
        let f1 = FE::new(MODULUS + change);
        let f2 = FE::new(f1.representative());
        assert_eq!(f1, f2);
    }

    #[test]
    fn creating_a_field_element_from_its_representative_returns_the_same_element_2() {
        let change = 8;
        let f1 = FE::new(MODULUS + change);
        let f2 = FE::new(f1.representative());
        assert_eq!(f1, f2);
    }
}
