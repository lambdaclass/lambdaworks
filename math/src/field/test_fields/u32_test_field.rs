use crate::{
    errors::CreationError,
    field::errors::FieldError,
    field::traits::{IsFFTField, IsField, IsPrimeField},
};

#[derive(Debug, Clone, PartialEq, Eq)]

pub struct U32Field<const MODULUS: u32>;

impl<const MODULUS: u32> IsField for U32Field<MODULUS> {
    type BaseType = u32;

    fn add(a: &u32, b: &u32) -> u32 {
        ((*a as u128 + *b as u128) % MODULUS as u128) as u32
    }

    fn sub(a: &u32, b: &u32) -> u32 {
        (((*a as u128 + MODULUS as u128) - *b as u128) % MODULUS as u128) as u32
    }

    fn neg(a: &u32) -> u32 {
        MODULUS - a
    }

    fn mul(a: &u32, b: &u32) -> u32 {
        ((*a as u128 * *b as u128) % MODULUS as u128) as u32
    }

    fn div(a: &u32, b: &u32) -> u32 {
        Self::mul(a, &Self::inv(b).unwrap())
    }

    fn inv(a: &u32) -> Result<u32, FieldError> {
        if *a == 0 {
            return Err(FieldError::InvZeroError);
        }
        Ok(Self::pow(a, MODULUS - 2))
    }

    fn eq(a: &u32, b: &u32) -> bool {
        Self::from_base_type(*a) == Self::from_base_type(*b)
    }

    fn zero() -> u32 {
        0
    }

    fn one() -> u32 {
        1
    }

    fn from_u64(x: u64) -> u32 {
        (x % MODULUS as u64) as u32
    }

    fn from_base_type(x: u32) -> u32 {
        x % MODULUS
    }
}

impl<const MODULUS: u32> IsPrimeField for U32Field<MODULUS> {
    type RepresentativeType = u32;

    fn representative(a: &Self::BaseType) -> u32 {
        *a
    }

    /// Returns how many bits do you need to represent the biggest field element
    /// It expects the MODULUS to be a Prime
    fn field_bit_size() -> usize {
        ((MODULUS - 1).ilog2() + 1) as usize
    }

    /// Unimplemented for test fields
    fn from_hex(hex_string: &str) -> Result<Self::BaseType, crate::errors::CreationError> {
        let mut hex_string = hex_string;
        // Remove 0x if it's on the string
        let mut char_iterator = hex_string.chars();
        if hex_string.len() > 2
            && char_iterator.next().unwrap() == '0'
            && char_iterator.next().unwrap() == 'x'
        {
            hex_string = &hex_string[2..];
        }

        u32::from_str_radix(hex_string, 16).map_err(|_| CreationError::InvalidHexString)
    }
}

// 15 * 2^27 + 1;
pub type U32TestField = U32Field<2013265921>;

// These params correspond to the 2013265921 modulus.
impl IsFFTField for U32TestField {
    const TWO_ADICITY: u64 = 27;
    const TWO_ADIC_PRIMITVE_ROOT_OF_UNITY: u32 = 440532289;
}

#[cfg(test)]
mod tests_u32_test_field {
    use crate::field::{test_fields::u32_test_field::U32TestField, traits::IsPrimeField};

    #[test]
    fn from_hex_for_b_is_11() {
        assert_eq!(U32TestField::from_hex("B").unwrap(), 11);
    }

    #[test]
    fn bit_size_of_test_field_is_31() {
        assert_eq!(
            <U32TestField as crate::field::traits::IsPrimeField>::field_bit_size(),
            31
        );
    }
}
