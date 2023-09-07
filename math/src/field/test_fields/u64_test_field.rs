use crate::{
    errors::CreationError,
    field::errors::FieldError,
    field::traits::{IsFFTField, IsField, IsPrimeField},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct U64Field<const MODULUS: u64>;

impl<const MODULUS: u64> IsField for U64Field<MODULUS> {
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

impl<const MODULUS: u64> IsPrimeField for U64Field<MODULUS> {
    type RepresentativeType = u64;

    fn representative(x: &u64) -> u64 {
        *x
    }

    /// Returns how many bits do you need to represent the biggest field element
    /// It expects the MODULUS to be a Prime
    fn field_bit_size() -> usize {
        ((MODULUS - 1).ilog2() + 1) as usize
    }

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

        u64::from_str_radix(hex_string, 16).map_err(|_| CreationError::InvalidHexString)
    }
}

pub type U64TestField = U64Field<18446744069414584321>;

// These params correspond to the 18446744069414584321 modulus.
impl IsFFTField for U64TestField {
    const TWO_ADICITY: u64 = 32;
    const TWO_ADIC_PRIMITVE_ROOT_OF_UNITY: u64 = 1753635133440165772;
}

#[cfg(test)]
mod tests_u64_test_field {
    use crate::field::{test_fields::u64_test_field::U64TestField, traits::IsPrimeField};

    #[test]
    fn from_hex_for_b_is_11() {
        assert_eq!(U64TestField::from_hex("B").unwrap(), 11);
    }

    #[test]
    fn bit_size_of_test_field_is_64() {
        assert_eq!(
            <U64TestField as crate::field::traits::IsPrimeField>::field_bit_size(),
            64
        );
    }
}
