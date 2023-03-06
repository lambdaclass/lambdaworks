use crate::field::traits::{IsField, IsTwoAdicField};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct U64TestField<const MODULUS: u64>;

impl<const MODULUS: u64> IsField for U64TestField<MODULUS> {
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
        assert_ne!(*a, 0, "Cannot invert zero element");
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

impl<const MODULUS: u64> IsTwoAdicField for U64TestField<MODULUS> {
    const TWO_ADICITY: u64 = 32;
    const TWO_ADIC_PRIMITVE_ROOT_OF_UNITY: u64 = 1753635133440165772;
    const GENERATOR: u64 = 7;
}
