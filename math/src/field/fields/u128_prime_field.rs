use crate::field::element::FieldElement;
use crate::unsigned_integer::element::U128;
use crate::{field::traits::IsField, unsigned_integer::element::UnsignedInteger};
use std::fmt::Debug;

/// Type representing prime fields over unsigned 64-bit integers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct U128PrimeField<const MODULUS: u128>;
pub type U128FieldElement<const MODULUS: u128> = FieldElement<U128PrimeField<MODULUS>>;

impl<const MODULUS: u128> U128PrimeField<MODULUS> {
    const ZERO: U128 = UnsignedInteger::from_u64(0);
}

impl<const MODULUS: u128> IsField for U128PrimeField<MODULUS> {
    type BaseType = U128;

    fn add(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        let (sum, overflow) = UnsignedInteger::add(a, b);
        if !overflow {
            if sum < U128::from_u128(MODULUS) {
                sum
            } else {
                sum - U128::from_u128(MODULUS)
            }
        } else {
            let (diff, _) = UnsignedInteger::sub(&sum, &UnsignedInteger::<2>::from_u128(MODULUS));
            diff
        }
    }

    fn mul(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        a * b
    }

    fn sub(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        if b <= a {
            a - b
        } else {
            U128::from_u128(MODULUS) - (b - a)
        }
    }

    fn neg(a: &Self::BaseType) -> Self::BaseType {
        if a == &Self::ZERO {
            *a
        } else {
            U128::from_u128(MODULUS) - a
        }
    }

    fn inv(a: &Self::BaseType) -> Self::BaseType {
        if a == &Self::ZERO {
            panic!("Division by zero error.")
        }
        Self::pow(a, U128::from_u128(MODULUS) - Self::BaseType::from_u64(2))
    }

    fn div(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        Self::mul(a, &Self::inv(b))
    }

    fn eq(a: &Self::BaseType, b: &Self::BaseType) -> bool {
        a == b
    }

    fn zero() -> Self::BaseType {
        Self::ZERO
    }

    fn one() -> Self::BaseType {
        Self::from_u64(1)
    }

    fn from_u64(x: u64) -> Self::BaseType {
        U128::from_u64(x)
    }

    fn from_base_type(x: Self::BaseType) -> Self::BaseType {
        x
    }
}
