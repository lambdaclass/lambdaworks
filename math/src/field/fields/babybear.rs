use crate::field::traits::IsField;
use crate::unsigned_integer::element::UnsignedInteger;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct P31BabyBearPrimeField;

//Babybear Prime p = 2^31 - 2^27 + 1
//0x78000001
pub const P31_BABYBEAR_PRIME_FIELD_ORDER: u64 = 2013265921;

// 31 bit unsiged integer represented as u64.
impl IsField for P31BabyBearPrimeField {
    type BaseType = u64;

    fn add(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        let mut c = a+b;
        Self::weak_reduce(&mut c);
        c
    }

    fn mul(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        let mut c = a*b;
        Self::weak_reduce(&mut c);
        c
    }

    fn sub(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        let mask: u64= 1u64 << 32 -1;
        let co2: u64 = (b ^ mask) + 1; //2's complement of b
        let mut c: u64 = a + co2;
        Self::weak_reduce(&mut c);
        c
    }

    fn neg(a: &Self::BaseType) -> Self::BaseType {
        let zero: u64 = Self::zero();
        Self::sub(&zero, a)
    }

    fn inv(a: &Self::BaseType) -> Self::BaseType {
        Self::pow(a, P31_BABYBEAR_PRIME_FIELD_ORDER-2)
    }

    fn div(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        let b_inv = Self::inv(b);
        Self::mul(a, &b_inv)
    }

    fn eq(a: &Self::BaseType, b: &Self::BaseType) -> bool {
        a == b
    }

    fn zero() -> Self::BaseType {
        0u64
    }

    fn one() -> Self::BaseType {
        1u64
    }

    fn from_u64(x: u64) -> Self::BaseType {
        x
    }

    fn from_base_type(x: Self::BaseType) -> Self::BaseType {
        let mut x = x;
        Self::weak_reduce(&mut x);
        x
    }


}

impl P31BabyBearPrimeField {
    fn weak_reduce(a: &mut <P31BabyBearPrimeField as IsField>::BaseType) {
        *a = *a % P31_BABYBEAR_PRIME_FIELD_ORDER;
    }
}