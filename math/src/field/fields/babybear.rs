use crate::field::traits::IsField;
//use crate::unsigned_integer::element::UnsignedInteger;

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
        if a>= b {
            a-b
        } else {
            let c = *a as i64 - *b as i64 + P31_BABYBEAR_PRIME_FIELD_ORDER as i64;
            c as u64
        }
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
        let mut x: u64 = x;
        Self::weak_reduce(&mut x);
        x
    }

    fn from_base_type(x: Self::BaseType) -> Self::BaseType {
        let mut x: u64 = x;
        Self::weak_reduce(&mut x);
        x
    }


}

impl P31BabyBearPrimeField {
    fn weak_reduce(a: &mut <P31BabyBearPrimeField as IsField>::BaseType) {
        *a = *a % P31_BABYBEAR_PRIME_FIELD_ORDER;
    }
}

// Tests section
#[cfg(test)]
mod tests{
    use super::*;

    #[test]
    fn p31_add_test_1(){
        let num1: u64 = 1297749315;
        let num2: u64 = 2610772852;
        let num3 = P31BabyBearPrimeField::add(&num1, &num2);
        assert_eq!(num3, 1895256246);
    }

    #[test]
    fn p31_sub_test_1() {
        let num1: u64 = 1297749315;
        let num2: u64 = 2610772852;
        let num3 = P31BabyBearPrimeField::sub(&num1, &num2);
        assert_eq!(num3, 700242384);
    }

    #[test]
    fn p31_sub_test_2() {
        let num2: u64 = 1297749315;
        let num1: u64 = 2610772852;
        let num3 = P31BabyBearPrimeField::sub(&num1, &num2);
        assert_eq!(num3, 1313023537);
    }

    #[test]
    fn p31_neg_test_1() {
        let num1: u64 = 1383673337;
        let num2: u64 = P31BabyBearPrimeField::neg(&num1);
        assert_eq!(num2, 629592584);
    }

    #[test]
    fn p31_mul_test1() {
        let num1: u64 = 1529643217;
        let num2: u64 =  732012185;
        let num3: u64 = P31BabyBearPrimeField::mul(&num1, &num2);
        assert_eq!(num3,442794260 );
    }

    #[test]
    fn p31_pow_test_1() {
        let num1: u64 = 1058320007;
        let num2: u64 = P31BabyBearPrimeField::pow(&num1, 65537u64);
        debug_assert_eq!(num2, 1888086939);
    }

    #[test]
    fn p31_inv_test_1() {
        let num1: u64 = 429803738;
        let num2: u64 = 1959415436;
        let num3: u64 = P31BabyBearPrimeField::div(&num1, &num2);
        assert_eq!(num3, 1482026200);
    }
}