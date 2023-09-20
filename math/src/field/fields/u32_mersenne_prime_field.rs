use core::ops::BitXorAssign;

use crate::field::{errors::FieldError, traits::IsField};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mersenne31Field;

pub const MERSENNE_31_PRIME_FIELD_ORDER: u32 = (1 << 31) - 1;

//TODO implement Montgomery Backend

//NOTE: This implementation was inspired by and borrows from the work done by the Plonky3 team
// https://github.com/Plonky3/Plonky3/blob/main/mersenne-31/src/lib.rs
// Thank you for pushing this technology forward.
impl IsField for Mersenne31Field {
    type BaseType = u32;

    /// Returns the sum of `a` and `b`.
    fn add(a: &u32, b: &u32) -> u32 {
        let mut sum = a + b;
        // If sum's most significant bit is set, we clear it and add 1, since 2^31 = 1 mod p.
        // This addition of 1 cannot overflow 2^31, since sum has a max of
        // 2 * (2^31 - 1) = 2^32 - 2.
        let msb = sum & (1 << 31);
        sum.bitxor_assign(msb);
        sum += u32::from(msb != 0);
        Self::from_base_type(sum)
    }

    /// Returns the multiplication of `a` and `b`.
    fn mul(a: &u32, b: &u32) -> u32 {
        let prod = u64::from(*a) * u64::from(*b);
        let prof_lo = (prod as u32) & ((1 << 31) - 1);
        let prod_hi = (prod >> 31) as u32;
        Self::add(
            &Self::from_base_type(prof_lo),
            &Self::from_base_type(prod_hi),
        )
    }

    // Need to optimize
    fn sub(a: &u32, b: &u32) -> u32 {
        Self::add(a, &Self::neg(b))
    }

    /// Returns the additive inverse of `a`.
    fn neg(a: &u32) -> u32 {
        Self::sub(&MERSENNE_31_PRIME_FIELD_ORDER, a)
    }

    /// Returns the multiplicative inverse of `a`.
    fn inv(a: &u32) -> Result<u32, FieldError> {
        if *a == Self::zero() {
            return Err(FieldError::InvZeroError);
        }
        let p1 = *a;
        let p1_1 = Self::mul(&Self::pow(&p1, 2u32), &p1);
        let p4 = Self::mul(&Self::square(&p1_1), &p1_1);
        let p8 = Self::mul(&Self::pow(&p4, 8u32), &p4);
        let p16 = Self::mul(&Self::pow(&p8, 16u32), &p8);
        let p24 = Self::mul(&Self::pow(&p16, 16u32), &p8);
        let p28 = Self::mul(&Self::pow(&p24, 8u32), &p4);
        let p29_1 = Self::mul(&Self::pow(&p28, 6u32), &p1_1);
        Ok(p29_1)
    }

    /// Returns the division of `a` and `b`.
    fn div(a: &u32, b: &u32) -> u32 {
        let b_inv = Self::inv(b).expect("InvZeroError");
        Self::mul(a, &b_inv)
    }

    /// Returns a boolean indicating whether `a` and `b` are equal or not.
    fn eq(a: &u32, b: &u32) -> bool {
        a == b
    }

    /// Returns the additive neutral element.
    fn zero() -> u32 {
        0u32
    }

    /// Returns the multiplicative neutral element.
    fn one() -> u32 {
        1u32
    }

    /// Returns the element `x * 1` where 1 is the multiplicative neutral element.
    fn from_u64(x: u64) -> u32 {
        let x_u32: u32 = x.try_into().expect("Too large to be canonical Mersenne31");
        Self::from_base_type(x_u32)
    }

    /// Takes as input an element of BaseType and returns the internal representation
    /// of that element in the field.
    fn from_base_type(x: u32) -> u32 {
        //NOTE: we could abstract this check to a ::new() function of a wrapper struct for u32. I decided to go for the simplest form.
        debug_assert!(x < MERSENNE_31_PRIME_FIELD_ORDER);
        debug_assert!((x >> 31) == 0);
        x
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    type Fp = Mersenne31Field;

    #[test]
    fn one_plus_one_is_two() {
        let num1 = Fp::one();
        let num2 = Fp::one();
        let num3 = Fp::add(&num1, &num2);
        assert_eq!(num3, 2u32);
    }

    #[test]
    fn neg_one_plus_one_is_zero() {
        let num1 = Fp::neg(&Fp::one());
        let num2 = Fp::one();
        let num3 = Fp::add(&num1, &num2);
        assert_eq!(num3, Fp::zero());
    }

    #[test]
    fn neg_one_plus_two_is_one() {
        let num1 = Fp::neg(&Fp::one());
        let num2 = Fp::from_base_type(2u32);
        let num3 = Fp::add(&num1, &num2);
        assert_eq!(num3, Fp::one());
    }

    #[test]
    fn neg_one_plus_neg_one_is_() {
        let num1 = Fp::from_base_type(MERSENNE_31_PRIME_FIELD_ORDER - 1);
        let num2 = Fp::one();
        let num3 = Fp::add(&num1, &num2);
        assert_eq!(num3, Fp::zero());
    }

    #[test]
    fn sub_test_1() {
        let num1 = Fp::one();
        let num2 = Fp::one();
        let num3 = Fp::sub(&num1, &num2);
        assert_eq!(num3, Fp::zero())
    }

    #[test]
    fn sub_test_2() {
        let num1 = Fp::from_base_type(2u32);
        let num2 = Fp::from_base_type(2u32);
        let num3 = Fp::sub(&num1, &num2);
        assert_eq!(num3, Fp::zero());
    }

    #[test]
    fn sub_test_3() {
        let num1 = Fp::neg(&Fp::one());
        let num2 = Fp::neg(&Fp::one());
        let num3 = Fp::sub(&num1, &num2);
        assert_eq!(num3, Fp::zero());
    }

    #[test]
    fn sub_test_4() {
        let num1 = Fp::from_base_type(2u32);
        let num2 = Fp::one();
        let num3 = Fp::sub(&num1, &num2);
        assert_eq!(num3, Fp::one());
    }

    #[test]
    fn sub_test_5() {
        let num1 = Fp::neg(&Fp::one());
        let num2 = Fp::zero();
        let num3 = Fp::sub(&num1, &num2);
        assert_eq!(num3, Fp::neg(&Fp::one()));
    }

    /*
    #[test]
    fn neg_test() {
        let num1 = Fp::from_base_type();
        let num2 = Fp::neg();
        assert_eq!(num2, Fp::from_base_type());
    }

    #[test]
    fn mul_test_1() {
        let num1 = Fp::from_base_type();
        let num2 = Fp::from_base_type();
        let num3 = Fp::mul(&num1, &num2);
        assert_eq!(num3, Fp::from_base_type());
    }

    #[test]
    fn mul_test_2() {
        let num1 = Fp::from_base_type();
        let num2 = Fp::from_base_type();
        let num3 = Fp::mul(&num1, &num2);
        assert_eq!(num3, Fp::from_base_type());
    }

    #[test]
    fn pow_test() {
        let num1 = Fp::from_base_type();
        let num2 = Fp::pow(&num1, 65537u64);
        assert_eq!(num2, Fp::from_base_type());
    }
    */

    #[test]
    fn inv_0_test() {
        let result = Fp::inv(&Fp::zero());
        assert!(matches!(result, Err(FieldError::InvZeroError)));
    }

    #[test]
    fn inv_2_test() {
        let result = Fp::inv(&Fp::from_base_type(2u32));
        assert!(matches!(result, Err(FieldError::InvZeroError)));
    }

    #[test]
    fn from_u64_test() {
        let num = Fp::from_u64(1u64);
        assert_eq!(num, Fp::one());
    }

    #[test]
    #[should_panic]
    fn from_u64_max_test() {
        Fp::from_u64(u64::MAX);
    }

    #[test]
    fn from_base_type_test() {
        let num2 = Fp::from_base_type(1u32);
        assert_eq!(num2, Fp::one());
    }
}
