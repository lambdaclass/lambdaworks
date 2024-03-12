use crate::{
    errors::CreationError,
    field::{
        element::FieldElement,
        errors::FieldError,
        traits::{IsField, IsPrimeField},
    },
};
use core::fmt::{self, Display};

/// Represents a 31 bit integer value
/// Invariants:
///      31st bit is clear
///      n < MODULUS
#[derive(Debug, Clone, Copy, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct Mersenne31Field;

impl Mersenne31Field {
    fn weak_reduce(n: u32) -> u32 {
        // To reduce 'n' to 31 bits we clear its MSB, then add it back in its reduced form.
        let msb = n & (1 << 31);
        let msb_reduced = msb >> 31;
        let res = msb ^ n;

        // assert msb_reduced fits within 31 bits
        debug_assert!((res >> 31) == 0 && (msb_reduced >> 1) == 0);
        res + msb_reduced
    }

    fn as_representative(n: &u32) -> u32 {
        if *n == MERSENNE_31_PRIME_FIELD_ORDER {
            0
        } else {
            *n
        }
    }

    #[inline]
    pub fn sum<I: Iterator<Item = <Self as IsField>::BaseType>>(
        iter: I,
    ) -> <Self as IsField>::BaseType {
        // Delayed reduction
        Self::from_u64(iter.map(|x| (x as u64)).sum::<u64>())
    }
}

pub const MERSENNE_31_PRIME_FIELD_ORDER: u32 = (1 << 31) - 1;

//NOTE: This implementation was inspired by and borrows from the work done by the Plonky3 team
// https://github.com/Plonky3/Plonky3/blob/main/mersenne-31/src/lib.rs
// Thank you for pushing this technology forward.
impl IsField for Mersenne31Field {
    type BaseType = u32;

    /// Returns the sum of `a` and `b`.
    fn add(a: &u32, b: &u32) -> u32 {
        // Avoids conditional https://github.com/Plonky3/Plonky3/blob/6049a30c3b1f5351c3eb0f7c994dc97e8f68d10d/mersenne-31/src/lib.rs#L249
        // Working with i32 means we get a flag which informs us if overflow happens
        let (sum_i32, over) = (*a as i32).overflowing_add(*b as i32);
        let sum_u32 = sum_i32 as u32;
        let sum_corr = sum_u32.wrapping_sub(MERSENNE_31_PRIME_FIELD_ORDER);

        //assert 31 bit clear
        // If self + rhs did not overflow, return it.
        // If self + rhs overflowed, sum_corr = self + rhs - (2**31 - 1).
        let sum = if over { sum_corr } else { sum_u32 };
        debug_assert!((sum >> 31) == 0);
        Self::as_representative(&sum)
    }

    /// Returns the multiplication of `a` and `b`.
    // Note: for powers of 2 we can perform bit shifting this would involve overriding the trait implementation
    fn mul(a: &u32, b: &u32) -> u32 {
        Self::from_u64(u64::from(*a) * u64::from(*b))
    }

    fn sub(a: &u32, b: &u32) -> u32 {
        let (mut sub, over) = a.overflowing_sub(*b);

        // If we didn't overflow we have the correct value.
        // Otherwise we have added 2**32 = 2**31 + 1 mod 2**31 - 1.
        // Hence we need to remove the most significant bit and subtract 1.
        sub -= over as u32;
        sub & MERSENNE_31_PRIME_FIELD_ORDER
    }

    /// Returns the additive inverse of `a`.
    fn neg(a: &u32) -> u32 {
        // NOTE: MODULUS known to have 31 bit clear
        MERSENNE_31_PRIME_FIELD_ORDER - a
    }

    /// Returns the multiplicative inverse of `a`.
    fn inv(a: &u32) -> Result<u32, FieldError> {
        if *a == Self::zero() || *a == MERSENNE_31_PRIME_FIELD_ORDER {
            return Err(FieldError::InvZeroError);
        }
        let p101 = Self::mul(&Self::pow(a, 4u32), a);
        let p1111 = Self::mul(&Self::square(&p101), &p101);
        let p11111111 = Self::mul(&Self::pow(&p1111, 16u32), &p1111);
        let p111111110000 = Self::pow(&p11111111, 16u32);
        let p111111111111 = Self::mul(&p111111110000, &p1111);
        let p1111111111111111 = Self::mul(&Self::pow(&p111111110000, 16u32), &p11111111);
        let p1111111111111111111111111111 =
            Self::mul(&Self::pow(&p1111111111111111, 4096u32), &p111111111111);
        let p1111111111111111111111111111101 =
            Self::mul(&Self::pow(&p1111111111111111111111111111, 8u32), &p101);
        Ok(p1111111111111111111111111111101)
    }

    /// Returns the division of `a` and `b`.
    fn div(a: &u32, b: &u32) -> u32 {
        let b_inv = Self::inv(b).expect("InvZeroError");
        Self::mul(a, &b_inv)
    }

    /// Returns a boolean indicating whether `a` and `b` are equal or not.
    fn eq(a: &u32, b: &u32) -> bool {
        Self::as_representative(a) == Self::representative(b)
    }

    /// Returns the additive neutral element.
    fn zero() -> Self::BaseType {
        0u32
    }

    /// Returns the multiplicative neutral element.
    fn one() -> u32 {
        1u32
    }

    /// Returns the element `x * 1` where 1 is the multiplicative neutral element.
    fn from_u64(x: u64) -> u32 {
        let (lo, hi) = (x as u32 as u64, x >> 32);
        // 2^32 = 2 (mod Mersenne 31 bit prime)
        // t <= (2^32 - 1) + 2 * (2^32 - 1) = 3 * 2^32 - 3 = 6 * 2^31 - 3
        let t = lo + 2 * hi;

        const MASK: u64 = (1 << 31) - 1;
        let (lo, hi) = ((t & MASK) as u32, (t >> 31) as u32);
        // 2^31 = 1 mod Mersenne31
        // lo < 2^31, hi < 6, so lo + hi < 2^32.
        Self::weak_reduce(lo + hi)
    }

    /// Takes as input an element of BaseType and returns the internal representation
    /// of that element in the field.
    fn from_base_type(x: u32) -> u32 {
        Self::weak_reduce(x)
    }
}

impl IsPrimeField for Mersenne31Field {
    type RepresentativeType = u32;

    // Since our invariant guarantees that `value` fits in 31 bits, there is only one possible value
    // `value` that is not canonical, namely 2^31 - 1 = p = 0.
    fn representative(x: &u32) -> u32 {
        debug_assert!((x >> 31) == 0);
        Self::as_representative(x)
    }

    fn field_bit_size() -> usize {
        ((MERSENNE_31_PRIME_FIELD_ORDER - 1).ilog2() + 1) as usize
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
        u32::from_str_radix(hex_string, 16).map_err(|_| CreationError::InvalidHexString)
    }

    #[cfg(feature = "std")]
    fn to_hex(x: &u32) -> String {
        format!("{:X}", x)
    }
}

impl FieldElement<Mersenne31Field> {
    #[cfg(feature = "alloc")]
    pub fn to_bytes_le(&self) -> alloc::vec::Vec<u8> {
        self.representative().to_le_bytes().to_vec()
    }

    #[cfg(feature = "alloc")]
    pub fn to_bytes_be(&self) -> alloc::vec::Vec<u8> {
        self.representative().to_be_bytes().to_vec()
    }
}

impl Display for FieldElement<Mersenne31Field> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:x}", self.representative())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    type F = Mersenne31Field;

    #[test]
    fn from_hex_for_b_is_11() {
        assert_eq!(F::from_hex("B").unwrap(), 11);
    }

    #[test]
    fn sum_delayed_reduction() {
        let up_to = u32::pow(2, 23);
        let pow = u64::pow(2, 60);

        let iter = (0..up_to).map(F::weak_reduce).map(|e| F::pow(&e, pow));

        assert_eq!(F::from_u64(1314320703), F::sum(iter));
    }

    #[test]
    fn from_hex_for_0x1_a_is_26() {
        assert_eq!(F::from_hex("0x1a").unwrap(), 26);
    }

    #[test]
    fn bit_size_of_field_is_31() {
        assert_eq!(
            <F as crate::field::traits::IsPrimeField>::field_bit_size(),
            31
        );
    }

    #[test]
    fn one_plus_1_is_2() {
        let a = F::one();
        let b = F::one();
        let c = F::add(&a, &b);
        assert_eq!(c, 2u32);
    }

    #[test]
    fn neg_1_plus_1_is_0() {
        let a = F::neg(&F::one());
        let b = F::one();
        let c = F::add(&a, &b);
        assert_eq!(c, F::zero());
    }

    #[test]
    fn neg_1_plus_2_is_1() {
        let a = F::neg(&F::one());
        let b = F::from_base_type(2u32);
        let c = F::add(&a, &b);
        assert_eq!(c, F::one());
    }

    #[test]
    fn max_order_plus_1_is_0() {
        let a = F::from_base_type(MERSENNE_31_PRIME_FIELD_ORDER - 1);
        let b = F::one();
        let c = F::add(&a, &b);
        assert_eq!(c, F::zero());
    }

    #[test]
    fn comparing_13_and_13_are_equal() {
        let a = F::from_base_type(13);
        let b = F::from_base_type(13);
        assert_eq!(a, b);
    }

    #[test]
    fn comparing_13_and_8_they_are_not_equal() {
        let a = F::from_base_type(13);
        let b = F::from_base_type(8);
        assert_ne!(a, b);
    }

    #[test]
    fn one_sub_1_is_0() {
        let a = F::one();
        let b = F::one();
        let c = F::sub(&a, &b);
        assert_eq!(c, F::zero());
    }

    #[test]
    fn zero_sub_1_is_order_minus_1() {
        let a = F::zero();
        let b = F::one();
        let c = F::sub(&a, &b);
        assert_eq!(c, MERSENNE_31_PRIME_FIELD_ORDER - 1);
    }

    #[test]
    fn neg_1_sub_neg_1_is_0() {
        let a = F::neg(&F::one());
        let b = F::neg(&F::one());
        let c = F::sub(&a, &b);
        assert_eq!(c, F::zero());
    }

    #[test]
    fn neg_1_sub_1_is_neg_1() {
        let a = F::neg(&F::one());
        let b = F::zero();
        let c = F::sub(&a, &b);
        assert_eq!(c, F::neg(&F::one()));
    }

    #[test]
    fn mul_neutral_element() {
        let a = F::from_base_type(1);
        let b = F::from_base_type(2);
        let c = F::mul(&a, &b);
        assert_eq!(c, F::from_base_type(2));
    }

    #[test]
    fn mul_2_3_is_6() {
        let a = F::from_base_type(2);
        let b = F::from_base_type(3);
        assert_eq!(a * b, F::from_base_type(6));
    }

    #[test]
    fn mul_order_neg_1() {
        let a = F::from_base_type(MERSENNE_31_PRIME_FIELD_ORDER - 1);
        let b = F::from_base_type(MERSENNE_31_PRIME_FIELD_ORDER - 1);
        let c = F::mul(&a, &b);
        assert_eq!(c, F::from_base_type(1));
    }

    #[test]
    fn pow_p_neg_1() {
        assert_eq!(
            F::pow(&F::from_base_type(2), MERSENNE_31_PRIME_FIELD_ORDER - 1),
            F::one()
        )
    }

    #[test]
    fn inv_0_error() {
        let result = F::inv(&F::zero());
        assert!(matches!(result, Err(FieldError::InvZeroError)));
    }

    #[test]
    fn inv_2() {
        let result = F::inv(&F::from_base_type(2u32)).unwrap();
        // sage: 1 / F(2) = 1073741824
        assert_eq!(result, 1073741824);
    }

    #[test]
    fn pow_2_3() {
        assert_eq!(F::pow(&F::from_base_type(2), 3_u64), 8)
    }

    #[test]
    fn div_1() {
        assert_eq!(F::div(&F::from_base_type(2), &F::from_base_type(1)), 2)
    }

    #[test]
    fn div_4_2() {
        assert_eq!(F::div(&F::from_base_type(4), &F::from_base_type(2)), 2)
    }

    // 1431655766
    #[test]
    fn div_4_3() {
        // sage: F(4) / F(3) = 1431655766
        assert_eq!(
            F::div(&F::from_base_type(4), &F::from_base_type(3)),
            1431655766
        )
    }

    #[test]
    fn two_plus_its_additive_inv_is_0() {
        let two = F::from_base_type(2);

        assert_eq!(F::add(&two, &F::neg(&two)), F::zero())
    }

    #[test]
    fn from_u64_test() {
        let num = F::from_u64(1u64);
        assert_eq!(num, F::one());
    }

    #[test]
    fn creating_a_field_element_from_its_representative_returns_the_same_element_1() {
        let change = 1;
        let f1 = F::from_base_type(MERSENNE_31_PRIME_FIELD_ORDER + change);
        let f2 = F::from_base_type(Mersenne31Field::representative(&f1));
        assert_eq!(f1, f2);
    }

    #[test]
    fn creating_a_field_element_from_its_representative_returns_the_same_element_2() {
        let change = 8;
        let f1 = F::from_base_type(MERSENNE_31_PRIME_FIELD_ORDER + change);
        let f2 = F::from_base_type(Mersenne31Field::representative(&f1));
        assert_eq!(f1, f2);
    }

    #[test]
    fn from_base_type_test() {
        let b = F::from_base_type(1u32);
        assert_eq!(b, F::one());
    }

    #[cfg(feature = "std")]
    #[test]
    fn to_hex_test() {
        let num = F::from_hex("B").unwrap();
        assert_eq!(F::to_hex(&num), "B");
    }
}
