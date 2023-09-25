use crate::{
    errors::CreationError,
    field::{
        errors::FieldError,
        traits::{IsField, IsPrimeField},
    },
};

/// Goldilocks Prime Field F_p where p = 2^64 - 2^32 + 1;
#[derive(Debug, Clone, Copy, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct Goldilocks64Field;

impl Goldilocks64Field {
    const ORDER: u64 = 0xFFFF_FFFF_0000_0001;
    const NEG_ORDER: u64 = Self::ORDER.wrapping_neg();
}

//NOTE: This implementation was inspired by and borrows from the work done by the Plonky3 team
//https://github.com/Plonky3/Plonky3/blob/main/goldilocks/src/lib.rs
// Thank you for pushing this technology forward.
impl IsField for Goldilocks64Field {
    type BaseType = u64;

    fn add(a: &u64, b: &u64) -> u64 {
        let (sum, over) = a.overflowing_add(*b);
        let (mut sum, over) = sum.overflowing_add(u64::from(over) * Self::NEG_ORDER);
        if over {
            //TODO: add assume and branch hint()
            sum += Self::NEG_ORDER
        }
        sum
    }

    fn mul(a: &u64, b: &u64) -> u64 {
        reduce_128(u128::from(*a) * u128::from(*b))
    }

    fn sub(a: &u64, b: &u64) -> u64 {
        let (diff, under) = a.overflowing_sub(*b);
        let (mut diff, under) = diff.overflowing_sub(u64::from(under) * Self::NEG_ORDER);
        if under {
            diff -= Self::NEG_ORDER;
        }
        diff
    }

    fn neg(a: &u64) -> u64 {
        //NOTE: This should be conducted as a canonical u64;
        Self::sub(&Self::ORDER,a)
    }

    /// Returns the multiplicative inverse of `a`.
    fn inv(a: &u64) -> Result<u64, FieldError> {
        todo!()
    }

    /// Returns the division of `a` and `b`.
    fn div(a: &u64, b: &u64) -> u64 {
        let b_inv = Self::inv(b).unwrap();
        Self::mul(a, &b_inv)
    }

    /// Returns a boolean indicating whether `a` and `b` are equal or not.
    fn eq(a: &u64, b: &u64) -> bool {
        //TODO: Check if this is a canonical check
        a == b
    }

    /// Returns the additive neutral element.
    fn zero() -> u64 {
        0u64
    }

    /// Returns the multiplicative neutral element.
    fn one() -> u64 {
        1u64
    }

    /// Returns the element `x * 1` where 1 is the multiplicative neutral element.
    fn from_u64(x: u64) -> u64 {
        todo!()
    }

    /// Takes as input an element of BaseType and returns the internal representation
    /// of that element in the field.
    fn from_base_type(x: u64) -> u64 {
        todo!()
    }
}

impl IsPrimeField for Goldilocks64Field {
    type RepresentativeType = u64;

    // Since our invariant guarantees that `value` fits in 31 bits, there is only one possible
    // `value` that is not canonical, namely 2^31 - 1 = p = 0.
    fn representative(x: &u64) -> u64 {
        todo!()
    }

    fn field_bit_size() -> usize {
        ((self::Goldilocks64Field::ORDER - 1).ilog2() + 1) as usize
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

#[inline]
fn reduce_128(x: u128) -> u64 {
    let (x_lo, x_hi) = (x as u64, (x >> 64) as u64);
    let x_hi_hi = x_hi >> 32;
    let x_hi_lo = x_hi & Goldilocks64Field::NEG_ORDER;

    let (mut t0, borrow) = x_lo.overflowing_sub(x_hi_hi);
    if borrow {
        //TODO: add branch hinting
        t0 -= Goldilocks64Field::NEG_ORDER // Cannot underflow
    }

    let t1 = x_hi_lo * Goldilocks64Field::NEG_ORDER;
    //NOTE: add optimized unsafe
    let (res_wrapped, carry) = t0.overflowing_add(t1);
    // Below cannot overflow unless the assumption if x + y < 2**64 + ORDER is incorrect.
    let t2 = res_wrapped + Goldilocks64Field::NEG_ORDER * u64::from(carry);

    t2
}
