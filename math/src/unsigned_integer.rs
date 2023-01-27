use crypto_bigint::U384;
use std::ops::{BitAnd, Shr};

pub trait IsUnsignedInteger<T>:
    Shr<usize, Output = T> + BitAnd<Output = T> + Eq + Ord + From<u16> + Copy
{
}

impl IsUnsignedInteger<U384> for U384 {}
impl IsUnsignedInteger<u128> for u128 {}
impl IsUnsignedInteger<u64> for u64 {}
impl IsUnsignedInteger<u32> for u32 {}
impl IsUnsignedInteger<u16> for u16 {}
impl IsUnsignedInteger<usize> for usize {}
