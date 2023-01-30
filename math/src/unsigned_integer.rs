use crypto_bigint::U384;
use std::ops::{BitAnd, Shr};

pub trait IsUnsignedInteger:
    Shr<usize, Output = Self> + BitAnd<Output = Self> + Eq + Ord + From<u16> + Copy
{
}

impl IsUnsignedInteger for U384 {}
impl IsUnsignedInteger for u128 {}
impl IsUnsignedInteger for u64 {}
impl IsUnsignedInteger for u32 {}
impl IsUnsignedInteger for u16 {}
impl IsUnsignedInteger for usize {}
