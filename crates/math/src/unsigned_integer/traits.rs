use core::{
    fmt::{Display, LowerHex, UpperHex},
    ops::{Add, BitAnd, Shr, ShrAssign},
};

pub trait IsUnsignedInteger:
    Shr<usize, Output = Self>
    + ShrAssign<usize>
    + BitAnd<Output = Self>
    + Eq
    + Ord
    + From<u16>
    + Copy
    + Display
    + LowerHex
    + UpperHex
    + Add<Self, Output = Self>
{
}

impl IsUnsignedInteger for u128 {}
impl IsUnsignedInteger for u64 {}
impl IsUnsignedInteger for u32 {}
impl IsUnsignedInteger for u16 {}
impl IsUnsignedInteger for usize {}
