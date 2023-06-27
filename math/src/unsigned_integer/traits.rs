use core::{
    fmt::Display,
    ops::{Add, BitAnd, Shr},
};

pub trait IsUnsignedInteger:
    Shr<usize, Output = Self>
    + BitAnd<Output = Self>
    + Eq
    + Ord
    + From<u16>
    + Copy
    + Display
    + Add<Self, Output = Self>
{
}

impl IsUnsignedInteger for u128 {}
impl IsUnsignedInteger for u64 {}
impl IsUnsignedInteger for u32 {}
impl IsUnsignedInteger for u16 {}
impl IsUnsignedInteger for usize {}
