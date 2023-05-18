use std::{
    fmt::Display,
    ops::{BitAnd, Shr},
};

pub trait IsUnsignedInteger:
    Shr<usize, Output = Self> + BitAnd<Output = Self> + Eq + Ord + From<u16> + Copy + Display
{
}

impl IsUnsignedInteger for u128 {}
impl IsUnsignedInteger for u64 {}
impl IsUnsignedInteger for u32 {}
impl IsUnsignedInteger for u16 {}
impl IsUnsignedInteger for usize {}

/// Trait for converting a type (generally a collection of UnsignedIntegers) to and from unsigned 32 bit
/// limbs, if it can be represented as that. Useful for sending data to other 32 bit UnsignedInteger
/// implementations, like in Metal.
pub trait U32Limbs {
    fn from_u32_limbs(limbs: &[u32]) -> Self;
    fn to_u32_limbs(&self) -> Vec<u32>;
}
