use core::{
    fmt::Display,
    ops::{Add, BitAnd, Shr, ShrAssign},
};

/// Trait for fixed-size unsigned integer types that can be used in field operations
pub trait IsUnsignedInteger:
    Shr<usize, Output = Self>
    + ShrAssign<usize>
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
