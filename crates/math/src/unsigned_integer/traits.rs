use core::{
    fmt::Display,
    ops::{Add, BitAnd, Shr, ShrAssign},
};
use num_bigint::BigUint;

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

/// Trait for arbitrary-precision unsigned integer types
pub trait IsBigUnsignedInteger:
    Shr<usize, Output = Self>
    + ShrAssign<usize>
    + BitAnd<Output = Self>
    + Eq
    + Ord
    + From<u16>
    + Clone
    + Display
    + Add<Self, Output = Self>
{
    /// Convert from a hex string
    fn from_hex_str(hex: &str) -> Option<Self>;
    /// Convert to bytes in big-endian order
    fn to_bytes_be(&self) -> Vec<u8>;
    /// Convert from bytes in big-endian order
    fn from_bytes_be(bytes: &[u8]) -> Self;
    /// Get the number of bits needed to represent this number
    fn bits(&self) -> u32;
}

impl IsUnsignedInteger for u128 {}
impl IsUnsignedInteger for u64 {}
impl IsUnsignedInteger for u32 {}
impl IsUnsignedInteger for u16 {}
impl IsUnsignedInteger for usize {}

impl IsBigUnsignedInteger for BigUint {
    fn from_hex_str(hex: &str) -> Option<Self> {
        let hex_str = hex.strip_prefix("0x").unwrap_or(hex);
        use num_traits::Num;
        Self::from_str_radix(hex_str, 16).ok()
    }

    fn to_bytes_be(&self) -> Vec<u8> {
        self.to_bytes_be()
    }

    fn from_bytes_be(bytes: &[u8]) -> Self {
        Self::from_bytes_be(bytes)
    }

    fn bits(&self) -> u32 {
        self.bits() as u32
    }
}
