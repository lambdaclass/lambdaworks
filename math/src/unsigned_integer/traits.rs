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
pub trait U32Limbs<const NUM_LIMBS: usize>
where
    Self: Sized,
{
    /// Create a `Self` from its representation in u32 limbs.
    fn from_u32_limbs(limbs: &[u32]) -> Self;
    /// Returns a collection of u32 limbs which represents `self`.
    fn to_u32_limbs(&self) -> Vec<u32>;

    /// Returns a collection of elements made from `limbs`, using `Self`'s
    /// `from_u32_limbs()`. If `NUM_LIMBS` doesn't divide `limbs.len()`, the last element will be
    /// made with less than `NUM_LIMBS` limbs, in that case check the previous method implementation.
    fn from_flat_u32_limbs(limbs: &[u32]) -> Vec<Self> {
        let items = limbs.chunks(NUM_LIMBS).map(Self::from_u32_limbs);
        items.collect()
    }

    /// Returns a flat collection of u32 limbs which represent all of this iterator's elements.
    fn to_flat_u32_limbs(elems: &[Self]) -> Vec<u32> {
        elems.iter().flat_map(|e| e.to_u32_limbs()).collect()
    }
}
