use crate::{
    errors::DeserializationError,
    field::{element::FieldElement, traits::IsField},
};

use crate::errors::ByteConversionError;
/// A trait for converting an element to and from its byte representation and
/// for getting an element from its byte representation in big-endian or
/// little-endian order.
pub trait ByteConversion {
    /// Returns the byte representation of the element in big-endian order.}
    #[cfg(feature = "alloc")]
    fn to_bytes_be(&self) -> alloc::vec::Vec<u8>;

    /// Returns the byte representation of the element in little-endian order.
    #[cfg(feature = "alloc")]
    fn to_bytes_le(&self) -> alloc::vec::Vec<u8>;

    /// Returns the element from its byte representation in big-endian order.
    fn from_bytes_be(bytes: &[u8]) -> Result<Self, ByteConversionError>
    where
        Self: Sized;

    /// Returns the element from its byte representation in little-endian order.
    fn from_bytes_le(bytes: &[u8]) -> Result<Self, ByteConversionError>
    where
        Self: Sized;
}

/// Serialize function without args
/// Used for serialization when formatting options are not relevant
#[cfg(feature = "alloc")]
pub trait AsBytes {
    /// Default serialize without args
    fn as_bytes(&self) -> alloc::vec::Vec<u8>;
}

#[cfg(feature = "alloc")]
impl AsBytes for u32 {
    fn as_bytes(&self) -> alloc::vec::Vec<u8> {
        self.to_le_bytes().to_vec()
    }
}

#[cfg(feature = "alloc")]
impl AsBytes for u64 {
    fn as_bytes(&self) -> alloc::vec::Vec<u8> {
        self.to_le_bytes().to_vec()
    }
}

/// Deserialize function without args
pub trait Deserializable {
    fn deserialize(bytes: &[u8]) -> Result<Self, DeserializationError>
    where
        Self: Sized;
}

pub trait IsRandomFieldElementGenerator<F: IsField> {
    fn generate(&self) -> FieldElement<F>;
}

/// Serialize a value with a u32 big-endian length prefix.
/// This is a common pattern used in proof serialization.
#[cfg(feature = "alloc")]
pub fn serialize_with_length<T: AsBytes>(item: &T) -> alloc::vec::Vec<u8> {
    let bytes = item.as_bytes();
    let mut result = alloc::vec::Vec::with_capacity(4 + bytes.len());
    result.extend_from_slice(&(bytes.len() as u32).to_be_bytes());
    result.extend_from_slice(&bytes);
    result
}

/// Deserialize a value with a u32 big-endian length prefix.
/// Returns the new offset and the deserialized value.
/// This is a common pattern used in proof deserialization.
pub fn deserialize_with_length<T: Deserializable>(
    bytes: &[u8],
    offset: usize,
) -> Result<(usize, T), DeserializationError> {
    const SIZE_OF_U32: usize = core::mem::size_of::<u32>();
    let size_bytes: [u8; SIZE_OF_U32] = bytes
        .get(offset..offset + SIZE_OF_U32)
        .ok_or(DeserializationError::InvalidAmountOfBytes)?
        .try_into()
        .map_err(|_| DeserializationError::InvalidAmountOfBytes)?;
    let size = u32::from_be_bytes(size_bytes) as usize;
    let new_offset = offset + SIZE_OF_U32;
    let item = T::deserialize(
        bytes
            .get(new_offset..new_offset + size)
            .ok_or(DeserializationError::InvalidAmountOfBytes)?,
    )?;
    Ok((new_offset + size, item))
}

/// Deserialize a field element with a u32 big-endian length prefix.
/// Returns the new offset and the deserialized field element.
/// This is a common pattern used in proof deserialization for field elements
/// that implement ByteConversion but not Deserializable.
pub fn deserialize_field_element_with_length<F: IsField>(
    bytes: &[u8],
    offset: usize,
) -> Result<(usize, FieldElement<F>), DeserializationError>
where
    FieldElement<F>: ByteConversion,
{
    const SIZE_OF_U32: usize = core::mem::size_of::<u32>();
    let size_bytes: [u8; SIZE_OF_U32] = bytes
        .get(offset..offset + SIZE_OF_U32)
        .ok_or(DeserializationError::InvalidAmountOfBytes)?
        .try_into()
        .map_err(|_| DeserializationError::InvalidAmountOfBytes)?;
    let size = u32::from_be_bytes(size_bytes) as usize;
    let new_offset = offset + SIZE_OF_U32;
    let field_element = FieldElement::from_bytes_be(
        bytes
            .get(new_offset..new_offset + size)
            .ok_or(DeserializationError::InvalidAmountOfBytes)?,
    )?;
    Ok((new_offset + size, field_element))
}
