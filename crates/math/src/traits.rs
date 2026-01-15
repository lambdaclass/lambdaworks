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

/// Serialize an item with a u32 length prefix (big-endian).
/// The format is: [4 bytes length (BE)] [serialized data]
#[cfg(feature = "alloc")]
pub fn serialize_with_length<T: AsBytes>(item: &T) -> alloc::vec::Vec<u8> {
    let bytes = item.as_bytes();
    let mut result = alloc::vec::Vec::with_capacity(4 + bytes.len());
    result.extend_from_slice(&(bytes.len() as u32).to_be_bytes());
    result.extend_from_slice(&bytes);
    result
}

/// Deserialize an item with a u32 length prefix (big-endian).
/// Returns the new offset and the deserialized item.
pub fn deserialize_with_length<T: Deserializable>(
    bytes: &[u8],
    offset: usize,
) -> Result<(usize, T), DeserializationError> {
    const SIZE_OF_U32: usize = core::mem::size_of::<u32>();
    let mut offset = offset;
    let element_size_bytes: [u8; SIZE_OF_U32] = bytes
        .get(offset..offset + SIZE_OF_U32)
        .ok_or(DeserializationError::InvalidAmountOfBytes)?
        .try_into()
        .map_err(|_| DeserializationError::InvalidAmountOfBytes)?;
    let element_size = u32::from_be_bytes(element_size_bytes) as usize;
    offset += SIZE_OF_U32;
    let item = T::deserialize(
        bytes
            .get(offset..offset + element_size)
            .ok_or(DeserializationError::InvalidAmountOfBytes)?,
    )?;
    offset += element_size;
    Ok((offset, item))
}
