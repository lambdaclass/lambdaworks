use crate::errors::ByteConversionError;

/// A trait for converting an element to and from its byte representation and
/// for getting an element from its byte representation in big-endian or
/// little-endian order.
pub trait ByteConversion {
    /// Returns the byte representation of the element in big-endian order.
    fn to_bytes_be(&self) -> Result<Vec<u8>, ByteConversionError>;

    /// Returns the byte representation of the element in little-endian order.
    fn to_bytes_le(&self) -> Result<Vec<u8>, ByteConversionError>;

    /// Returns the element from its byte representation in big-endian order.
    fn from_bytes_be(bytes: &[u8]) -> Result<Self, ByteConversionError> where Self: std::marker::Sized;

    /// Returns the element from its byte representation in little-endian order.
    fn from_bytes_le(bytes: &[u8]) -> Result<Self, ByteConversionError> where Self: std::marker::Sized;
}
