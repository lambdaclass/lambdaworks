/// A trait for converting an element to its byte representation.
pub trait ToBytes {
    /// Returns the byte representation of the element in big-endian order.
    fn to_bytes_be(&self) -> Vec<u8>;

    /// Returns the byte representation of the element in little-endian order.
    fn to_bytes_le(&self) -> Vec<u8>;
}
