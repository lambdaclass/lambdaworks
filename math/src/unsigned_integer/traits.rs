use std::{
    fmt::Display,
    ops::{Add, BitAnd, Shr},
};

use crate::{
    elliptic_curve::short_weierstrass::errors::DeserializationError,
    traits::{Deserializable, Serializable},
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
    + Serializable
    + Deserializable
{
}

impl IsUnsignedInteger for u128 {}
impl IsUnsignedInteger for u64 {}
impl IsUnsignedInteger for u32 {}
impl IsUnsignedInteger for u16 {}
impl IsUnsignedInteger for usize {}

impl Serializable for u128 {
    fn serialize(&self) -> Vec<u8> {
        self.to_be_bytes().to_vec()
    }
}

impl Deserializable for u128 {
    fn deserialize(bytes: &[u8]) -> std::result::Result<Self, DeserializationError> {
        if bytes.len() != 16 {
            return Err(DeserializationError::InvalidAmountOfBytes);
        }
        let mut bytes_buffer = [0; 16];
        bytes_buffer.copy_from_slice(&bytes[0..16]);
        Ok(Self::from_be_bytes(bytes_buffer))
    }
}

impl Serializable for u64 {
    fn serialize(&self) -> Vec<u8> {
        self.to_be_bytes().to_vec()
    }
}

impl Deserializable for u64 {
    fn deserialize(bytes: &[u8]) -> std::result::Result<Self, DeserializationError> {
        if bytes.len() != 8 {
            return Err(DeserializationError::InvalidAmountOfBytes);
        }
        let mut bytes_buffer = [0; 8];
        bytes_buffer.copy_from_slice(&bytes[0..8]);
        Ok(Self::from_be_bytes(bytes_buffer))
    }
}

impl Serializable for u32 {
    fn serialize(&self) -> Vec<u8> {
        self.to_be_bytes().to_vec()
    }
}

impl Deserializable for u32 {
    fn deserialize(bytes: &[u8]) -> std::result::Result<Self, DeserializationError> {
        if bytes.len() != 4 {
            return Err(DeserializationError::InvalidAmountOfBytes);
        }
        let mut bytes_buffer = [0; 4];
        bytes_buffer.copy_from_slice(&bytes[0..4]);
        Ok(Self::from_be_bytes(bytes_buffer))
    }
}

impl Serializable for u16 {
    fn serialize(&self) -> Vec<u8> {
        self.to_be_bytes().to_vec()
    }
}

impl Deserializable for u16 {
    fn deserialize(bytes: &[u8]) -> std::result::Result<Self, DeserializationError> {
        if bytes.len() != 2 {
            return Err(DeserializationError::InvalidAmountOfBytes);
        }
        let mut bytes_buffer = [0; 2];
        bytes_buffer.copy_from_slice(&bytes[0..2]);
        Ok(Self::from_be_bytes(bytes_buffer))
    }
}

impl Serializable for usize {
    fn serialize(&self) -> Vec<u8> {
        self.to_be_bytes().to_vec()
    }
}

impl Deserializable for usize {
    fn deserialize(bytes: &[u8]) -> std::result::Result<Self, DeserializationError> {
        if bytes.len() != 8 {
            return Err(DeserializationError::InvalidAmountOfBytes);
        }
        let mut bytes_buffer = [0; 8];
        bytes_buffer.copy_from_slice(&bytes[0..8]);
        Ok(Self::from_be_bytes(bytes_buffer))
    }
}
