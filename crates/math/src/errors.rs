use core::fmt;

#[derive(Debug, PartialEq, Eq)]
pub enum ByteConversionError {
    FromBEBytesError,
    FromLEBytesError,
    InvalidValue,
    PointNotInSubgroup,
    ValueNotCompressed,
    ValueNotReduced,
}

impl fmt::Display for ByteConversionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ByteConversionError::FromBEBytesError => {
                write!(f, "Failed to convert from big-endian bytes")
            }
            ByteConversionError::FromLEBytesError => {
                write!(f, "Failed to convert from little-endian bytes")
            }
            ByteConversionError::InvalidValue => {
                write!(f, "Invalid value encountered during byte conversion")
            }
            ByteConversionError::PointNotInSubgroup => {
                write!(f, "Point is not in the expected subgroup")
            }
            ByteConversionError::ValueNotCompressed => {
                write!(f, "Value is not in compressed form")
            }
            ByteConversionError::ValueNotReduced => {
                write!(f, "Value is not reduced modulo the field prime")
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ByteConversionError {}

#[derive(Debug, PartialEq, Eq)]
pub enum CreationError {
    InvalidHexString,
    InvalidDecString,
    HexStringIsTooBig,
    CanonicalValueOutOfRange,
    EmptyString,
}

impl fmt::Display for CreationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CreationError::InvalidHexString => write!(f, "Invalid hexadecimal string"),
            CreationError::InvalidDecString => write!(f, "Invalid decimal string"),
            CreationError::HexStringIsTooBig => {
                write!(f, "Hexadecimal string exceeds maximum allowed size")
            }
            CreationError::CanonicalValueOutOfRange => {
                write!(f, "Canonical value is out of the valid range for the field")
            }
            CreationError::EmptyString => write!(f, "Cannot create element from empty string"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for CreationError {}

#[derive(Debug, PartialEq, Eq)]
pub enum DeserializationError {
    InvalidAmountOfBytes,
    FieldFromBytesError,
    PointerSizeError,
    InvalidValue,
}

impl fmt::Display for DeserializationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeserializationError::InvalidAmountOfBytes => {
                write!(f, "Invalid number of bytes for deserialization")
            }
            DeserializationError::FieldFromBytesError => {
                write!(f, "Failed to deserialize field element from bytes")
            }
            DeserializationError::PointerSizeError => {
                write!(f, "Pointer size mismatch during deserialization")
            }
            DeserializationError::InvalidValue => {
                write!(f, "Invalid value encountered during deserialization")
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for DeserializationError {}

#[derive(Debug, PartialEq, Eq)]
pub enum PairingError {
    PointNotInSubgroup,
    DivisionByZero,
}

impl fmt::Display for PairingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PairingError::PointNotInSubgroup => {
                write!(f, "Point is not in the expected subgroup for pairing")
            }
            PairingError::DivisionByZero => {
                write!(f, "Division by zero during pairing computation")
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for PairingError {}

impl From<ByteConversionError> for DeserializationError {
    fn from(error: ByteConversionError) -> Self {
        match error {
            ByteConversionError::FromBEBytesError => DeserializationError::FieldFromBytesError,
            ByteConversionError::FromLEBytesError => DeserializationError::FieldFromBytesError,
            _ => DeserializationError::InvalidValue,
        }
    }
}
