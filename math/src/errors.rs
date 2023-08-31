#[derive(Debug, PartialEq, Eq)]
pub enum ByteConversionError {
    FromBEBytesError,
    FromLEBytesError,
    #[error("Invalid value")]
    InvalidValue,
    #[error("The point is not in the subgroup")]
    PointNotInSubgroup,
    #[error("Value is not compressed")]
    ValueNotCompressed,
}

#[derive(Debug, PartialEq, Eq)]
pub enum CreationError {
    InvalidHexString,
    InvalidDecString,
    EmptyString,
}

#[derive(Debug, PartialEq, Eq)]
pub enum DeserializationError {
    InvalidAmountOfBytes,
    FieldFromBytesError,
    PointerSizeError,
}

impl From<ByteConversionError> for DeserializationError {
    fn from(error: ByteConversionError) -> Self {
        match error {
            ByteConversionError::FromBEBytesError => DeserializationError::FieldFromBytesError,
            ByteConversionError::FromLEBytesError => DeserializationError::FieldFromBytesError,
        }
    }
}
