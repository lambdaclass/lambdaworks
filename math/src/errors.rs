#[derive(Debug, PartialEq, Eq)]
pub enum ByteConversionError {
    FromBEBytesError,
    FromLEBytesError,
}

#[derive(Debug, PartialEq, Eq)]
pub enum CreationError {
    InvalidHexString,
    InvalidDecString
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
