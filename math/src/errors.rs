use thiserror::Error;

#[derive(Error, Debug)]
pub enum ByteConversionError {
    #[error("from_be_bytes failed: {0}")]
    FromBEBytesError(String),
    #[error("from_le_bytes failed: {0}")]
    FromLEBytesError(String),
    #[error("to_be_bytes failed: {0}")]
    ToBEBytesError(String),
    #[error("to_le_bytes failed: {0}")]
    ToLEBytesError(String),
    #[error("try_from_slice failed in from method: {0}")]
    TryFromSliceError(#[from] std::array::TryFromSliceError),
    #[error("{0}")]
    Custom(String)
}
