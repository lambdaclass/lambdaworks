use thiserror::Error;

#[derive(Error, Debug)]
pub enum ByteConversionError {
    #[error("from_be_bytes failed")]
    FromBEBytesError,
    #[error("from_le_bytes failed")]
    FromLEBytesError,
    #[error("to_be_bytes failed")]
    ToBEBytesError,
    #[error("to_le_bytes failed")]
    ToLEBytesError,
    #[error("{0}")]
    Custom(String),
}
