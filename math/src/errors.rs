use thiserror::Error;

#[derive(Error, Debug)]
pub enum ByteConversionError {
    #[error("from_be_bytes failed")]
    FromBEBytesError,
    #[error("from_le_bytes failed")]
    FromLEBytesError,
}
