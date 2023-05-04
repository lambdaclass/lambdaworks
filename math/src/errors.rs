use thiserror::Error;

#[derive(Error, Debug, PartialEq, Eq)]
pub enum ByteConversionError {
    #[error("from_be_bytes failed")]
    FromBEBytesError,
    #[error("from_le_bytes failed")]
    FromLEBytesError,
    #[error("Invalid value")]
    InvalidValue,
    #[error("The point is not in the subgroup")]
    PointNotInSubgroup,
    #[error("Value is not compressed")]
    ValueNotCompressed,
}
