use std::io;
use thiserror::Error;

use lambdaworks_math::errors::DeserializationError;

#[derive(Debug, Error)]
pub enum SrsFromFileError {
    #[error("File I/O error: {0}")]
    FileError(#[from] io::Error),
    #[error("Deserialization error: {0}")]
    DeserializationError(#[from] DeserializationError),
}
