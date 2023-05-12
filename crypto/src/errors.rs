use lambdaworks_math::elliptic_curve::short_weierstrass::errors::DeserializationError;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum SrsFromFileError {
    #[error("IO Error")]
    FileError(#[from] std::io::Error),
    #[error("Error when deserializing")]
    DeserializationError(#[from] DeserializationError),
}
