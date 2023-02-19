use thiserror::Error;

use crate::field::errors::FieldError;

#[derive(Debug, Error)]
pub enum FFTError {
    #[error("The order of the polynomial is not correct")]
    InvalidOrder(String),
    #[error("Could not calculate {1} root of unity")]
    RootOfUnityError(String, u64),
    #[error("Field error occurred: {0}")]
    FieldError(FieldError),
}
