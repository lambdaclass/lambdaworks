use crate::metal::abstractions::errors::MetalError;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum FFTError {
    #[error("The order of the polynomial is not correct")]
    InvalidOrder(String),
    #[error("Could not calculate {1} root of unity")]
    RootOfUnityError(String, u64),

    #[error("Couldn't find a system default device for Metal")]
    MetalDeviceNotFound(),
    #[error("Couldn't create a new Metal library: {0}")]
    MetalLibraryError(String),
    #[error("Couldn't create a new Metal function object: {0}")]
    MetalFunctionError(String),
    #[error("Couldn't create a new Metal compute pipeline: {0}")]
    MetalPipelineError(String),
}

#[derive(Debug, Error)]
pub enum FFTMetalError {
    #[error("A FFT related error has ocurred")]
    FFT(FFTError),
    #[error("A Metal related error has ocurred")]
    Metal(MetalError),
}

impl From<FFTError> for FFTMetalError {
    fn from(error: FFTError) -> Self {
        FFTMetalError::FFT(error)
    }
}

impl From<MetalError> for FFTMetalError {
    fn from(error: MetalError) -> Self {
        FFTMetalError::Metal(error)
    }
}
