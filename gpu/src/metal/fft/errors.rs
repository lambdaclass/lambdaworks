use crate::metal::abstractions::errors::MetalError;
use lambdaworks_math::fft::errors::FFTError;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum FFTMetalError {
    #[error("A FFT related error has ocurred")]
    FFT(FFTError),
    #[error("A Metal related error has ocurred")]
    Metal(MetalError),
}
