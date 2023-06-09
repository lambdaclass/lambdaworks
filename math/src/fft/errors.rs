use crate::field::errors::FieldError;
use thiserror::Error;

#[cfg(feature = "metal")]
use lambdaworks_gpu::metal::abstractions::errors::MetalError;

#[cfg(feature = "cuda")]
use lambdaworks_gpu::cuda::abstractions::errors::CudaError;

#[derive(Debug, Error)]
pub enum FFTError {
    #[error("Input length is {0}, which is not a power of two")]
    InputError(usize),
    #[error("A finite field related error has ocurred")]
    FieldError(#[from] FieldError),
    #[cfg(feature = "metal")]
    #[error("A Metal related error has ocurred")]
    MetalError(#[from] MetalError),
    #[cfg(feature = "cuda")]
    #[error("A CUDA related error has ocurred")]
    CudaError(#[from] CudaError),
}
