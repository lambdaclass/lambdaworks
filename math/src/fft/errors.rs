use crate::field::errors::FieldError;
use thiserror::Error;

#[cfg(feature = "metal")]
use lambdaworks_gpu::metal::abstractions::errors::MetalError;

#[cfg(feature = "cuda")]
use crate::gpu::cuda::abstractions::errors::CudaError;

#[derive(Debug, Error)]
pub enum FFTError {
    #[error("Could not calculate {1} root of unity")]
    RootOfUnityError(String, u64),
    #[error("Input length is {0}, which is not a power of two")]
    InputError(usize),
    #[cfg(feature = "metal")]
    #[error("A Metal related error has ocurred")]
    MetalError(#[from] MetalError),
    #[cfg(feature = "cuda")]
    #[error("A CUDA related error has ocurred")]
    CudaError(#[from] CudaError),
}

impl From<FieldError> for FFTError {
    fn from(error: FieldError) -> Self {
        match error {
            FieldError::DivisionByZero => {
                panic!("Can't divide by zero during FFT");
            }
            FieldError::RootOfUnityError(error, order) => FFTError::RootOfUnityError(error, order),
        }
    }
}
