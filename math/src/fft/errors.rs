use crate::field::errors::FieldError;
use thiserror::Error;

#[cfg(feature = "metal")]
use lambdaworks_gpu::metal::abstractions::errors::MetalError;

#[cfg(feature = "cuda")]
use lambdaworks_gpu::cuda::abstractions::errors::CudaError;

#[derive(Debug, Error)]
pub enum FFTError {
    #[error("Could not calculate root of unity")]
    RootOfUnityError(u64),
    #[error("Input length is {0}, which is not a power of two")]
    InputError(usize),
    #[error("Order should be less than or equal to 63, but is {0}")]
    OrderError(u64),
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
            FieldError::InvZeroError => {
                panic!("Can't calculate inverse of zero during FFT");
            }
            FieldError::RootOfUnityError(order) => FFTError::RootOfUnityError(order),
        }
    }
}
