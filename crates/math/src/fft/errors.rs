use core::fmt;

use crate::field::errors::FieldError;

#[cfg(feature = "cuda")]
use lambdaworks_gpu::cuda::abstractions::errors::CudaError;

#[derive(Debug)]
pub enum FFTError {
    RootOfUnityError(u64),
    InputError(usize),
    OrderError(u64),
    DomainSizeError(usize),
    /// Field error during FFT operation
    FieldError(FieldError),
    #[cfg(feature = "cuda")]
    CudaError(CudaError),
}

impl fmt::Display for FFTError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FFTError::RootOfUnityError(order) => {
                write!(
                    f,
                    "Could not calculate primitive root of unity for order {}",
                    order
                )
            }
            FFTError::InputError(len) => {
                write!(f, "Input length {} is not a power of two", len)
            }
            FFTError::OrderError(order) => {
                write!(
                    f,
                    "Order should be less than or equal to 63, but is {}",
                    order
                )
            }
            FFTError::DomainSizeError(size) => {
                write!(f, "Domain size {} exceeds two adicity of the field", size)
            }
            FFTError::FieldError(err) => {
                write!(f, "Field error during FFT: {}", err)
            }
            #[cfg(feature = "cuda")]
            FFTError::CudaError(err) => {
                write!(f, "CUDA error: {}", err)
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for FFTError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        #[cfg(feature = "cuda")]
        if let FFTError::CudaError(err) = self {
            return Some(err);
        }
        None
    }
}

#[cfg(feature = "cuda")]
impl From<CudaError> for FFTError {
    fn from(error: CudaError) -> Self {
        FFTError::CudaError(error)
    }
}

impl From<FieldError> for FFTError {
    fn from(error: FieldError) -> Self {
        match error {
            FieldError::RootOfUnityError(order) => FFTError::RootOfUnityError(order),
            other => FFTError::FieldError(other),
        }
    }
}
