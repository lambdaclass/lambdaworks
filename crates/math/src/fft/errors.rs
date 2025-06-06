use core::fmt::Display;

use crate::field::errors::FieldError;

#[cfg(feature = "cuda")]
use lambdaworks_gpu::cuda::abstractions::errors::CudaError;

#[derive(Debug)]
pub enum FFTError {
    RootOfUnityError(u64),
    InputError(usize),
    OrderError(u64),
    DomainSizeError(usize),
    #[cfg(feature = "cuda")]
    CudaError(CudaError),
}

impl Display for FFTError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            FFTError::RootOfUnityError(_) => write!(f, "Could not calculate root of unity"),
            FFTError::InputError(v) => {
                write!(f, "Input length is {v}, which is not a power of two")
            }
            FFTError::OrderError(v) => {
                write!(f, "Order should be less than or equal to 63, but is {v}")
            }
            FFTError::DomainSizeError(_) => {
                write!(f, "Domain size exceeds two adicity of the field")
            }
            #[cfg(feature = "cuda")]
            FFTError::CudaError(_) => {
                write!(f, "A CUDA related error has ocurred")
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for FFTError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            #[cfg(feature = "cuda")]
            FFTError::CudaError(_) => Some(e),
            _ => None,
        }
    }
}

#[cfg(feature = "cuda")]
impl From<CudaError> for FFTError {
    fn from(error: CudaError) -> Self {
        Self::CudaError(error)
    }
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
