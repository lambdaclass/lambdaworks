use lambdaworks_math::field::errors::FieldError;
use thiserror::Error;

#[cfg(feature = "metal")]
use lambdaworks_gpu::metal::abstractions::errors::MetalError;

#[derive(Debug, Error)]
pub enum FFTError {
    #[error("Could not calculate {1} root of unity")]
    RootOfUnityError(String, u64),
    #[error("Input length is {0}, which is not a power of two")]
    InputError(usize),
    #[cfg(feature = "metal")]
    #[error("A Metal related error has ocurred")]
    MetalError(MetalError),
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

#[cfg(feature = "metal")]
impl From<MetalError> for FFTError {
    fn from(error: MetalError) -> Self {
        FFTError::MetalError(error)
    }
}
