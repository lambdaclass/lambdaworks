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
    /// Division by zero encountered during FFT computation
    DivisionByZero,
    /// Attempted to compute inverse of zero during FFT
    InverseOfZero,
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
            FFTError::DivisionByZero => {
                write!(f, "Division by zero encountered during FFT computation")
            }
            FFTError::InverseOfZero => {
                write!(f, "Cannot compute inverse of zero during FFT")
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
        match self {
            #[cfg(feature = "cuda")]
            FFTError::CudaError(e) => Some(e),
            _ => None,
        }
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
            FieldError::DivisionByZero => FFTError::DivisionByZero,
            FieldError::InvZeroError => FFTError::InverseOfZero,
            FieldError::RootOfUnityError(order) => FFTError::RootOfUnityError(order),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::format;

    #[test]
    fn test_fft_error_display_root_of_unity() {
        let error = FFTError::RootOfUnityError(8);
        let msg = format!("{}", error);
        assert!(msg.contains("root of unity"));
        assert!(msg.contains("8"));
    }

    #[test]
    fn test_fft_error_display_input_error() {
        let error = FFTError::InputError(7);
        let msg = format!("{}", error);
        assert!(msg.contains("Input length"));
        assert!(msg.contains("7"));
        assert!(msg.contains("power of two"));
    }

    #[test]
    fn test_fft_error_display_order_error() {
        let error = FFTError::OrderError(64);
        let msg = format!("{}", error);
        assert!(msg.contains("Order"));
        assert!(msg.contains("64"));
    }

    #[test]
    fn test_fft_error_display_domain_size() {
        let error = FFTError::DomainSizeError(1024);
        let msg = format!("{}", error);
        assert!(msg.contains("Domain size"));
        assert!(msg.contains("1024"));
        assert!(msg.contains("two adicity"));
    }

    #[test]
    fn test_fft_error_display_division_by_zero() {
        let error = FFTError::DivisionByZero;
        let msg = format!("{}", error);
        assert!(msg.contains("Division by zero"));
        assert!(msg.contains("FFT"));
    }

    #[test]
    fn test_fft_error_display_inverse_of_zero() {
        let error = FFTError::InverseOfZero;
        let msg = format!("{}", error);
        assert!(msg.contains("inverse of zero"));
        assert!(msg.contains("FFT"));
    }

    #[test]
    fn test_field_error_to_fft_error_division_by_zero() {
        let field_err = FieldError::DivisionByZero;
        let fft_err: FFTError = field_err.into();
        assert!(matches!(fft_err, FFTError::DivisionByZero));
    }

    #[test]
    fn test_field_error_to_fft_error_inverse_of_zero() {
        let field_err = FieldError::InvZeroError;
        let fft_err: FFTError = field_err.into();
        assert!(matches!(fft_err, FFTError::InverseOfZero));
    }

    #[test]
    fn test_field_error_to_fft_error_root_of_unity() {
        let field_err = FieldError::RootOfUnityError(16);
        let fft_err: FFTError = field_err.into();
        match fft_err {
            FFTError::RootOfUnityError(order) => assert_eq!(order, 16),
            _ => panic!("Expected RootOfUnityError"),
        }
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_fft_error_source_returns_none() {
        use std::error::Error;
        let error = FFTError::DivisionByZero;
        assert!(error.source().is_none());
    }
}
