use core::fmt;

/// Errors that can occur during Groth16 proving, verification, or setup.
#[derive(Debug)]
pub enum Groth16Error {
    /// Error during FFT operation
    FFTError(String),
    /// Batch inversion failed (likely due to zero element)
    BatchInversionFailed,
    /// Pairing computation failed
    PairingError(String),
    /// Multi-scalar multiplication failed
    MSMError(String),
    /// QAP computation error
    QAPError(String),
    /// Setup error
    SetupError(String),
}

impl fmt::Display for Groth16Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Groth16Error::FFTError(msg) => write!(f, "FFT error: {}", msg),
            Groth16Error::BatchInversionFailed => write!(f, "Batch inversion failed"),
            Groth16Error::PairingError(msg) => write!(f, "Pairing error: {}", msg),
            Groth16Error::MSMError(msg) => write!(f, "MSM error: {}", msg),
            Groth16Error::QAPError(msg) => write!(f, "QAP error: {}", msg),
            Groth16Error::SetupError(msg) => write!(f, "Setup error: {}", msg),
        }
    }
}

impl std::error::Error for Groth16Error {}
