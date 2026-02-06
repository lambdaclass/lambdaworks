use thiserror::Error;

/// Errors that can occur during Groth16 proving, verification, or setup.
#[derive(Debug, Error)]
pub enum Groth16Error {
    #[error("FFT error: {0}")]
    FFTError(String),
    #[error("Batch inversion failed, likely due to zero element")]
    BatchInversionFailed,
    #[error("Pairing error: {0}")]
    PairingError(String),
    #[error("Multi-scalar multiplication error: {0}")]
    MSMError(String),
    #[error("QAP computation error: {0}")]
    QAPError(String),
    #[error("Setup error: {0}")]
    SetupError(String),
}

impl Groth16Error {
    /// Creates an MSMError from any Debug-printable error.
    pub fn msm<E: core::fmt::Debug>(e: E) -> Self {
        Self::MSMError(format!("{:?}", e))
    }

    /// Creates a PairingError from any Debug-printable error.
    pub fn pairing<E: core::fmt::Debug>(e: E) -> Self {
        Self::PairingError(format!("{:?}", e))
    }
}
