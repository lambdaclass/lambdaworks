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
