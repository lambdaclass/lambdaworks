/// Errors that can occur during the Spartan proving protocol.
#[derive(Debug, thiserror::Error)]
pub enum SpartanError {
    #[error("R1CS error: {0}")]
    R1CSError(String),
    #[error("MLE encoding error: {0}")]
    MleError(String),
    #[error("Sumcheck error: {0}")]
    SumcheckError(String),
    #[error("PCS error: {0}")]
    PcsError(String),
    #[error("Verification failed: {0}")]
    VerificationFailed(String),
}
