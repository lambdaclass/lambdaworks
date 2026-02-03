//! Error types for Polynomial Commitment Schemes.

use core::fmt;

/// Errors that can occur during PCS operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PCSError {
    /// Setup failed due to invalid parameters.
    SetupError(PCSErrorKind),

    /// Commitment operation failed.
    CommitmentError(PCSErrorKind),

    /// Opening operation failed.
    OpeningError(PCSErrorKind),

    /// Verification failed.
    VerificationError(PCSErrorKind),

    /// Polynomial degree exceeds maximum supported.
    DegreeTooLarge {
        /// Maximum supported degree.
        max: usize,
        /// Actual polynomial degree.
        actual: usize,
    },

    /// Serialization or deserialization error.
    SerializationError(PCSErrorKind),

    /// Invalid proof structure.
    InvalidProof,

    /// Invalid commitment structure.
    InvalidCommitment,

    /// Mismatched lengths in batch operations.
    LengthMismatch {
        /// Expected length.
        expected: usize,
        /// Actual length.
        actual: usize,
    },

    /// Division by zero or inverse of zero.
    DivisionByZero,

    /// Point not on curve (for pairing-based PCS).
    InvalidPoint,

    /// Pairing check failed.
    PairingCheckFailed,
}

/// Detailed error information.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PCSErrorKind {
    /// Error message.
    pub message: &'static str,
}

impl PCSErrorKind {
    /// Create a new error kind with the given message.
    pub const fn new(message: &'static str) -> Self {
        Self { message }
    }
}

impl fmt::Display for PCSError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PCSError::SetupError(kind) => write!(f, "Setup error: {}", kind.message),
            PCSError::CommitmentError(kind) => write!(f, "Commitment error: {}", kind.message),
            PCSError::OpeningError(kind) => write!(f, "Opening error: {}", kind.message),
            PCSError::VerificationError(kind) => write!(f, "Verification error: {}", kind.message),
            PCSError::DegreeTooLarge { max, actual } => {
                write!(
                    f,
                    "Polynomial degree {} exceeds maximum supported degree {}",
                    actual, max
                )
            }
            PCSError::SerializationError(kind) => {
                write!(f, "Serialization error: {}", kind.message)
            }
            PCSError::InvalidProof => write!(f, "Invalid proof"),
            PCSError::InvalidCommitment => write!(f, "Invalid commitment"),
            PCSError::LengthMismatch { expected, actual } => {
                write!(f, "Length mismatch: expected {}, got {}", expected, actual)
            }
            PCSError::DivisionByZero => write!(f, "Division by zero"),
            PCSError::InvalidPoint => write!(f, "Invalid elliptic curve point"),
            PCSError::PairingCheckFailed => write!(f, "Pairing check failed"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for PCSError {}

impl PCSError {
    /// Create a setup error with the given message.
    pub const fn setup(message: &'static str) -> Self {
        PCSError::SetupError(PCSErrorKind::new(message))
    }

    /// Create a commitment error with the given message.
    pub const fn commitment(message: &'static str) -> Self {
        PCSError::CommitmentError(PCSErrorKind::new(message))
    }

    /// Create an opening error with the given message.
    pub const fn opening(message: &'static str) -> Self {
        PCSError::OpeningError(PCSErrorKind::new(message))
    }

    /// Create a verification error with the given message.
    pub const fn verification(message: &'static str) -> Self {
        PCSError::VerificationError(PCSErrorKind::new(message))
    }

    /// Create a serialization error with the given message.
    pub const fn serialization(message: &'static str) -> Self {
        PCSError::SerializationError(PCSErrorKind::new(message))
    }

    /// Create a degree too large error.
    pub const fn degree_too_large(max: usize, actual: usize) -> Self {
        PCSError::DegreeTooLarge { max, actual }
    }

    /// Create a length mismatch error.
    pub const fn length_mismatch(expected: usize, actual: usize) -> Self {
        PCSError::LengthMismatch { expected, actual }
    }
}
