use core::fmt::Display;

/// Errors that can occur during polynomial operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PolynomialError {
    /// Cannot divide by the zero polynomial.
    DivisionByZero,
    /// xgcd(0, 0) is undefined.
    XgcdBothZero,
}

impl Display for PolynomialError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            PolynomialError::DivisionByZero => {
                write!(f, "Cannot divide by the zero polynomial")
            }
            PolynomialError::XgcdBothZero => {
                write!(f, "xgcd(0, 0) is undefined")
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for PolynomialError {}

#[derive(Debug)]
pub enum MultilinearError {
    InvalidMergeLength,
    IncorrectNumberofEvaluationPoints(usize, usize),
    ChisAndEvalsLengthMismatch(usize, usize),
}

impl Display for MultilinearError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            MultilinearError::InvalidMergeLength => write!(f, "Invalid Merge Length"),
            MultilinearError::IncorrectNumberofEvaluationPoints(x, y) => {
                write!(f, "points: {x}, vars: {y}")
            }
            MultilinearError::ChisAndEvalsLengthMismatch(x, y) => {
                write!(f, "chis: {x}, evals: {y}")
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for MultilinearError {}
