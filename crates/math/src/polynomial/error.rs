use core::fmt;

/// Errors that can occur during polynomial operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PolynomialError {
    /// Cannot divide by the zero polynomial.
    DivisionByZero,
    /// xgcd(0, 0) is undefined.
    XgcdBothZero,
}

impl fmt::Display for PolynomialError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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

impl fmt::Display for MultilinearError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MultilinearError::InvalidMergeLength => {
                write!(f, "Invalid merge length for multilinear polynomial")
            }
            MultilinearError::IncorrectNumberofEvaluationPoints(points, vars) => {
                write!(
                    f,
                    "Incorrect number of evaluation points: got {} points, expected {} variables",
                    points, vars
                )
            }
            MultilinearError::ChisAndEvalsLengthMismatch(chis, evals) => {
                write!(
                    f,
                    "Chis and evals length mismatch: chis has {} elements, evals has {} elements",
                    chis, evals
                )
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for MultilinearError {}
