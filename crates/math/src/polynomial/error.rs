use core::fmt::{self, Display};

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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MultilinearError {
    InvalidMergeLength,
    IncorrectNumberOfEvaluationPoints(usize, usize),
    ChisAndEvalsLengthMismatch(usize, usize),
}

impl fmt::Display for MultilinearError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MultilinearError::InvalidMergeLength => {
                write!(f, "Invalid merge length for multilinear polynomial")
            }
            MultilinearError::IncorrectNumberOfEvaluationPoints(points, vars) => {
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

#[derive(Debug)]
pub enum InterpolateError {
    UnequalLengths(usize, usize),
    NonUniqueXs,
}

impl Display for InterpolateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InterpolateError::UnequalLengths(x, y) => {
                write!(
                    f,
                    "xs and ys must be the same length: got {} xs and {} ys",
                    x, y
                )
            }
            InterpolateError::NonUniqueXs => write!(f, "xs values must be unique"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for InterpolateError {}
