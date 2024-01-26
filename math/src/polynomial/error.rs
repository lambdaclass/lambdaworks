use core::fmt::Display;

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
