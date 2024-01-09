use core::fmt::Display;

#[derive(Debug)]
pub enum MultilinearError {
    InvalidMergeLength,
    IncorrectNumberofEvaluationPoints(usize, usize),
    ChisAndEvalsMismatch(usize, usize),
}

impl Display for MultilinearError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            MultilinearError::InvalidMergeLength => write!(f, "The lengths of the multilinear polynomials being merged does not match"),
            MultilinearError::IncorrectNumberofEvaluationPoints(x, y) => write!(f, "The number of points to evaluate the polynomial does not match the number of variables (points: {x}, vars: {y})"),
            MultilinearError::ChisAndEvalsMismatch(x, y) => write!(f, "The number of computed Chis does not match evaluations comprising the polynomial (chis: {x}, evals: {y})"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for MultilinearError {}
