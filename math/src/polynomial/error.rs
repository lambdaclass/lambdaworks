use thiserror::Error;

#[derive(Error, Debug)]
pub enum Polynomial {
    #[error("{0}")]
    Multilinear(#[from] Multilinear),
}

#[derive(Error, Debug)]
pub enum Multilinear {
    #[error("{0}")]
    DenseMultilinear(#[from] DenseMultilinear),
}

#[derive(Error, Debug)]
pub enum DenseMultilinear {
    #[error("The lengths of the multilinear polynomials being merged does not match")]
    InvalidMergeLength,
    #[error("The number of points to evaluate the polynomial does not match the number of variables (points: {0}, vars: {1})")]
    IncorrectNumberofEvaluationPoints(usize, usize),
    #[error("The number of computed Chis does not match evaluations comprising the polynomial (chis: {0}, evals: {1})")]
    ChisAndEvalsMismatch(usize, usize),
}
