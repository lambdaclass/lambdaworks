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
    MergeInvalidLength,
}
