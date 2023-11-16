use lambdaworks_math::field::{element::FieldElement, traits::IsField};
use thiserror::Error;
use super::cq::errors::CachedQuotientError;

#[derive(Error, Debug, PartialEq)]
pub enum Error<F: IsField> {
    #[error("{0}")]
    Table(#[from] TableError<F>),
    #[error("{0}")]
    CachedQuotient(#[from] CachedQuotientError),
}

#[derive(Error, Debug, PartialEq)]
pub enum TableError<F: IsField> {
    #[error("DuplicateValueInTable:")]
    DuplicateValueInTable(FieldElement<F>),
    #[error("Table size not power of 2: {0}")]
    TableSizeNotPow2(usize)
}
