use thiserror::Error;
#[derive(Error, Debug, PartialEq)]
pub enum CachedQuotientError {
    #[error("Shits F*cked in the worse way possible")]
    Error,
}