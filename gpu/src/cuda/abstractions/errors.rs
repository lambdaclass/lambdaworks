use thiserror::Error;

#[derive(Debug, Error)]
pub enum CudaError {
    #[error("The order of polynomial + 1 should a be power of 2. Got: {}")]
    InvalidOrder(usize),
    #[error("An error occured while working in CPU with fields")]
    FieldError(#[from] FieldError),
}
