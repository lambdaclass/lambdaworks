use thiserror::Error;

#[derive(Debug, Error)]
pub enum FFTError {
    #[error("The polynomial order is not correct")]
    InvalidOrder(String),
    #[error("Could not calculate {1} root of unity")]
    RootOfUnityError(String, u64),
}
