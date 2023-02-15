use thiserror::Error;

#[derive(Debug, Error)]
pub enum FFTError {
    #[error("The order of the polynomial is not correct")]
    InvalidOrder(String),
    #[error("Could not calculate {1} root of unity")]
    RootOfUnityError(String, u64),
}
