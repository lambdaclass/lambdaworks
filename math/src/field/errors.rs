use thiserror::Error;

#[derive(Error, Debug)]
pub enum FieldError {
    #[error("Can't divide by zero")]
    DivisionByZero,
    #[error("Could not calculate {1} root of unity")]
    RootOfUnityError(String, u64),
}
