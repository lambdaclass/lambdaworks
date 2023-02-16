use thiserror::Error;

#[derive(Error, Debug)]
pub enum HashError {
    #[error("Input for this hash is incorrect")]
    InputIsIncorrect,
    #[error("An error occurred while hashing")]
    ErrorHashing,
}
