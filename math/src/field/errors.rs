use thiserror::Error;

#[derive(Error, Debug)]
pub enum FieldError {
    #[error("Can't divide by zero")]
    DivisionByZero
}
