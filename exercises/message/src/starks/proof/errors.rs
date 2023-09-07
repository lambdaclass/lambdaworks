use thiserror::Error;

#[derive(Debug, Error)]
pub enum InsecureOptionError {
    #[error("Field size is not large enough")]
    FieldSize,
    #[error("The number of security bits is not large enough")]
    SecurityBits,
}
