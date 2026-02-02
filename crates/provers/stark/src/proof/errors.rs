use thiserror::Error;

#[derive(Debug, Error)]
pub enum InsecureOptionError {
    #[error("Field size is not big enough for the required security level")]
    FieldSize,
    #[error("Number of security bits is insufficient")]
    LowSecurityBits,
}
