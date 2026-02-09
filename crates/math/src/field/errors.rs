use core::fmt;

#[derive(Debug)]
pub enum FieldError {
    DivisionByZero,
    /// Returns order of the calculated root of unity
    RootOfUnityError(u64),
    /// Can't calculate inverse of zero
    InvZeroError,
}

impl fmt::Display for FieldError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FieldError::DivisionByZero => write!(f, "Division by zero"),
            FieldError::RootOfUnityError(order) => {
                write!(f, "Cannot find primitive root of unity for order {}", order)
            }
            FieldError::InvZeroError => {
                write!(f, "Cannot calculate multiplicative inverse of zero")
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for FieldError {}
