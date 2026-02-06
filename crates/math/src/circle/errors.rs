use core::fmt;

#[derive(Debug)]
pub enum CircleError {
    PointDoesntSatisfyCircleEquation,
}

impl fmt::Display for CircleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CircleError::PointDoesntSatisfyCircleEquation => {
                write!(
                    f,
                    "Point does not satisfy the circle equation x^2 + y^2 = 1"
                )
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for CircleError {}
