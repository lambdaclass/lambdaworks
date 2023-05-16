use lambdaworks_fft::errors::FFTError;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum AIRError {
    #[error("AIR transition constraints field is set to zero")]
    TransitionConstraintsError,
    #[error("Could not compute AIR transition divisors: {0}")]
    TransitionDivisorsError(FFTError),
}
