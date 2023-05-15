use lambdaworks_fft::errors::FFTError;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ProverError {
    #[error("Could not evaluate polynomial: {0}")]
    PolynomialEvaluationError(FFTError),
}
