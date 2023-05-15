use lambdaworks_fft::errors::FFTError;
use lambdaworks_math::field::errors::FieldError;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum StarkError {
    #[error("Could not evaluate polynomial: {0}")]
    PolynomialEvaluationError(FFTError),
    #[error("Number of trace term gammas should be {0} * {1} = {} but is {2}", .0 * .1)]
    DeepTraceTermGammasError(usize, usize, usize),
    #[error("Number of composition poly even evaluations should be {0} but number of LDE roots of unity is {1}")]
    CompositionPolyEvenEvaluationsError(usize, usize),
    #[error("Number of composition poly odd evaluations should be {0} but number of LDE roots of unity is {1}")]
    CompositionPolyOddEvaluationsError(usize, usize),
    #[error("Could not create domain: {0}")]
    DomainFieldError(FieldError),
    #[error("Could not create domain: {0}")]
    DomainFFTError(FFTError),
}
