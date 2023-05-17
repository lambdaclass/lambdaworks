use lambdaworks_fft::errors::FFTError;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum AIRError {
    #[error("AIR transition constraints field is set to zero")]
    TransitionConstraints,
    #[error("Could not compute AIR transition divisors: {0}")]
    TransitionDivisors(FFTError),
    #[error("Could not compute trace polynomials: {0}")]
    TracePolynomialsComputation(FFTError),
    #[error("Could not create constraint evaluator")]
    ConstraintEvaluatorCreation,
    #[error("Row index {0} is out of bounds for table with {1} rows")]
    RowIndexOutOfTableBounds(usize, usize),
}
