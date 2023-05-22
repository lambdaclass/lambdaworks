use lambdaworks_fft::errors::FFTError;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum AIRError {
    #[error("Number of AIR transition constraints should be {0} but it's {1}")]
    TransitionConstraints(usize, usize),
    #[error("Could not compute AIR transition divisors: {0}")]
    TransitionDivisors(FFTError),
    #[error("Could not compute trace polynomials: {0}")]
    TracePolynomialsComputation(FFTError),
    #[error("Could not create constraint evaluator")]
    ConstraintEvaluatorCreation,
    #[error("Row index {0} is out of bounds for table with {1} rows")]
    RowIndexOutOfTableBounds(usize, usize),
    #[error("Row index {0} is out of bounds for frame with {1} rows")]
    RowIndexOutOfFrameBounds(usize, usize),
    #[error("Attempt to create table with zero columns")]
    TableColumns,
    #[error("Attempt to create table with columns with different lengths")]
    TableColumnLengths,
    #[error("Attempt to create frame with zero columns")]
    FrameColumns,
    #[error("Could not compute transition divisors because exemption polynomial {0} is zero")]
    TransitionDivisorExemption(usize),
}
