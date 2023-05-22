use lambdaworks_fft::errors::FFTError;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum AIRError {
    #[error("Number of AIR transition constraints should be {0} but it's {1}")]
    TransitionConstraints(usize, usize),
    #[error(transparent)]
    FFT(#[from] FFTError),
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
    #[error(
        "Could not compute composition polynomial {0} because boundary zerofier polynomial is zero"
    )]
    CPBoundaryConstraintDivision(usize),
    #[error("Could not compute composition polynomial {0} because transition zerofier polynomial is zero")]
    CPTransitionConstraintDivision(usize),
}
