#[derive(Debug, PartialEq, Eq)]
pub enum SolverError {
    InconsistentSystem,
    UnableToSolve,
}
