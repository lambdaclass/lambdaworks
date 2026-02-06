use thiserror::Error;

#[derive(Debug, PartialEq, Eq, Error)]
pub enum SolverError {
    #[error("Constraint system is inconsistent and has no solution")]
    InconsistentSystem,
    #[error("Unable to solve the constraint system")]
    UnableToSolve,
}
