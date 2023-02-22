use thiserror::Error;

#[derive(Debug, Error)]
pub enum ProverError {
    #[error("The order of the polynomial is not correct")]
    CompositionPolyError(winterfell::prover::ProverError),
}
