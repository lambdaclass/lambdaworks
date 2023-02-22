use thiserror::Error;

#[derive(Debug, Error)]
pub enum ProverError {
    #[error("Could not compute composition polynomial")]
    CompositionPolyError(winterfell::prover::ProverError),
}
