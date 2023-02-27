use thiserror::Error;

#[derive(Debug, Error)]
pub enum ProverError {
    #[error("Could not compute composition polynomial")]
    CompositionPolyError(winter_prover::ProverError),
}
