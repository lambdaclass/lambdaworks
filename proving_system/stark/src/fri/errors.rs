use thiserror::Error;

#[derive(Debug, Error)]
pub enum FriError {
    #[error("Layer {0} evaluation size should be greater or equal than {0}, but is {1}")]
    LayerEvaluationError(usize, usize, usize),
    #[error("Could not get merkle proof in layer {0}")]
    LayerMerkleProofError(usize),
    #[error("Number of queries cannot be zero")]
    NumberOfQueriesError,
}
