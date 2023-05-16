use thiserror::Error;

#[derive(Debug, Error)]
pub enum FriError {
    #[error("Layer {0} evaluation size should be greater or equal than {0}, but is {1}")]
    LayerEvaluation(usize, usize, usize),
    #[error("Could not get merkle proof in layer {0}")]
    LayerMerkleProof(usize),
    #[error("Number of queries cannot be zero")]
    NumberOfQueries,
}
