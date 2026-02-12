use core::fmt;

#[derive(Debug)]
pub enum CircleFriError {
    InvalidEvaluationsLength { expected: usize, got: usize },
    QueryIndexOutOfBounds { index: usize, domain_size: usize },
    InconsistentProof(&'static str),
    MerkleTreeBuildFailed,
    MerkleProofFailed(usize),
    FinalValueMismatch,
    VerificationFailed(usize, &'static str),
}

impl fmt::Display for CircleFriError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidEvaluationsLength { expected, got } => {
                write!(
                    f,
                    "Evaluations length {got} does not match domain size {expected}"
                )
            }
            Self::QueryIndexOutOfBounds { index, domain_size } => {
                write!(
                    f,
                    "Query index {index} is out of bounds for domain size {domain_size}"
                )
            }
            Self::InconsistentProof(msg) => {
                write!(f, "Inconsistent proof: {msg}")
            }
            Self::MerkleTreeBuildFailed => write!(f, "Failed to build Merkle tree for FRI layer"),
            Self::MerkleProofFailed(pos) => {
                write!(f, "Failed to get Merkle proof at position {pos}")
            }
            Self::FinalValueMismatch => {
                write!(f, "Final FRI value does not match expected constant")
            }
            Self::VerificationFailed(layer, msg) => {
                write!(f, "FRI verification failed at layer {layer}: {msg}")
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for CircleFriError {}
