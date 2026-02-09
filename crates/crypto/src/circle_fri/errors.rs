use core::fmt;

#[derive(Debug)]
pub enum CircleFriError {
    MerkleTreeBuildFailed,
    MerkleProofFailed(usize),
    FinalValueMismatch,
    VerificationFailed(usize, &'static str),
}

impl fmt::Display for CircleFriError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
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
