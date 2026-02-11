// CUDA-accelerated Merkle tree backend
//
// This module provides GPU-accelerated Merkle tree construction using CUDA.
// It implements the IsMerkleTreeBackend trait for parallel tree building.

mod backend;
mod state;

pub use backend::CudaPoseidonBackend;
pub use state::CudaMerkleState;
