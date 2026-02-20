//! Metal GPU-accelerated Merkle tree backend.
//!
//! This module provides GPU-accelerated Merkle tree construction using
//! Apple's Metal framework for compute shaders.

mod backend;
mod poseidon2;

pub use backend::MetalPoseidonBackend;
pub use poseidon2::MetalPoseidon2Backend;
