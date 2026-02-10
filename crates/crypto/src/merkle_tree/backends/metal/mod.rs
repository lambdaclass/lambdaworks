//! Metal GPU-accelerated Merkle tree backend.
//!
//! This module provides GPU-accelerated Merkle tree construction using
//! Apple's Metal framework for compute shaders.

mod backend;

pub use backend::MetalPoseidonBackend;
