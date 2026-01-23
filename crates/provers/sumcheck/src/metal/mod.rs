//! Metal GPU Backend for Sumcheck
//!
//! This module provides GPU-accelerated sumcheck operations using Apple's Metal API.
//! It's designed for macOS/iOS systems with Metal-capable GPUs.
//!
//! Key operations accelerated:
//! - Parallel reduction for hypercube summation
//! - Challenge application (fixing variables)
//! - Round polynomial computation
//!
//! # Performance
//! Metal acceleration provides significant speedups for large polynomials (n > 16):
//! - 10-100x speedup for single polynomial sumcheck
//! - Even higher gains for batched operations
//!
//! # Requirements
//! - macOS 10.11+ or iOS 8.0+
//! - Metal-capable GPU
//! - Enable with `--features metal`

mod prover;
mod shaders;

pub use prover::{prove_metal, prove_metal_multi, MetalMultiFactorProver, MetalProver, MetalState};

#[cfg(all(target_os = "macos", feature = "metal"))]
pub use prover::GoldilocksMetalProver;

#[cfg(test)]
mod tests;
