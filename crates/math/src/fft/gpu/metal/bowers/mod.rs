//! Bowers-G Metal NTT + fused LDE pipeline.
//!
//! Design: docs/superpowers/specs/2026-05-19-metal-bowers-fft-lde-design.md
//! The bitrev leaf-ordering convention is load-bearing — see the spec.

pub mod dispatcher;
pub mod twiddles;

#[cfg(test)]
mod tests;

pub use dispatcher::{metal_bowers_lde, metal_bowers_lde_fp3};
