//! # Reed-Solomon Codes
//!
//! This crate provides an educational implementation of Reed-Solomon codes,
//! including encoding, unique decoding (Berlekamp-Welch), and list decoding
//! (Sudan and Guruswami-Sudan algorithms).
//!
//! ## Overview
//!
//! Reed-Solomon codes are a family of error-correcting codes with optimal
//! distance properties (they are Maximum Distance Separable). They are widely
//! used in:
//! - Data storage (CDs, DVDs, QR codes, RAID systems)
//! - Digital communications
//! - Zero-knowledge proof systems (STARKs, FRI protocol)
//!
//! ## Code Parameters
//!
//! An RS[n, k, d] code over a field F_q has:
//! - n: code length (number of evaluation points)
//! - k: dimension (message length / polynomial degree + 1)
//! - d = n - k + 1: minimum distance (achieves Singleton bound)
//!
//! ## Modules
//!
//! - [`reed_solomon`]: Core RS code structure and encoding
//! - [`distance`]: Hamming distance and weight computations
//! - [`berlekamp_welch`]: Unique decoding up to (n-k)/2 errors
//! - [`sudan`]: List decoding up to n - sqrt(2nk) errors
//! - [`guruswami_sudan`]: Optimal list decoding up to n - sqrt(nk) errors
//! - [`polynomial_utils`]: Bivariate polynomials and root finding

pub mod berlekamp_welch;
pub mod distance;
pub mod guruswami_sudan;
pub mod polynomial_utils;
pub mod reed_solomon;
pub mod sudan;

// Re-export the field we use throughout the examples
pub use lambdaworks_math::field::element::FieldElement;
pub use lambdaworks_math::field::fields::fft_friendly::babybear::Babybear31PrimeField;
pub use lambdaworks_math::polynomial::Polynomial;

/// Type alias for field elements over BabyBear
pub type FE = FieldElement<Babybear31PrimeField>;

/// The BabyBear prime: p = 2^31 - 2^27 + 1 = 2013265921
///
/// This field is chosen because:
/// - It has TWO_ADICITY = 24, allowing FFT domains up to 2^24 elements
/// - Field operations fit in 32 bits, making arithmetic efficient
/// - It's widely used in modern proof systems (Plonky3, RISC Zero)
pub const BABYBEAR_PRIME: u64 = 2013265921;
