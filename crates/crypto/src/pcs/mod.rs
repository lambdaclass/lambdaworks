//! Polynomial Commitment Schemes (PCS).
//!
//! This module provides a unified interface for polynomial commitment schemes,
//! allowing different implementations (KZG, FRI, IPA, etc.) to be used
//! interchangeably in proof systems.
//!
//! # Overview
//!
//! A polynomial commitment scheme allows a prover to:
//! 1. **Commit** to a polynomial, producing a short commitment
//! 2. **Open** the commitment at evaluation points, producing proofs
//! 3. **Verify** that the evaluations are correct
//!
//! # Trait Hierarchy
//!
//! - [`PolynomialCommitmentScheme`]: Base trait with setup, commit, open, verify
//! - [`BatchPCS`]: Extension for efficient batch operations
//! - [`SerializablePCS`]: Extension for serialization support
//! - [`HidingPCS`]: Extension for hiding (zero-knowledge) commitments
//!
//! # Implementations
//!
//! - [`kzg::KZG`]: Kate-Zaverucha-Goldberg scheme (pairing-based)
//! - [`fri::FRIPcs`]: FRI-based scheme (hash-based, wrapper around STARK FRI)
//!
//! # Example
//!
//! ```ignore
//! use lambdaworks_crypto::pcs::{PolynomialCommitmentScheme, kzg::KZG};
//!
//! // Setup
//! let pp = KZG::<F, BLS12381>::setup(max_degree, &mut rng)?;
//! let (ck, vk) = KZG::trim(&pp, degree)?;
//!
//! // Commit
//! let polynomial = Polynomial::new(&coefficients);
//! let (commitment, state) = KZG::commit(&ck, &polynomial)?;
//!
//! // Open
//! let point = FieldElement::from(42u64);
//! let proof = KZG::open(&ck, &polynomial, &state, &point)?;
//!
//! // Verify
//! let evaluation = polynomial.evaluate(&point);
//! assert!(KZG::verify(&vk, &commitment, &point, &evaluation, &proof)?);
//! ```
//!
//! # Design Notes
//!
//! This design is inspired by:
//! - [arkworks poly-commit](https://github.com/arkworks-rs/poly-commit)
//! - [Plonky3](https://github.com/Plonky3/Plonky3)
//!
//! Key differences from the old `IsCommitmentScheme` trait:
//! - Proper error handling with `Result` types
//! - Separate `CommitterKey` and `VerifierKey` types
//! - Explicit `setup` and `trim` methods
//! - Separate `Commitment` and `Proof` types
//! - Transcript integration for Fiat-Shamir

pub mod error;
pub mod traits;
pub mod transcript;

pub mod kzg;

#[cfg(feature = "alloc")]
pub mod fri;

// Re-exports
pub use error::{PCSError, PCSErrorKind};
pub use traits::HidingPCS;
pub use traits::PolynomialCommitmentScheme;
#[cfg(feature = "alloc")]
pub use traits::{BatchPCS, SerializablePCS};
pub use transcript::{AsBytes, FromBytes, PCSTranscript, PCSTranscriptExt};
