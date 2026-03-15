//! Polynomial commitment scheme (PCS) traits and implementations.
//!
//! Provides the IsMultilinearPCS trait for committing to multilinear polynomials,
//! the TrivialPCS implementation for testing, and the ZeromorphPCS implementation
//! backed by KZG (re-exported from lambdaworks-crypto).

pub use lambdaworks_crypto::commitments::multilinear::{IsMultilinearPCS, PcsError};
pub use lambdaworks_crypto::commitments::zeromorph::{
    ZeromorphCommitment, ZeromorphPCS, ZeromorphProof,
};

pub mod trivial;

pub use trivial::{TrivialCommitment, TrivialPCS, TrivialProof};
