//! Binius Prover Implementation
//!
//! This module provides a full implementation of the Binius zk-SNARK prover
//! using binary fields (towers of binary field extensions).
//!
//! ## Architecture
//!
//! The prover is built in several layers:
//! - **Binary Fields**: Tower field arithmetic (already exists in lambdaworks-math)
//! - **Multilinear Polynomials**: Representation of polynomials in multilinear form
//! - **FRI**: Fast Reed-Solomon IOP commitment scheme
//! - **Sum-check Protocol**: Interactive proof for polynomial evaluation
//! - **Constraint System**: Circuit representation for computations
//!
//! ## References
//!
//! - [Binius Paper](https://eprint.iacr.org/2023/1786)
//! - [Binius64 Implementation](https://github.com/binius-zk/binius64)
//! - [Vitalik's Binius Explanation](https://vitalik.eth.limo/general/2024/04/29/binius.html)

pub mod constraints;
pub mod fields;
pub mod fri;
pub mod polynomial;
pub mod prover;
pub mod sumcheck;
pub mod verifier;

pub use constraints::{ConstraintSystem, Gate, Variable, Witness};
pub use fields::{
    tower::{self, BinaryFieldError, Tower, TowerFieldElement},
    FieldLevel,
};
pub use fri::{FriParams, FriProof, FriProver, FriVerifier};
pub use polynomial::MultilinearPolynomial;
pub use prover::{BiniusProof, BiniusProver};
pub use verifier::{BiniusVerifier, VerificationError};
