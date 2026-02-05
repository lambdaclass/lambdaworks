//! Groth16 zero-knowledge proof system implementation.
//!
//! This crate provides a complete implementation of the Groth16 zk-SNARK proving system,
//! which enables proving knowledge of a witness satisfying an R1CS constraint system
//! without revealing the witness itself.
//!
//! # Overview
//!
//! Groth16 is a pairing-based zk-SNARK with:
//! - **Constant-size proofs**: 3 group elements regardless of circuit size
//! - **Fast verification**: Single pairing check
//! - **Trusted setup**: Requires a circuit-specific ceremony
//!
//! # Usage
//!
//! ```ignore
//! use lambdaworks_groth16::{QuadraticArithmeticProgram, setup, verify, Prover};
//!
//! // 1. Define your circuit as a QAP
//! let qap = QuadraticArithmeticProgram::from_variable_matrices(
//!     num_public_inputs, &l_matrix, &r_matrix, &o_matrix
//! )?;
//!
//! // 2. Run trusted setup (use MPC ceremony in production!)
//! let (proving_key, verifying_key) = setup(&qap)?;
//!
//! // 3. Generate a proof with your witness
//! let proof = Prover::prove(&witness, &qap, &proving_key)?;
//!
//! // 4. Verify the proof
//! let is_valid = verify(&verifying_key, &proof, &public_inputs)?;
//! ```
//!
//! # Security Warning
//!
//! The [`setup`] function uses local randomness for the toxic waste.
//! In production, you MUST use a multi-party computation (MPC) ceremony
//! to generate the proving and verifying keys securely.
//!
//! # Curve Configuration
//!
//! This implementation is configured for BLS12-381. To use a different curve,
//! modify the type aliases in [`common`]. The curve must be pairing-friendly
//! and have a scalar field that supports FFT (i.e., has roots of unity).
//!
//! # References
//!
//! - [Groth16 paper](https://eprint.iacr.org/2016/260)
//! - [Vitalik's QAP explanation](https://vitalik.ca/general/2016/12/10/qap.html)

pub mod common;
pub mod errors;
pub mod qap;
pub mod r1cs;

mod prover;
mod setup;
mod verifier;

pub use errors::Groth16Error;
pub use prover::{Proof, Prover};
pub use qap::QuadraticArithmeticProgram;
pub use r1cs::*;
pub use setup::*;
pub use verifier::verify;
