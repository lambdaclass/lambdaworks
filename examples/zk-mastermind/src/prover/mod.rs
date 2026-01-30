//! Prover and Verifier for ZK Mastermind
//!
//! This module provides wrappers around the stark_platinum_prover
//! for generating and verifying STARK proofs.

pub mod prove;
pub mod verify;

pub use prove::{
    deserialize_proof, generate_proof, generate_proof_with_options, proof_size, serialize_proof,
    Proof,
};

pub use verify::{verify_proof, verify_proof_with_options, verify_proof_with_trace_length};
