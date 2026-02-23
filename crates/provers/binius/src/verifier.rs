//! Binius Verifier Implementation

use crate::constraints::ConstraintSystem;
use crate::fields::tower::Tower;
use crate::fri::{FriParams, FriVerifier};
use crate::prover::BiniusProof;
use crate::sumcheck::SumcheckVerifier;

/// Binius Verifier
pub struct BiniusVerifier {
    fri_verifier: FriVerifier,
}

impl BiniusVerifier {
    pub fn new(fri_params: FriParams) -> Self {
        Self {
            fri_verifier: FriVerifier::new(fri_params),
        }
    }

    /// Verify a Binius proof
    pub fn verify(
        &self,
        constraint_system: &ConstraintSystem,
        proof: &BiniusProof,
    ) -> Result<bool, VerificationError> {
        // Verify public inputs match
        if proof.public_inputs != constraint_system.public_inputs {
            return Err(VerificationError::PublicInputMismatch);
        }

        // Verify FRI proof
        if !self.fri_verifier.verify(&proof.fri_proof) {
            return Err(VerificationError::FriVerificationFailed);
        }

        // Verify sum-check proof (placeholder)
        if !SumcheckVerifier::verify(&proof.sumcheck_proof) {
            return Err(VerificationError::SumcheckVerificationFailed);
        }

        Ok(true)
    }
}

/// Verification errors
#[derive(Debug)]
pub enum VerificationError {
    PublicInputMismatch,
    FriVerificationFailed,
    SumcheckVerificationFailed,
}
