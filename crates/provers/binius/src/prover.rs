//! Binius Prover Implementation

use crate::constraints::{ConstraintSystem, Witness};
use crate::fields::tower::Tower;
use crate::fri::{FriParams, FriProof, FriProver};
use crate::polynomial::MultilinearPolynomial;
use crate::sumcheck::{SumcheckProof, SumcheckProver};

/// Binius Prover
pub struct BiniusProver {
    fri_prover: FriProver,
}

impl BiniusProver {
    pub fn new(fri_params: FriParams) -> Self {
        Self {
            fri_prover: FriProver::new(fri_params),
        }
    }

    /// Generate proof for a computation
    ///
    /// Input:
    /// - constraint_system: The circuit being proved
    /// - witness: The private inputs ( witness values)
    /// - public_inputs: Public inputs
    pub fn prove(&self, constraint_system: &ConstraintSystem, witness: &Witness) -> BiniusProof {
        // Step 1: Build multilinear polynomial from witness
        let poly = self.build_polynomial(constraint_system, witness);

        // Step 2: Commit to polynomial using FRI
        let fri_proof = self.fri_prover.prove(&poly);

        // Step 3: Run sum-check protocol (placeholder)
        let sumcheck_proof = SumcheckProver::prove(&poly, Tower::zero());

        BiniusProof {
            fri_proof,
            sumcheck_proof,
            public_inputs: constraint_system.public_inputs.clone(),
        }
    }

    /// Build multilinear polynomial from constraint system and witness
    fn build_polynomial(
        &self,
        constraint_system: &ConstraintSystem,
        witness: &Witness,
    ) -> MultilinearPolynomial {
        // Placeholder: create a simple polynomial from witness values
        MultilinearPolynomial::new(witness.values.clone())
            .unwrap_or_else(|_| MultilinearPolynomial::default())
    }
}

/// Binius Proof
#[derive(Clone, Debug)]
pub struct BiniusProof {
    pub fri_proof: FriProof,
    pub sumcheck_proof: SumcheckProof,
    pub public_inputs: Vec<Tower>,
}

impl BiniusProof {
    /// Serialize proof to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        // Placeholder serialization
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&self.fri_proof.commitment);
        bytes
    }
}
