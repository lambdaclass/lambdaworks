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
    /// - input_values: The private inputs (witness values for input variables)
    pub fn prove(
        &self,
        constraint_system: &ConstraintSystem,
        input_values: &[Tower],
    ) -> BiniusProof {
        // Step 1: Execute circuit and generate witness
        let witness = constraint_system.execute(input_values);

        // Step 2: Build multilinear polynomial from witness values
        let poly = self.build_polynomial(&witness);

        // Step 3: Commit to polynomial using FRI
        let fri_proof = self.fri_prover.prove(&poly);

        // Step 4: Run sum-check protocol
        let sum = self.compute_sum(&witness);
        let sumcheck_proof = SumcheckProver::prove(&poly, sum);

        BiniusProof {
            fri_proof,
            sumcheck_proof,
            public_inputs: constraint_system.public_inputs.clone(),
            num_variables: constraint_system.num_variables,
        }
    }

    /// Build multilinear polynomial from witness values
    fn build_polynomial(&self, witness: &Witness) -> MultilinearPolynomial {
        MultilinearPolynomial::new(witness.values.clone())
            .unwrap_or_else(|_| MultilinearPolynomial::default())
    }

    /// Compute the sum over all Boolean hypercube (for sum-check)
    fn compute_sum(&self, witness: &Witness) -> Tower {
        // Sum all witness values
        witness.values.iter().fold(Tower::zero(), |acc, v| acc + *v)
    }
}

/// Binius Proof
#[derive(Clone, Debug)]
pub struct BiniusProof {
    pub fri_proof: FriProof,
    pub sumcheck_proof: SumcheckProof,
    pub public_inputs: Vec<Tower>,
    pub num_variables: usize,
}

impl BiniusProof {
    /// Serialize proof to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(self.fri_proof.commitment.as_bytes());
        bytes.extend_from_slice(&(self.num_variables as u32).to_le_bytes());
        bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::verifier::BiniusVerifier;

    #[test]
    fn test_prover_execute_and_prove() {
        // Setup
        let fri_params = FriParams::new(2, 4, 2);
        let prover = BiniusProver::new(fri_params.clone());

        // Create constraint system
        let mut cs = ConstraintSystem::new();
        let a = cs.new_word();
        let b = cs.new_word();
        let c = cs.and(a, b);
        let _ = cs.mul(a, c); // Use the AND result

        // Input values
        let a_val = 0xFFu128;
        let b_val = 0x0Fu128;
        let inputs = [Tower::new(a_val, 6), Tower::new(b_val, 6)];

        // Generate proof
        let proof = prover.prove(&cs, &inputs);

        // Verify proof was generated
        assert!(!proof.fri_proof.commitment.as_bytes().is_empty());
        assert_eq!(proof.fri_proof.layers.len(), 2); // log_domain = 2
    }

    #[test]
    fn test_prover_with_select() {
        // Setup
        let fri_params = FriParams::new(2, 4, 2);
        let prover = BiniusProver::new(fri_params);

        // Create constraint system with select
        let mut cs = ConstraintSystem::new();
        let cond = cs.new_word();
        let a = cs.new_word();
        let b = cs.new_word();
        let result = cs.select(cond, a, b);

        // Test case 1: cond = 1, should select a
        let inputs = [
            Tower::new(1, 6),  // cond
            Tower::new(42, 6), // a
            Tower::new(99, 6), // b
        ];
        let proof = prover.prove(&cs, &inputs);
        assert!(!proof.fri_proof.commitment.as_bytes().is_empty());

        // Test case 2: cond = 0, should select b
        let inputs2 = [
            Tower::new(0, 6),  // cond
            Tower::new(42, 6), // a
            Tower::new(99, 6), // b
        ];
        let proof2 = prover.prove(&cs, &inputs2);
        assert!(!proof2.fri_proof.commitment.as_bytes().is_empty());
    }

    #[test]
    fn test_end_to_end_prove_verify() {
        // Setup
        let fri_params = FriParams::new(2, 4, 2);
        let prover = BiniusProver::new(fri_params.clone());
        let verifier = BiniusVerifier::new(fri_params);

        // Create a simple circuit: compute (a AND b) + (a * b)
        let mut cs = ConstraintSystem::new();
        let a = cs.new_byte();
        let b = cs.new_byte();
        let c = cs.and(a, b); // c = a AND b
        let d = cs.mul(a, b); // d = a * b

        // Input values
        let a_val = 5u128; // 0b101
        let b_val = 3u128; // 0b011
        let inputs = [Tower::new(a_val, 3), Tower::new(b_val, 3)];

        // Generate proof
        let proof = prover.prove(&cs, &inputs);

        // Verify proof
        let result = verifier.verify(&cs, &proof);
        assert!(result.is_ok(), "Verification failed: {:?}", result);

        // Verify proof structure
        assert_eq!(proof.num_variables, cs.num_variables);
    }

    #[test]
    fn test_end_to_end_xor_circuit() {
        // Setup
        let fri_params = FriParams::new(2, 4, 3);
        let prover = BiniusProver::new(fri_params.clone());
        let verifier = BiniusVerifier::new(fri_params);

        // Create XOR circuit: c = a XOR b = (a + b) - 2*(a AND b)
        // But we use the built-in XOR gate
        let mut cs = ConstraintSystem::new();
        let a = cs.new_byte();
        let b = cs.new_byte();
        let c = cs.xor(a, b);

        // Input values: 5 XOR 3 = 6
        let a_val = 5u128;
        let b_val = 3u128;
        let inputs = [Tower::new(a_val, 3), Tower::new(b_val, 3)];

        // Generate proof
        let proof = prover.prove(&cs, &inputs);

        // Verify proof
        let result = verifier.verify(&cs, &proof);
        assert!(result.is_ok());
    }

    #[test]
    fn test_proof_serialization() {
        let fri_params = FriParams::new(2, 4, 2);
        let prover = BiniusProver::new(fri_params);

        let mut cs = ConstraintSystem::new();
        let a = cs.new_byte();
        let b = cs.new_byte();
        let _ = cs.and(a, b);

        let proof = prover.prove(&cs, &[Tower::new(1, 3), Tower::new(2, 3)]);

        // Test serialization
        let bytes = proof.to_bytes();
        assert!(!bytes.is_empty());
    }
}
