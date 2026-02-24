//! Binius Prover Implementation
//!
//! Composes sumcheck + FRI into a complete Binius proof system with proper
//! Fiat-Shamir throughout.
//!
//! ## Protocol Flow
//!
//! 1. Take witness as multilinear polynomial f over GF(2^128)
//! 2. RS-encode f via additive NTT
//! 3. Merkle-commit, append root to transcript
//! 4. Derive random evaluation point r from transcript
//! 5. Compute claimed evaluation f(r)
//! 6. Prove evaluation via sumcheck: sum_{x in {0,1}^n} eq(x,r) * f(x) = f(r)
//! 7. FRI proves committed polynomial is low-degree

use crate::constraints::{ConstraintSystem, Witness};
use crate::fields::tower::Tower;
use crate::fri::{FriParams, FriProof, FriProver, MerkleRoot};
use crate::ntt;
use crate::polynomial::MultilinearPolynomial;
use crate::sumcheck::{self, BiniusSumcheckProof, SumcheckProof, SumcheckProver};
use lambdaworks_crypto::fiat_shamir::default_transcript::DefaultTranscript;
use lambdaworks_crypto::fiat_shamir::is_transcript::IsTranscript;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::binary::tower_field::BinaryTowerField128;
use lambdaworks_math::polynomial::dense_multilinear_poly::{
    eq_polynomial, DenseMultilinearPolynomial,
};

type FE = FieldElement<BinaryTowerField128>;

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

    /// Prove a polynomial evaluation identity using the Binius protocol.
    ///
    /// Protocol:
    /// 1. RS-encode witness polynomial and Merkle-commit
    /// 2. Derive evaluation point r from transcript
    /// 3. Compute f(r) as claimed evaluation
    /// 4. Build eq(x, r) and prove: sum_{x in {0,1}^n} eq(x,r) * f(x) = f(r)
    /// 5. FRI proves committed polynomial is low-degree
    pub fn prove_polynomial(
        &self,
        witness: &DenseMultilinearPolynomial<BinaryTowerField128>,
    ) -> Result<BiniusProofV2, ProverError> {
        let num_vars = witness.num_vars();
        let evals = witness.evals();

        // Step 1: RS-encode and commit
        let codeword = ntt::rs_encode(evals, self.fri_prover.params().log_blowup);
        let commitment = MerkleRoot::from_field_elements(&codeword);

        // Step 2: Derive evaluation point from transcript
        let mut transcript = DefaultTranscript::<BinaryTowerField128>::new(b"binius_prove");
        transcript.append_bytes(commitment.as_bytes());

        let evaluation_point: Vec<FE> = (0..num_vars)
            .map(|_| transcript.sample_field_element())
            .collect();

        // Step 3: Compute claimed evaluation f(r)
        let claimed_evaluation = witness
            .evaluate(evaluation_point.clone())
            .map_err(|_| ProverError::EvaluationFailed)?;

        // Step 4: Build eq(x, r) polynomial and run product sumcheck
        let eq_poly = eq_polynomial::<BinaryTowerField128>(&evaluation_point);

        let sumcheck_proof = sumcheck::prove_product_sumcheck(eq_poly, witness.clone())
            .map_err(|e| ProverError::SumcheckFailed(format!("{e:?}")))?;

        // Step 5: FRI proof
        let fri_proof = self.fri_prover.prove_fe(evals);

        Ok(BiniusProofV2 {
            fri_proof,
            sumcheck_proof,
            claimed_evaluation,
            evaluation_point,
            num_vars,
        })
    }

    // --- Legacy API for backward compatibility ---

    /// Generate proof for a computation (legacy API).
    pub fn prove(
        &self,
        constraint_system: &ConstraintSystem,
        input_values: &[Tower],
    ) -> BiniusProof {
        let witness = constraint_system.execute(input_values);
        let poly = self.build_polynomial(&witness);
        let fri_proof = self.fri_prover.prove(&poly);
        let sum = self.compute_sum(&witness);
        let sumcheck_proof = SumcheckProver::prove(&poly, sum);

        BiniusProof {
            fri_proof,
            sumcheck_proof,
            public_inputs: constraint_system.public_inputs.clone(),
            num_variables: constraint_system.num_variables,
        }
    }

    fn build_polynomial(&self, witness: &Witness) -> MultilinearPolynomial {
        MultilinearPolynomial::new(witness.values.clone())
            .unwrap_or_else(|_| MultilinearPolynomial::default())
    }

    fn compute_sum(&self, witness: &Witness) -> Tower {
        witness.values.iter().fold(Tower::zero(), |acc, v| acc + *v)
    }
}

/// Binius proof using the complete protocol (sumcheck + FRI + Fiat-Shamir).
#[derive(Clone, Debug)]
pub struct BiniusProofV2 {
    /// FRI proof (includes commitment to the RS codeword)
    pub fri_proof: FriProof,
    /// Sumcheck proof: sum_{x} eq(x,r) * f(x) = f(r)
    pub sumcheck_proof: BiniusSumcheckProof,
    /// The claimed evaluation f(r)
    pub claimed_evaluation: FE,
    /// The random evaluation point r derived via Fiat-Shamir
    pub evaluation_point: Vec<FE>,
    /// Number of variables in the polynomial
    pub num_vars: usize,
}

/// Legacy proof type (backward compatibility).
#[derive(Clone, Debug)]
pub struct BiniusProof {
    pub fri_proof: FriProof,
    pub sumcheck_proof: SumcheckProof,
    pub public_inputs: Vec<Tower>,
    pub num_variables: usize,
}

impl BiniusProof {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(self.fri_proof.commitment.as_bytes());
        bytes.extend_from_slice(&(self.num_variables as u32).to_le_bytes());
        bytes
    }
}

/// Prover errors
#[derive(Debug)]
pub enum ProverError {
    EvaluationFailed,
    SumcheckFailed(String),
    FriFailed(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::verifier::BiniusVerifier;

    #[test]
    fn test_prove_polynomial_honest() {
        let fri_params = FriParams {
            log_message_size: 2,
            log_blowup: 1,
            num_queries: 2,
        };
        let prover = BiniusProver::new(fri_params.clone());

        // Create a simple 2-variable polynomial
        let evals: Vec<FE> = vec![
            FE::new(1u128),
            FE::new(2u128),
            FE::new(3u128),
            FE::new(4u128),
        ];
        let witness = DenseMultilinearPolynomial::<BinaryTowerField128>::new(evals);

        let proof = prover.prove_polynomial(&witness);
        assert!(proof.is_ok(), "Prover failed: {:?}", proof.err());

        let proof = proof.unwrap();
        assert_eq!(proof.num_vars, 2);
        assert_eq!(proof.evaluation_point.len(), 2);
    }

    #[test]
    fn test_prove_verify_honest_accepted() {
        let fri_params = FriParams {
            log_message_size: 2,
            log_blowup: 1,
            num_queries: 2,
        };
        let prover = BiniusProver::new(fri_params.clone());
        let verifier = BiniusVerifier::new(fri_params);

        let evals: Vec<FE> = vec![
            FE::new(5u128),
            FE::new(3u128),
            FE::new(7u128),
            FE::new(11u128),
        ];
        let witness = DenseMultilinearPolynomial::<BinaryTowerField128>::new(evals);

        let proof = prover.prove_polynomial(&witness).unwrap();
        let result = verifier.verify_proof(&proof);
        assert!(result.is_ok(), "Verification failed: {:?}", result.err());
    }

    #[test]
    fn test_tampered_claimed_evaluation_rejected() {
        let fri_params = FriParams {
            log_message_size: 2,
            log_blowup: 1,
            num_queries: 2,
        };
        let prover = BiniusProver::new(fri_params.clone());
        let verifier = BiniusVerifier::new(fri_params);

        let evals: Vec<FE> = vec![
            FE::new(1u128),
            FE::new(2u128),
            FE::new(3u128),
            FE::new(4u128),
        ];
        let witness = DenseMultilinearPolynomial::<BinaryTowerField128>::new(evals);

        let mut proof = prover.prove_polynomial(&witness).unwrap();
        proof.claimed_evaluation = FE::new(999u128); // tamper

        let result = verifier.verify_proof(&proof);
        assert!(result.is_err(), "Tampered proof should be rejected");
    }

    #[test]
    fn test_different_witnesses_different_proofs() {
        let fri_params = FriParams {
            log_message_size: 2,
            log_blowup: 1,
            num_queries: 2,
        };
        let prover = BiniusProver::new(fri_params);

        let w1 = DenseMultilinearPolynomial::<BinaryTowerField128>::new(vec![
            FE::new(1u128),
            FE::new(2u128),
            FE::new(3u128),
            FE::new(4u128),
        ]);
        let w2 = DenseMultilinearPolynomial::<BinaryTowerField128>::new(vec![
            FE::new(10u128),
            FE::new(20u128),
            FE::new(30u128),
            FE::new(40u128),
        ]);

        let p1 = prover.prove_polynomial(&w1).unwrap();
        let p2 = prover.prove_polynomial(&w2).unwrap();

        assert_ne!(
            p1.fri_proof.commitment.as_bytes(),
            p2.fri_proof.commitment.as_bytes()
        );
        assert_ne!(p1.claimed_evaluation, p2.claimed_evaluation);
    }

    #[test]
    fn test_eq_polynomial_identity() {
        // eq(x, r) should satisfy: sum_{x in {0,1}^n} eq(x, r) * f(x) = f(r)
        let r = vec![FE::new(7u128), FE::new(13u128)];
        let f_evals: Vec<FE> = vec![
            FE::new(1u128),
            FE::new(2u128),
            FE::new(3u128),
            FE::new(4u128),
        ];

        let eq_poly = eq_polynomial::<BinaryTowerField128>(&r);
        let eq_evals = eq_poly.evals();

        // Compute sum_{x} eq(x,r) * f(x)
        let sum: FE = eq_evals
            .iter()
            .zip(f_evals.iter())
            .map(|(e, f)| *e * *f)
            .fold(FE::zero(), |acc, v| acc + v);

        // Compute f(r) directly
        let f = DenseMultilinearPolynomial::<BinaryTowerField128>::new(f_evals);
        let f_r = f.evaluate(r).unwrap();

        assert_eq!(sum, f_r, "eq identity must hold");
    }

    // --- Legacy tests (backward compatibility) ---

    #[test]
    fn test_prover_execute_and_prove() {
        let fri_params = FriParams::new(2, 4, 2);
        let prover = BiniusProver::new(fri_params.clone());

        let mut cs = ConstraintSystem::new();
        let a = cs.new_word();
        let b = cs.new_word();
        let c = cs.and(a, b);
        let _ = cs.mul(a, c);

        let a_val = 0xFFu128;
        let b_val = 0x0Fu128;
        let inputs = [Tower::new(a_val, 6), Tower::new(b_val, 6)];

        let proof = prover.prove(&cs, &inputs);
        assert!(!proof.fri_proof.commitment.as_bytes().is_empty());
    }

    #[test]
    fn test_prover_with_select() {
        let fri_params = FriParams::new(2, 4, 2);
        let prover = BiniusProver::new(fri_params);

        let mut cs = ConstraintSystem::new();
        let cond = cs.new_word();
        let a = cs.new_word();
        let b = cs.new_word();
        let _result = cs.select(cond, a, b);

        let inputs = [Tower::new(1, 6), Tower::new(42, 6), Tower::new(99, 6)];
        let proof = prover.prove(&cs, &inputs);
        assert!(!proof.fri_proof.commitment.as_bytes().is_empty());

        let inputs2 = [Tower::new(0, 6), Tower::new(42, 6), Tower::new(99, 6)];
        let proof2 = prover.prove(&cs, &inputs2);
        assert!(!proof2.fri_proof.commitment.as_bytes().is_empty());
    }

    #[test]
    fn test_end_to_end_prove_verify() {
        let fri_params = FriParams::new(2, 4, 2);
        let prover = BiniusProver::new(fri_params.clone());
        let verifier = BiniusVerifier::new(fri_params);

        let mut cs = ConstraintSystem::new();
        let a = cs.new_byte();
        let b = cs.new_byte();
        let _c = cs.and(a, b);
        let _d = cs.mul(a, b);

        let a_val = 5u128;
        let b_val = 3u128;
        let inputs = [Tower::new(a_val, 3), Tower::new(b_val, 3)];

        let proof = prover.prove(&cs, &inputs);
        let result = verifier.verify(&cs, &proof);
        assert!(result.is_ok(), "Verification failed: {:?}", result);
        assert_eq!(proof.num_variables, cs.num_variables);
    }

    #[test]
    fn test_end_to_end_xor_circuit() {
        let fri_params = FriParams::new(2, 4, 3);
        let prover = BiniusProver::new(fri_params.clone());
        let verifier = BiniusVerifier::new(fri_params);

        let mut cs = ConstraintSystem::new();
        let a = cs.new_byte();
        let b = cs.new_byte();
        let _c = cs.xor(a, b);

        let a_val = 5u128;
        let b_val = 3u128;
        let inputs = [Tower::new(a_val, 3), Tower::new(b_val, 3)];

        let proof = prover.prove(&cs, &inputs);
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
        let bytes = proof.to_bytes();
        assert!(!bytes.is_empty());
    }

    // --- E2E tests using the V2 protocol (prove_polynomial + verify_proof) ---

    #[test]
    fn test_e2e_and_circuit_v2() {
        let mut cs = ConstraintSystem::new();
        let a = cs.new_byte();
        let b = cs.new_byte();
        let _c = cs.and(a, b);

        let witness = cs.execute(&[Tower::new(0xFF, 3), Tower::new(0x0F, 3)]);
        let poly = cs.witness_to_dense_poly(&witness);

        let fri_params = FriParams {
            log_message_size: poly.num_vars(),
            log_blowup: 1,
            num_queries: 2,
        };
        let prover = BiniusProver::new(fri_params.clone());
        let verifier = BiniusVerifier::new(fri_params);

        let proof = prover.prove_polynomial(&poly).unwrap();
        let result = verifier.verify_proof(&proof);
        assert!(
            result.is_ok(),
            "AND circuit V2 verification failed: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_e2e_xor_circuit_v2() {
        let mut cs = ConstraintSystem::new();
        let a = cs.new_byte();
        let b = cs.new_byte();
        let _c = cs.xor(a, b);

        let witness = cs.execute(&[Tower::new(5, 3), Tower::new(3, 3)]);
        let poly = cs.witness_to_dense_poly(&witness);

        let fri_params = FriParams {
            log_message_size: poly.num_vars(),
            log_blowup: 1,
            num_queries: 2,
        };
        let prover = BiniusProver::new(fri_params.clone());
        let verifier = BiniusVerifier::new(fri_params);

        let proof = prover.prove_polynomial(&poly).unwrap();
        let result = verifier.verify_proof(&proof);
        assert!(
            result.is_ok(),
            "XOR circuit V2 verification failed: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_e2e_mul_circuit_v2() {
        let mut cs = ConstraintSystem::new();
        let a = cs.new_byte();
        let b = cs.new_byte();
        let _c = cs.mul(a, b);

        let witness = cs.execute(&[Tower::new(5, 3), Tower::new(3, 3)]);
        let poly = cs.witness_to_dense_poly(&witness);

        let fri_params = FriParams {
            log_message_size: poly.num_vars(),
            log_blowup: 1,
            num_queries: 2,
        };
        let prover = BiniusProver::new(fri_params.clone());
        let verifier = BiniusVerifier::new(fri_params);

        let proof = prover.prove_polynomial(&poly).unwrap();
        let result = verifier.verify_proof(&proof);
        assert!(
            result.is_ok(),
            "MUL circuit V2 verification failed: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_e2e_select_circuit_v2() {
        let mut cs = ConstraintSystem::new();
        let cond = cs.new_byte();
        let a = cs.new_byte();
        let b = cs.new_byte();
        let _result = cs.select(cond, a, b);

        // cond = 1 (true): should select a = 42
        let witness = cs.execute(&[Tower::new(1, 3), Tower::new(42, 3), Tower::new(99, 3)]);
        let poly = cs.witness_to_dense_poly(&witness);

        let fri_params = FriParams {
            log_message_size: poly.num_vars(),
            log_blowup: 1,
            num_queries: 2,
        };
        let prover = BiniusProver::new(fri_params.clone());
        let verifier = BiniusVerifier::new(fri_params);

        let proof = prover.prove_polynomial(&poly).unwrap();
        let result = verifier.verify_proof(&proof);
        assert!(
            result.is_ok(),
            "SELECT circuit V2 verification failed: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_e2e_chained_gates_v2() {
        // Multi-gate circuit: d = mul(a, and(a, b))
        let mut cs = ConstraintSystem::new();
        let a = cs.new_byte();
        let b = cs.new_byte();
        let c = cs.and(a, b);
        let _d = cs.mul(a, c);

        let witness = cs.execute(&[Tower::new(0xFF, 3), Tower::new(0x0F, 3)]);
        let poly = cs.witness_to_dense_poly(&witness);

        let fri_params = FriParams {
            log_message_size: poly.num_vars(),
            log_blowup: 1,
            num_queries: 2,
        };
        let prover = BiniusProver::new(fri_params.clone());
        let verifier = BiniusVerifier::new(fri_params);

        let proof = prover.prove_polynomial(&poly).unwrap();
        let result = verifier.verify_proof(&proof);
        assert!(
            result.is_ok(),
            "Chained gates V2 verification failed: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_e2e_tampered_proof_rejected_v2() {
        let mut cs = ConstraintSystem::new();
        let a = cs.new_byte();
        let b = cs.new_byte();
        let _c = cs.and(a, b);

        let witness = cs.execute(&[Tower::new(5, 3), Tower::new(3, 3)]);
        let poly = cs.witness_to_dense_poly(&witness);

        let fri_params = FriParams {
            log_message_size: poly.num_vars(),
            log_blowup: 1,
            num_queries: 2,
        };
        let prover = BiniusProver::new(fri_params.clone());
        let verifier = BiniusVerifier::new(fri_params);

        let mut proof = prover.prove_polynomial(&poly).unwrap();
        proof.claimed_evaluation = FE::new(999u128); // tamper
        let result = verifier.verify_proof(&proof);
        assert!(result.is_err(), "Tampered proof should be rejected");
    }

    #[test]
    fn test_e2e_different_witnesses_v2() {
        let mut cs = ConstraintSystem::new();
        let a = cs.new_byte();
        let b = cs.new_byte();
        let _c = cs.and(a, b);

        let w1 = cs.execute(&[Tower::new(5, 3), Tower::new(3, 3)]);
        let w2 = cs.execute(&[Tower::new(10, 3), Tower::new(7, 3)]);
        let poly1 = cs.witness_to_dense_poly(&w1);
        let poly2 = cs.witness_to_dense_poly(&w2);

        let fri_params = FriParams {
            log_message_size: poly1.num_vars(),
            log_blowup: 1,
            num_queries: 2,
        };
        let prover = BiniusProver::new(fri_params.clone());
        let verifier = BiniusVerifier::new(fri_params);

        let proof1 = prover.prove_polynomial(&poly1).unwrap();
        let proof2 = prover.prove_polynomial(&poly2).unwrap();

        // Both should verify
        assert!(verifier.verify_proof(&proof1).is_ok());
        assert!(verifier.verify_proof(&proof2).is_ok());

        // But should be different proofs
        assert_ne!(proof1.claimed_evaluation, proof2.claimed_evaluation);
    }
}
