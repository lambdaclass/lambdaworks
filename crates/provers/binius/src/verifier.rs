//! Binius Verifier Implementation
//!
//! Verifies Binius proofs by:
//! 1. Replaying the Fiat-Shamir transcript to derive the evaluation point
//! 2. Checking sumcheck round polynomial consistency
//! 3. Verifying the FRI proof (committed polynomial is low-degree)
//! 4. Checking consistency between sumcheck and claimed evaluation

use crate::constraints::ConstraintSystem;
use crate::fri::{FriParams, FriVerifier};
use crate::prover::{BiniusProof, BiniusProofV2};
use crate::sumcheck;
use lambdaworks_crypto::fiat_shamir::default_transcript::DefaultTranscript;
use lambdaworks_crypto::fiat_shamir::is_transcript::IsTranscript;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::binary::tower_field::BinaryTowerField128;

type FE = FieldElement<BinaryTowerField128>;

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

    /// Verify a Binius proof (new protocol).
    ///
    /// Checks:
    /// 1. Replays Fiat-Shamir transcript to derive evaluation point
    /// 2. Verifies sumcheck round polynomial consistency
    /// 3. Verifies FRI proof (committed polynomial is low-degree)
    /// 4. Checks consistency between sumcheck claimed sum and claimed evaluation
    pub fn verify_proof(&self, proof: &BiniusProofV2) -> Result<(), VerificationError> {
        // Step 1: Replay transcript to derive evaluation point
        let mut transcript = DefaultTranscript::<BinaryTowerField128>::new(b"binius_prove");
        transcript.append_bytes(proof.fri_proof.commitment.as_bytes());

        let expected_point: Vec<FE> = (0..proof.num_vars)
            .map(|_| transcript.sample_field_element())
            .collect();

        if expected_point != proof.evaluation_point {
            return Err(VerificationError::TranscriptMismatch);
        }

        // Step 2: Verify sumcheck round polynomial consistency
        let round_polys = &proof.sumcheck_proof.round_polynomials;

        if round_polys.is_empty() {
            return Err(VerificationError::SumcheckFailed);
        }

        // The claimed_sum from the product sumcheck should equal f(r),
        // because sum_{x in {0,1}^n} eq(x,r) * f(x) = f(r)
        if proof.sumcheck_proof.claimed_sum != proof.claimed_evaluation {
            return Err(VerificationError::EvaluationMismatch);
        }

        // Derive challenges using the same transcript protocol as the sumcheck crate
        let num_factors = 2; // product sumcheck: eq * f
        let challenges = sumcheck::derive_challenges_from_round_polys(
            round_polys,
            proof.num_vars,
            num_factors,
            &proof.sumcheck_proof.claimed_sum,
        );

        // Check g_1(0) + g_1(1) = claimed_sum
        let g1_sum = round_polys[0].evaluate(&FE::zero()) + round_polys[0].evaluate(&FE::one());
        if g1_sum != proof.sumcheck_proof.claimed_sum {
            return Err(VerificationError::SumcheckFailed);
        }

        // Check round consistency: g_{i+1}(0) + g_{i+1}(1) = g_i(r_i)
        for i in 0..round_polys.len().saturating_sub(1) {
            let expected = round_polys[i].evaluate(&challenges[i]);
            let actual_sum =
                round_polys[i + 1].evaluate(&FE::zero()) + round_polys[i + 1].evaluate(&FE::one());
            if expected != actual_sum {
                return Err(VerificationError::SumcheckFailed);
            }
        }

        // Step 3: Verify FRI proof
        self.fri_verifier
            .verify(&proof.fri_proof, &[])
            .map_err(|e| VerificationError::FriDetailedError(format!("{e:?}")))?;

        Ok(())
    }

    /// Verify a legacy proof (backward compatibility).
    pub fn verify(
        &self,
        constraint_system: &ConstraintSystem,
        proof: &BiniusProof,
    ) -> Result<bool, VerificationError> {
        if proof.public_inputs != constraint_system.public_inputs {
            return Err(VerificationError::PublicInputMismatch);
        }

        if proof.num_variables != constraint_system.num_variables {
            return Err(VerificationError::VariableCountMismatch);
        }

        // Verify FRI proof
        self.fri_verifier
            .verify(&proof.fri_proof, &[])
            .map_err(|e| VerificationError::FriDetailedError(format!("{e:?}")))?;

        Ok(true)
    }
}

/// Verification errors
#[derive(Debug)]
pub enum VerificationError {
    PublicInputMismatch,
    VariableCountMismatch,
    SumcheckFailed,
    FriFailed,
    FriDetailedError(String),
    EvaluationMismatch,
    TranscriptMismatch,
}
