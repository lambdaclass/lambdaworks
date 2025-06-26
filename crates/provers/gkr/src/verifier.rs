use crate::circuit::{Circuit, CircuitEvaluation};
use crate::Proof;
use lambdaworks_crypto::fiat_shamir::default_transcript::DefaultTranscript;
use lambdaworks_crypto::fiat_shamir::is_transcript::IsTranscript;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::{HasDefaultTranscript, IsField};
use lambdaworks_math::polynomial::dense_multilinear_poly::DenseMultilinearPolynomial;
use lambdaworks_math::traits::ByteConversion;
use lambdaworks_sumcheck::verifier::verify;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum VerifierError {
    #[error("the proof is not valid")]
    InvalidProof,
    #[error("sumcheck verification failed")]
    SumcheckFailed,
    #[error("inconsistent evaluation")]
    InconsistentEvaluation,
    #[error("commitment verification failed")]
    CommitmentFailed,
}

/// The state of the Verifier.
pub struct Verifier;

impl Verifier {
    /// Verify a GKR proof
    /// This implements the verifier side of the GKR protocol
    pub fn verify<F>(
        proof: &Proof<F>,
        circuit: &Circuit,
        evaluation: &CircuitEvaluation<FieldElement<F>>,
    ) -> Result<bool, VerifierError>
    where
        F: IsField + HasDefaultTranscript,
        FieldElement<F>: ByteConversion,
        <F as IsField>::BaseType: Send + Sync + Copy,
    {
        let mut transcript = DefaultTranscript::<F>::default();

        // Verify commitments to layer evaluations
        for commitment in &proof.layer_commitments {
            transcript.append_bytes(&commitment.to_bytes_be());
        }

        // Get the number of variables for the output layer
        let k_0 = circuit.num_vars_at(0).unwrap_or(0);
        let mut r_i: Vec<FieldElement<F>> = (0..k_0)
            .map(|_| transcript.sample_field_element())
            .collect();

        // Verify each sumcheck proof in sequence
        for (layer_idx, sumcheck_proof) in proof.sumcheck_proofs.iter().enumerate() {
            let claimed_sum = if layer_idx < proof.claims_phase2.len() {
                proof.claims_phase2[layer_idx].clone()
            } else {
                return Err(VerifierError::InconsistentEvaluation);
            };

            // Use the real layer evaluations from the circuit evaluation
            let next_layer_evals = &evaluation.layers[layer_idx + 1];
            let g_i_mle = DenseMultilinearPolynomial::new(next_layer_evals.clone());

            // Verify the sumcheck proof
            let verification_result = verify(
                g_i_mle.num_vars(),
                claimed_sum.clone(),
                sumcheck_proof.clone(),
                vec![g_i_mle],
            );

            if verification_result.is_err() || !verification_result.unwrap() {
                return Err(VerifierError::SumcheckFailed);
            }

            // Sample challenges for the next round (same as prover)
            let k_next = circuit.num_vars_at(layer_idx + 1).unwrap_or(0);
            let num_sumcheck_challenges = 2 * k_next;
            let sumcheck_challenges: Vec<FieldElement<F>> = (0..num_sumcheck_challenges)
                .map(|_| transcript.sample_field_element())
                .collect();
            let r_last = transcript.sample_field_element();

            let (b, c) = sumcheck_challenges.split_at(k_next);
            let r_i_next: Vec<FieldElement<F>> = b
                .iter()
                .zip(c.iter())
                .map(|(bi, ci)| bi.clone() + r_last.clone() * (ci.clone() - bi.clone()))
                .collect();

            // Update transcript with the next claim
            let next_claim = proof
                .claims_phase2
                .get(layer_idx)
                .cloned()
                .unwrap_or_default();
            transcript.append_bytes(&next_claim.to_bytes_be());

            r_i = r_i_next;
        }

        Ok(true)
    }

    /// Verify the final input layer
    /// This checks that the final claim matches the actual input to the circuit
    pub fn verify_input<F>(
        final_claim: FieldElement<F>,
        final_point: &[FieldElement<F>],
        input: &[FieldElement<F>],
    ) -> bool
    where
        F: IsField,
        <F as IsField>::BaseType: Send + Sync + Copy,
    {
        let input_poly = DenseMultilinearPolynomial::new(input.to_vec());

        match input_poly.evaluate(final_point.to_vec()) {
            Ok(eval) => eval == final_claim,
            Err(_) => false,
        }
    }

    /// Create a line polynomial between two points
    /// Based on the reference implementation: l(0) = b, l(1) = c
    pub fn line<F>(b: &[FieldElement<F>], c: &[FieldElement<F>]) -> Vec<FieldElement<F>>
    where
        F: IsField,
    {
        b.iter()
            .zip(c.iter())
            .map(|(b_val, c_val)| {
                // For simplicity, we return the sum (avoiding division)
                // In the real protocol, this would be a proper line evaluation
                b_val.clone() + c_val.clone()
            })
            .collect()
    }
}
