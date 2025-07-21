use crate::{circuit::Circuit, sumcheck::gkr_sumcheck_verify, GKRProof};
use lambdaworks_math::{
    field::{
        element::FieldElement,
        traits::{HasDefaultTranscript, IsField},
    },
    polynomial::dense_multilinear_poly::DenseMultilinearPolynomial,
    traits::ByteConversion,
};

use lambdaworks_crypto::fiat_shamir::{
    default_transcript::DefaultTranscript, is_transcript::IsTranscript,
};
#[derive(Debug)]
pub enum VerifierError {
    InvalidProof,
    InvalidDegree,
    MultilinearPolynomialEvaluationError,
    SumcheckError,
    CircuitError,
}

pub struct Verifier;

impl Verifier {
    pub fn verify<F>(proof: &GKRProof<F>, circuit: &Circuit) -> Result<bool, VerifierError>
    where
        F: IsField + HasDefaultTranscript,
        FieldElement<F>: ByteConversion,
        <F as IsField>::BaseType: Send + Sync + Copy,
    {
        // Fiat-Shamir heuristic:
        // Both parties need to to append to the transcript the circuit, the inputs and the outputs.
        // See https://eprint.iacr.org/2025/118.pdf, Sections 2.1 and 2.2
        let mut transcript = DefaultTranscript::<F>::default();
        // Append the circuit data to the transcript.
        transcript.append_bytes(&crate::circuit_to_bytes(circuit));
        // Append public inputs x.
        for x in &proof.input_values {
            transcript.append_bytes(&x.to_bytes_be());
        }
        // Append public outputs y (first layer of evaluation).
        for y in &proof.output_values {
            transcript.append_bytes(&y.to_bytes_be());
        }

        let k_0 = circuit.num_vars_at(0).ok_or(VerifierError::CircuitError)?;
        let mut r_i: Vec<FieldElement<F>> = (0..k_0)
            .map(|_| transcript.sample_field_element())
            .collect();

        // Calculate the initial claimed sum `m_0 = W_0(r_0)`.
        let output_poly_ext = DenseMultilinearPolynomial::new(proof.output_values.clone());
        let mut claimed_sum = output_poly_ext
            .evaluate(r_i.clone())
            .map_err(|_e| VerifierError::MultilinearPolynomialEvaluationError)?;

        // For each layer, verify the sumcheck proof and calculate the next layer's challenges and claimed sum.
        for (layer_idx, layer_proof) in proof.layer_proofs.iter().enumerate() {
            // Sumcheck verification.
            let (sumcheck_verified, sumcheck_challenges) = gkr_sumcheck_verify(
                claimed_sum.clone(),
                &layer_proof.sumcheck_proof,
                &mut transcript,
            )?;

            if !sumcheck_verified {
                println!("Sumcheck verification failed at layer {}", layer_idx);
                return Ok(false);
            }

            // Final sumcheck verification.
            // The verifier must check the last claim using the polynommial q sent by the prover.
            let last_poly = layer_proof
                .sumcheck_proof
                .round_polynomials
                .last()
                .ok_or(VerifierError::InvalidProof)?;
            let last_challenge = sumcheck_challenges
                .last()
                .ok_or(VerifierError::SumcheckError)?;
            let expected_final_eval = last_poly.evaluate::<F>(last_challenge);

            let q_at_0: FieldElement<F> = layer_proof.poly_q.evaluate(&FieldElement::zero());
            let q_at_1: FieldElement<F> = layer_proof.poly_q.evaluate(&FieldElement::one());

            let add_eval = circuit
                .add_i_ext(&r_i, layer_idx)
                .evaluate(sumcheck_challenges.clone())
                .map_err(|_| VerifierError::MultilinearPolynomialEvaluationError)?;
            let mul_eval = circuit
                .mul_i_ext(&r_i, layer_idx)
                .evaluate(sumcheck_challenges.clone())
                .map_err(|_| VerifierError::MultilinearPolynomialEvaluationError)?;

            let final_eval = add_eval * (&q_at_0 + &q_at_1) + mul_eval * q_at_0 * q_at_1;

            // Final claim verification.
            if final_eval != expected_final_eval {
                println!("Final sumcheck verification failed at layer {}.", layer_idx);
                return Ok(false);
            }

            // Sample challenges for the next round using line function.
            let num_vars_next = circuit
                .num_vars_at(layer_idx + 1)
                .ok_or(VerifierError::CircuitError)?;

            // r* in our blog post <https://blog.lambdaclass.com/gkr-protocol-a-step-by-step-example/>
            let r_new = transcript.sample_field_element();

            // Construct the next round's random point using line function that goes from `b` to `c`.
            let (b, c) = sumcheck_challenges.split_at(num_vars_next);
            r_i = crate::line(b, c, &r_new);
            // Set the next layer's claimed sum.
            claimed_sum = layer_proof.poly_q.evaluate(&r_new);
        }

        // At the last layer the verifier checks the last claim using the input multilinear polynomial extension W.
        let input_poly_ext = DenseMultilinearPolynomial::new(proof.input_values.clone());
        if claimed_sum
            != input_poly_ext
                .evaluate(r_i)
                .map_err(|_| VerifierError::MultilinearPolynomialEvaluationError)?
        {
            return Ok(false);
        }

        Ok(true)
    }
}
