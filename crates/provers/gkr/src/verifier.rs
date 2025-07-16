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
        // 1. Append the circuit data to the transcript.
        transcript.append_bytes(&crate::circuit_to_bytes(circuit));
        // 2. x public inputs
        for x in proof.input_values.clone() {
            transcript.append_bytes(&x.to_bytes_be());
        }
        // 3. y outputs (first layer of evaluation)
        for y in proof.output_values.clone() {
            transcript.append_bytes(&y.to_bytes_be());
        }

        let k_0 = circuit.num_vars_at(0).ok_or(VerifierError::CircuitError)?;
        let mut r_i: Vec<FieldElement<F>> = (0..k_0)
            .map(|_| transcript.sample_field_element())
            .collect();

        let output_poly_ext = DenseMultilinearPolynomial::new(proof.output_values.clone());
        let mut m_i = output_poly_ext
            .evaluate(r_i.clone())
            .map_err(|_e| VerifierError::MultilinearPolynomialEvaluationError)?;
        for (layer_idx, layer_proof) in proof.layer_proofs.iter().enumerate() {
            // Complete GKR sumcheck verification with final evaluation check
            let (sumcheck_verified, sumcheck_challenges) =
                gkr_sumcheck_verify(m_i.clone(), &layer_proof.sumcheck_proof, &mut transcript)?;

            if !sumcheck_verified {
                println!("Sumcheck verification failed at layer {}", layer_idx);
                return Ok(false);
            }

            // Final sumcheck verification
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

            if final_eval != expected_final_eval {
                println!("Final sumcheck verification failed at layer {}.", layer_idx);
                return Ok(false);
            }

            // Sample challenges for the next round using line function
            let k_i_plus_1 = circuit
                .num_vars_at(layer_idx + 1)
                .ok_or(VerifierError::CircuitError)?;

            // r* in the Lambda post
            let r_last = transcript.sample_field_element();

            // Construct the next round's random point using line function
            // This implements the line function l(x) = b + x * (c - b)
            let (b, c) = sumcheck_challenges.split_at(k_i_plus_1);
            r_i = crate::line(b, c, &r_last);
            m_i = layer_proof.poly_q.evaluate(&r_last);
        }

        // Final check using the inputs.
        let input_poly_ext = DenseMultilinearPolynomial::new(proof.input_values.clone());
        if m_i
            != input_poly_ext
                .evaluate(r_i)
                .map_err(|_| VerifierError::MultilinearPolynomialEvaluationError)?
        {
            return Ok(false);
        }

        Ok(true)
    }
}
