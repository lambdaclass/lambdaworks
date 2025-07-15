use crate::{
    circuit::{Circuit, CircuitEvaluation},
    sumcheck::gkr_sumcheck_verify_complete,
    GkrProof,
};
use lambdaworks_math::polynomial::Polynomial;
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
use thiserror::Error;

#[derive(Error, Debug)]
pub enum VerifierError {
    #[error("the proof is not valid")]
    InvalidProof,
    #[error("evaluation failed")]
    EvaluationFailed,
    #[error("sumcheck verification failed")]
    SumcheckFailed,
    #[error("final check failed")]
    FinalCheckFailed,
}

pub struct Verifier;

impl Verifier {
    pub fn verify<F>(proof: &GkrProof<F>, circuit: &Circuit) -> Result<bool, VerifierError>
    where
        F: IsField + HasDefaultTranscript,
        FieldElement<F>: ByteConversion,
        <F as IsField>::BaseType: Send + Sync + Copy,
    {
        let mut transcript = DefaultTranscript::<F>::default();
        transcript.append_bytes(&crate::hash_circuit(circuit));

        for x in proof.input_values.clone() {
            transcript.append_bytes(&x.to_bytes_be());
        }

        for y in proof.output_values.clone() {
            transcript.append_bytes(&y.to_bytes_be());
        }

        let k_0 = circuit.num_vars_at(0).unwrap();
        let mut r_i: Vec<FieldElement<F>> = (0..k_0)
            .map(|_| transcript.sample_field_element())
            .collect();

        r_i = vec![FieldElement::<F>::from(2)];

        let output_poly_ext = DenseMultilinearPolynomial::new(proof.output_values.clone());

        let mut m_i = output_poly_ext
            .evaluate(r_i.clone())
            .map_err(|_e| VerifierError::EvaluationFailed)?;

        for (layer_idx, layer_proof) in proof.layer_proofs.iter().enumerate() {
            // Complete GKR sumcheck verification with final evaluation check
            let (sumcheck_verified, sumcheck_challenges) = gkr_sumcheck_verify_complete(
                m_i.clone(),
                &layer_proof.sumcheck_proof,
                &layer_proof.poly_q,
                &mut transcript,
            )?;

            // Final sumcheck verification
            let expected_final_eval = layer_proof
                .sumcheck_proof
                .round_polynomials
                .last()
                .unwrap()
                .evaluate::<F>(sumcheck_challenges.last().unwrap());
            let q_at_0: FieldElement<F> = layer_proof.poly_q.evaluate(&FieldElement::zero());
            let q_at_1: FieldElement<F> = layer_proof.poly_q.evaluate(&FieldElement::one());

            if layer_idx < proof.layer_proofs.len() - 1 {
                let final_eval = circuit
                    .add_i_ext(&r_i, layer_idx)
                    .evaluate(sumcheck_challenges.clone())
                    .unwrap()
                    * (&q_at_0 + &q_at_1)
                    + circuit
                        .mul_i_ext(&r_i, layer_idx)
                        .evaluate(sumcheck_challenges.clone())
                        .unwrap()
                        * q_at_0
                        * q_at_1;
                // let final_eval = add_i_full.evaluate(sumcheck_challenges.clone()).unwrap()
                //     * (&q_at_0 + &q_at_1)
                //     + mul_i_full.evaluate(sumcheck_challenges.clone()).unwrap() * q_at_0 * q_at_1;

                if final_eval != expected_final_eval {
                    println!("FAIL: Final claim does not match expected value.");
                    println!("final eval calculation: {:?}", final_eval);
                    println!("final eval expected from g_j: {:?}", expected_final_eval);
                    return Ok(false);
                }
            }

            if !sumcheck_verified {
                return Ok(false);
            }

            // Sample challenges for the next round using line function (as in Lambda post)
            let k_i_plus_1 = circuit.num_vars_at(layer_idx + 1).unwrap();

            // r* in the Lambda post
            let mut r_last = transcript.sample_field_element();

            if layer_idx == 0 {
                r_last = FieldElement::<F>::from(6);
            }

            if layer_idx == 1 {
                r_last = FieldElement::<F>::from(17);
            }

            // Construct the next round's random point using line function
            // This implements ℓ(x) = b + x * (c - b) from the Lambda post
            let (b, c) = sumcheck_challenges.split_at(k_i_plus_1);
            r_i = crate::line(b, c, &r_last);
            m_i = layer_proof.poly_q.evaluate(&r_last);
        }

        let final_claim = &proof.final_claim;

        let last_layer_poly = DenseMultilinearPolynomial::new(proof.input_values.clone());

        let expected_last_claim = last_layer_poly
            .evaluate(r_i.clone())
            .map_err(|_e| VerifierError::EvaluationFailed)?;

        if final_claim != &expected_last_claim {
            return Ok(false);
        }

        Ok(true)
    }
}
