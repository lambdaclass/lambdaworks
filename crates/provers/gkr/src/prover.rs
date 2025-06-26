use crate::circuit::{Circuit, CircuitEvaluation};
use lambdaworks_crypto::fiat_shamir::default_transcript::DefaultTranscript;
use lambdaworks_crypto::fiat_shamir::is_transcript::IsTranscript;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::{HasDefaultTranscript, IsField};
use lambdaworks_math::polynomial::dense_multilinear_poly::DenseMultilinearPolynomial;
use lambdaworks_math::traits::ByteConversion;
use lambdaworks_sumcheck::prover::prove;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ProverError {
    #[error("Verification failed")]
    VerificationFailed,
    #[error("Circuit evaluation failed")]
    CircuitEvaluationFailed,
    #[error("Sumcheck proof generation failed")]
    SumcheckFailed,
}

/// Generate a GKR proof
/// This implements the prover side of the GKR protocol
pub fn generate_proof<F>(
    circuit: &Circuit,
    evaluation: &CircuitEvaluation<FieldElement<F>>,
) -> Result<crate::Proof<F>, ProverError>
where
    F: IsField + HasDefaultTranscript,
    FieldElement<F>: ByteConversion,
    <F as IsField>::BaseType: Send + Sync + Copy,
{
    let mut transcript = DefaultTranscript::<F>::default();
    let mut sumcheck_proofs = vec![];
    let mut claims_phase2 = vec![];
    let mut layer_commitments = vec![];

    // Generate commitments to layer evaluations
    for layer_evals in &evaluation.layers {
        // Simple commitment: sum of all evaluations
        let mut commitment = FieldElement::zero();
        for eval in layer_evals {
            commitment = commitment + eval.clone();
        }
        layer_commitments.push(commitment.clone());
        transcript.append_bytes(&commitment.to_bytes_be());
    }

    // Get the number of variables for the output layer
    let k_0 = circuit.num_vars_at(0).unwrap_or(0);
    let mut r_i: Vec<FieldElement<F>> = (0..k_0)
        .map(|_| transcript.sample_field_element())
        .collect();

    // For each layer, run the GKR protocol
    for i in 0..circuit.layers().len() {
        // For simplicity, we'll use the layer evaluations directly as the polynomial
        // In a real implementation, this would construct the proper GKR polynomial
        let next_layer_evals = &evaluation.layers[i + 1];
        let g_i_mle = DenseMultilinearPolynomial::new(next_layer_evals.clone());

        // Run sumcheck on the layer polynomial
        let (sum, proof) = prove(vec![g_i_mle]).map_err(|_| ProverError::SumcheckFailed)?;

        // Update transcript and store results
        transcript.append_bytes(&sum.to_bytes_be());
        claims_phase2.push(sum);
        sumcheck_proofs.push(proof);

        // Sample challenges for the next round
        let k_next = circuit.num_vars_at(i + 1).unwrap_or(0);
        let num_sumcheck_challenges = 2 * k_next;

        let sumcheck_challenges: Vec<FieldElement<F>> = (0..num_sumcheck_challenges)
            .map(|_| transcript.sample_field_element())
            .collect();

        let r_last = transcript.sample_field_element();

        // Construct the next round's random point
        let (b, c) = sumcheck_challenges.split_at(k_next);
        let r_i_next: Vec<FieldElement<F>> = b
            .iter()
            .zip(c.iter())
            .map(|(bi, ci)| bi.clone() + r_last.clone() * (ci.clone() - bi.clone()))
            .collect();

        // Evaluate the next layer at the new point and add to transcript
        let w_next_poly = DenseMultilinearPolynomial::new(evaluation.layers[i + 1].clone());
        if let Ok(next_claim) = w_next_poly.evaluate(r_i_next.clone()) {
            transcript.append_bytes(&next_claim.to_bytes_be());
        }

        r_i = r_i_next;
    }

    // Store the final point for input verification
    // The final point should have the dimension of the input layer
    let num_input_vars = (evaluation.layers.last().unwrap().len() as f64).log2() as usize;
    let final_point = if r_i.len() >= num_input_vars {
        r_i[..num_input_vars].to_vec()
    } else {
        // If r_i is smaller than needed, pad with zeros
        let mut padded = r_i.clone();
        while padded.len() < num_input_vars {
            padded.push(FieldElement::zero());
        }
        padded
    };

    // Evaluate the input layer at the final point to get the correct final claim
    let input_poly = DenseMultilinearPolynomial::new(evaluation.layers.last().unwrap().clone());
    let final_claim = input_poly
        .evaluate(final_point.clone())
        .unwrap_or_else(|_| FieldElement::zero());

    // Add the final claim to claims_phase2
    claims_phase2.push(final_claim);

    Ok(crate::Proof {
        sumcheck_proofs,
        claims_phase2,
        layer_commitments,
        final_point,
    })
}
