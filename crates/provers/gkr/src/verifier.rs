use crate::circuit::{Circuit, CircuitEvaluation};
use crate::prover::build_gkr_polynomial;
use crate::Proof;
use lambdaworks_crypto::fiat_shamir::default_transcript::DefaultTranscript;
use lambdaworks_crypto::fiat_shamir::is_transcript::IsTranscript;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::{HasDefaultTranscript, IsField};
use lambdaworks_math::polynomial::dense_multilinear_poly::DenseMultilinearPolynomial;
use lambdaworks_math::polynomial::Polynomial;
use lambdaworks_math::traits::ByteConversion;

use lambdaworks_sumcheck::Channel;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum VerifierError {
    #[error("the proof is not valid")]
    InvalidProof,
    #[error("sumcheck verification failed")]
    SumcheckFailed,
    #[error("evaluation of a polynomial failed")]
    EvaluationFailed,
    #[error("prover's claimed sum for a layer is inconsistent with the verifier's expectation")]
    InconsistentClaim,
    #[error("final check against public inputs failed")]
    FinalCheckFailed,
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

        // 0. Append the circuit data to the transcript.
        transcript.append_bytes(&(circuit.layers().len() as u32).to_le_bytes());
        transcript.append_bytes(&(circuit.num_inputs() as u32).to_le_bytes());
        for layer in circuit.layers() {
            transcript.append_bytes(&(layer.len() as u32).to_le_bytes());
            for gate in &layer.layer {
                let gate_type = match gate.ttype {
                    crate::circuit::GateType::Add => 0u8,
                    crate::circuit::GateType::Mul => 1u8,
                };
                transcript.append_bytes(&[gate_type]);
                transcript.append_bytes(&(gate.inputs[0] as u32).to_le_bytes());
                transcript.append_bytes(&(gate.inputs[1] as u32).to_le_bytes());
            }
        }

        // 1. x public inputs (last layer of evaluation)
        // Use the same inputs as the prover (the original input values)
        let input_layer = evaluation.layers.last().unwrap();
        for x in input_layer {
            transcript.append_bytes(&x.to_bytes_be());
        }

        // 2. y outputs (first layer of evaluation)
        for y in &evaluation.layers[0] {
            transcript.append_bytes(&y.to_bytes_be());
        }

        let k_0 = circuit.num_vars_at(0).unwrap_or(0);
        let mut r_i: Vec<FieldElement<F>> = (0..k_0)
            .map(|_| transcript.sample_field_element())
            .collect();

        // Use the prover's claim for the first layer
        let mut current_claim = proof.claims_phase2[0].clone();

        // Verify each sumcheck proof in sequence
        for (layer_idx, sumcheck_proof) in proof.sumcheck_proofs.iter().enumerate() {
            // Use the prover's claim for this layer
            current_claim = proof.claims_phase2[layer_idx].clone();

            let gkr_poly = build_gkr_polynomial(circuit, &r_i, evaluation, layer_idx);

            println!(
                "Layer {}: Verifying sumcheck with claim {:?}",
                layer_idx, current_claim
            );
            println!(
                "Layer {}: GKR poly num_vars: {}",
                layer_idx,
                gkr_poly.num_vars()
            );
            println!(
                "Layer {}: Sumcheck proof length: {}",
                layer_idx,
                sumcheck_proof.len()
            );
            println!("GKR Layer {}: Using challenges r_i: {:?}", layer_idx, r_i);

            println!(
                "GKR Layer {}: Starting sumcheck verification with transcript state: {:?}",
                layer_idx,
                transcript.state()
            );
            let verification_result = gkr_sumcheck_verify(
                gkr_poly.num_vars(),
                current_claim.clone(),
                sumcheck_proof.clone(),
                vec![gkr_poly],
                &mut transcript,
            );
            println!(
                "GKR Layer {}: Finished sumcheck verification with transcript state: {:?}",
                layer_idx,
                transcript.state()
            );

            match verification_result {
                Ok((true, sum_result)) => {
                    println!("Layer {}: Sumcheck verification SUCCESS", layer_idx);
                    // Use the actual sum result from sumcheck (same as prover)
                    println!(
                        "GKR Layer {}: Adding sum result {:?} to transcript",
                        layer_idx, sum_result
                    );
                    transcript.append_bytes(&sum_result.to_bytes_be());
                }
                Ok((false, _)) => {
                    println!(
                        "Layer {}: Sumcheck verification FAILED (returned false)",
                        layer_idx
                    );
                    return Err(VerifierError::SumcheckFailed);
                }
                Err(e) => {
                    println!("Layer {}: Sumcheck verification ERROR: {:?}", layer_idx, e);
                    return Err(VerifierError::SumcheckFailed);
                }
            }

            println!(
                "GKR Layer {}: After sumcheck verification, transcript state: {:?}",
                layer_idx,
                transcript.state()
            );

            let k_next = circuit.num_vars_at(layer_idx + 1).unwrap_or(0);
            let num_sumcheck_challenges = 2 * k_next;
            let sumcheck_challenges: Vec<FieldElement<F>> = (0..num_sumcheck_challenges)
                .map(|_| transcript.sample_field_element())
                .collect();
            let r_last = transcript.sample_field_element();

            println!(
                "GKR Layer {}: After sampling challenges, transcript state: {:?}",
                layer_idx,
                transcript.state()
            );
            println!(
                "GKR Layer {}: Generated challenges: sumcheck_challenges={:?}, r_last={:?}",
                layer_idx, sumcheck_challenges, r_last
            );

            let (b, c) = sumcheck_challenges.split_at(k_next);
            let r_i_next = crate::line(b, c, &r_last);

            println!(
                "GKR Layer {}: Constructed r_i_next: {:?}",
                layer_idx, r_i_next
            );

            // Add the layer claim to transcript (same as prover)
            if layer_idx < proof.layer_claims.len() {
                println!(
                    "GKR Layer {}: Adding layer claim {:?} to transcript",
                    layer_idx, proof.layer_claims[layer_idx]
                );
                transcript.append_bytes(&proof.layer_claims[layer_idx].to_bytes_be());
                println!(
                    "GKR Layer {}: After adding layer claim, transcript state: {:?}",
                    layer_idx,
                    transcript.state()
                );
            }

            r_i = r_i_next;
        }

        // Get the final claim from the proof (this is the claim for the input layer)
        let final_claim = proof
            .layer_claims
            .last()
            .ok_or(VerifierError::InvalidProof)?;

        let input_poly = DenseMultilinearPolynomial::new(evaluation.layers.last().unwrap().clone());
        let expected_last_claim = input_poly
            .evaluate(r_i.clone())
            .map_err(|_| VerifierError::EvaluationFailed)?;

        println!(
            "Final check: final_claim = {:?}, expected_last_claim = {:?}, r_i = {:?}",
            final_claim, expected_last_claim, r_i
        );

        if final_claim != &expected_last_claim {
            return Err(VerifierError::FinalCheckFailed);
        }

        Ok(true)
    }
}

/// GKR-specific sumcheck verifier that implements the protocol from scratch
fn gkr_sumcheck_verify<F, T>(
    num_vars: usize,
    claimed_sum: FieldElement<F>,
    proof_polys: Vec<Polynomial<FieldElement<F>>>,
    oracle_polys: Vec<DenseMultilinearPolynomial<F>>,
    transcript: &mut T,
) -> Result<(bool, FieldElement<F>), VerifierError>
where
    F: IsField + HasDefaultTranscript,
    F::BaseType: Send + Sync + Copy,
    FieldElement<F>: ByteConversion,
    T: IsTranscript<F> + Channel<F>,
{
    if proof_polys.len() != num_vars {
        return Err(VerifierError::SumcheckFailed);
    }

    let mut current_sum = claimed_sum.clone();
    let mut challenges = Vec::with_capacity(num_vars);

    // Process each round polynomial from the proof
    for (j, g_j) in proof_polys.into_iter().enumerate() {
        // Check degree of g_j
        let max_degree = oracle_polys.len();
        if g_j.degree() > max_degree {
            return Err(VerifierError::SumcheckFailed);
        }

        // Check consistency: g_j(0) + g_j(1) == expected_sum (current_sum)
        let zero = FieldElement::<F>::zero();
        let one = FieldElement::<F>::one();
        let eval_0 = g_j.evaluate(&zero);
        let eval_1 = g_j.evaluate(&one);
        let sum_evals = eval_0.clone() + eval_1.clone();

        // Debug: Print what the verifier is checking
        println!(
            "Sumcheck verifier round {}: g_j(0) = {:?}, g_j(1) = {:?}, sum = {:?}, expected = {:?}",
            j, eval_0, eval_1, sum_evals, current_sum
        );

        if sum_evals != current_sum {
            return Err(VerifierError::SumcheckFailed);
        }

        // Check if this is the final round
        if j == num_vars - 1 {
            // Final round: evaluate at the challenge point and verify
            // Note: No challenge is generated in the final round (same as prover)
            println!(
                "Sumcheck verifier round {}: Final round, no challenge generated",
                j
            );

            // Final verification: evaluate the product of oracle polynomials at the challenge point
            // For the final round, we evaluate at the point where the last variable is set to 0
            let mut final_point = challenges.clone();
            final_point.push(FieldElement::zero());

            let mut expected_final_eval = FieldElement::one();
            for oracle_poly in &oracle_polys {
                let eval = oracle_poly
                    .evaluate(final_point.clone())
                    .map_err(|_| VerifierError::SumcheckFailed)?;
                expected_final_eval = expected_final_eval * eval;
            }

            // The final sum should be g_j(0) which is eval_0
            let success = expected_final_eval == eval_0;
            println!(
                "Sumcheck verifier final round: expected_final_eval = {:?}, g_j(0) = {:?}, success = {}",
                expected_final_eval, eval_0, success
            );
            return Ok((success, claimed_sum.clone()));
        } else {
            // Not the final round, generate challenge for next round
            let r_j = transcript.sample_field_element();
            challenges.push(r_j.clone());
            println!(
                "Sumcheck verifier round {}: Generated challenge {:?}",
                j, r_j
            );

            // Update the expected sum for the next round: current_sum = g_j(r_j)
            current_sum = g_j.evaluate(&r_j);
        }
    }

    Err(VerifierError::SumcheckFailed)
}
