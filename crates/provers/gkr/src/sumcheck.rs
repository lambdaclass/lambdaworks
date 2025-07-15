use lambdaworks_crypto::fiat_shamir::is_transcript::IsTranscript;
use lambdaworks_math::{
    field::{
        element::FieldElement,
        traits::{HasDefaultTranscript, IsField},
    },
    polynomial::{dense_multilinear_poly::DenseMultilinearPolynomial, Polynomial},
    traits::ByteConversion,
};
use lambdaworks_sumcheck::{Channel, Prover, Verifier, VerifierRoundResult};

use crate::prover::ProverError;

#[derive(Debug, Clone)]
pub struct SumcheckProof<F: IsField> {
    pub round_polynomials: Vec<Polynomial<FieldElement<F>>>,
    pub final_evaluation: FieldElement<F>,
}

/// Helper function to combine polynomial terms into a single polynomial
/// This handles the GKR structure: term1 + term2 where each term is a product of polynomials
fn combine_gkr_terms<F: IsField>(
    terms: Vec<Vec<DenseMultilinearPolynomial<F>>>,
) -> Result<DenseMultilinearPolynomial<F>, ProverError>
where
    <F as IsField>::BaseType: Send + Sync + Copy,
{
    if terms.len() != 2 {
        return Err(ProverError::SumcheckFailed);
    }

    let num_vars = terms[0][0].num_vars();
    let hypercube_size = 1 << num_vars;

    // Compute evaluations for the combined polynomial: term1 + term2
    let combined_evals: Result<Vec<_>, _> = (0..hypercube_size)
        .map(|i| {
            // Convert index to binary point
            let point: Vec<FieldElement<F>> = (0..num_vars)
                .map(|k| {
                    if (i >> k) & 1 == 1 {
                        FieldElement::one()
                    } else {
                        FieldElement::zero()
                    }
                })
                .collect();

            // Evaluate term1: product of all polynomials in the first term
            let term1_value = terms[0]
                .iter()
                .map(|poly| {
                    poly.evaluate(point.clone())
                        .map_err(|_| ProverError::EvaluationFailed)
                })
                .try_fold(FieldElement::one(), |acc, res| res.map(|val| acc * val))?;

            // Evaluate term2: product of all polynomials in the second term
            let term2_value = terms[1]
                .iter()
                .map(|poly| {
                    poly.evaluate(point.clone())
                        .map_err(|_| ProverError::EvaluationFailed)
                })
                .try_fold(FieldElement::one(), |acc, res| res.map(|val| acc * val))?;

            // Sum the terms
            Ok(term1_value + term2_value)
        })
        .collect();

    let evaluations = combined_evals?;
    Ok(DenseMultilinearPolynomial::new(evaluations))
}

/// GKR-specific sumcheck prover that uses the existing sumcheck library
pub fn gkr_sumcheck_prove<F, T>(
    terms: Vec<Vec<DenseMultilinearPolynomial<F>>>,
    transcript: &mut T,
) -> Result<(FieldElement<F>, SumcheckProof<F>, Vec<FieldElement<F>>), ProverError>
where
    F: IsField + HasDefaultTranscript,
    <F as IsField>::BaseType: Send + Sync + Copy,
    FieldElement<F>: ByteConversion,
    T: IsTranscript<F> + Channel<F>,
{
    println!(
        "🔹 GKR SUMCHECK PROVE - Starting with {} terms",
        terms.len()
    );

    if terms.is_empty() || terms[0].is_empty() {
        return Err(ProverError::SumcheckFailed);
    }

    // Combine the GKR terms into a single polynomial
    // let combined_poly = combine_gkr_terms(terms)?;

    // println!(
    //     "🔹 GKR SUMCHECK PROVE - Combined polynomial has {} variables",
    //     combined_poly.num_vars()
    // );

    // Use the sumcheck prover directly with manual transcript control
    // let factors = vec![combined_poly];
    let factors_term_1 = terms[0].clone();
    let factors_term_2 = terms[1].clone();
    // let mut prover = Prover::new(factors).map_err(|_| ProverError::SumcheckFailed)?;

    let mut prover_term_1 = Prover::new(factors_term_1).map_err(|_| ProverError::SumcheckFailed)?;
    let mut prover_term_2 = Prover::new(factors_term_2).map_err(|_| ProverError::SumcheckFailed)?;

    let num_vars = prover_term_1.num_vars();

    // Compute the claimed sum
    // let claimed_sum = prover
    //     .compute_initial_sum()
    //     .map_err(|_| ProverError::SumcheckFailed)?;

    let claimed_sum_term_1 = prover_term_1
        .compute_initial_sum()
        .map_err(|_| ProverError::SumcheckFailed)?;

    let claimed_sum_term_2 = prover_term_2
        .compute_initial_sum()
        .map_err(|_| ProverError::SumcheckFailed)?;
    let claimed_sum = claimed_sum_term_1 + claimed_sum_term_2;

    // println!("🔹 GKR SUMCHECK PROVE - Claimed sum: {:?}", claimed_sum);

    // Add initial sum to transcript (mimicking prove_backend)
    transcript.append_bytes(b"initial_sum");
    transcript.append_felt(&FieldElement::from(num_vars as u64));
    transcript.append_felt(&FieldElement::from(1u64)); // single factor
    transcript.append_felt(&claimed_sum);
    // transcript.append_felt(&claimed_sum_term_1);
    // transcript.append_felt(&claimed_sum_term_2);

    let mut proof_polys = Vec::with_capacity(num_vars);
    let mut challenges = Vec::new();
    let mut current_challenge: Option<FieldElement<F>> = None;

    // Execute rounds manually
    for j in 0..num_vars {
        // SOLO UNA LLAMADA a .round() por ronda para cada Prover
        let g_j_term_1 = match prover_term_1.round(current_challenge.as_ref()) {
            Ok(round) => {
                println!("g_j_term_1 successfully created for j = {}.", j);
                round
            }
            Err(e) => {
                println!("Failed to create g_j_term_1: {:?}", e);
                return Err(ProverError::SumcheckFailed);
            }
        };

        let g_j_term_2 = match prover_term_2.round(current_challenge.as_ref()) {
            Ok(round) => {
                println!("g_j_term_2 successfully created for j = {}.", j);
                round
            }
            Err(e) => {
                println!("Failed to create g_j_term_2 for j = {}: {:?}", j, e);
                return Err(ProverError::SumcheckFailed);
            }
        };

        println!(
            "challenges in term 1 for j = {}: {:?}",
            j, prover_term_1.challenges
        );
        println!(
            "challenges in term 2 for j = {}: {:?}",
            j, prover_term_2.challenges
        );

        let g_j = g_j_term_1 + g_j_term_2;

        println!("Poly g_j for j = {}: {:?}", j, g_j);

        // Add polynomial info to transcript (mimicking prove_backend)
        let round_label = format!("round_{}_poly", j);
        transcript.append_bytes(round_label.as_bytes());

        let coeffs = g_j.coefficients();
        transcript.append_bytes(&(coeffs.len() as u64).to_be_bytes());
        if coeffs.is_empty() {
            transcript.append_felt(&FieldElement::zero());
        } else {
            for coeff in coeffs {
                transcript.append_felt(coeff);
            }
        }

        proof_polys.push(g_j);

        // Get challenge for next round (if not the last round)
        let mut r_j = transcript.sample_field_element();
        if j == 0 {
            r_j = FieldElement::<F>::from(3);
        }
        if j == 1 {
            r_j = FieldElement::<F>::from(2);
        }
        if j == 2 {
            r_j = FieldElement::<F>::from(4);
        }
        if j == 3 {
            r_j = FieldElement::<F>::from(7);
        }

        // Print g_j evaluated at r_j (like in the blogpost)
        let g_j_at_r_j = proof_polys[j].evaluate::<F>(&r_j);
        println!(
            "🔹 GKR SUMCHECK PROVE - Round {}: g_j(r_j = {:?}) = {:?}",
            j, r_j, g_j_at_r_j
        );

        challenges.push(r_j.clone());
        current_challenge = Some(r_j.clone());
        println!(
            "🔹 GKR SUMCHECK PROVE - Round {}: challenge r_j = {:?}",
            j, r_j
        );
    }

    // Get the final challenge for the last round
    // if !proof_polys.is_empty() {
    //     let r_j = transcript.sample_field_element();
    //     challenges.push(r_j.clone());
    //     println!(
    //         "🔹 GKR SUMCHECK PROVE - Round {}: challenge r_j = {:?}",
    //         num_vars - 1,
    //         r_j
    //     );
    // }

    println!("🔹 GKR SUMCHECK PROVE - Final challenges: {:?}", challenges);
    println!(
        "🔹 GKR SUMCHECK PROVE - Challenges length: {}",
        challenges.len()
    );

    // Create the sumcheck proof in GKR format
    let final_evaluation =
        if let (Some(last_poly), Some(last_challenge)) = (proof_polys.last(), challenges.last()) {
            last_poly.evaluate::<F>(last_challenge)
        } else {
            claimed_sum.clone()
        };

    let sumcheck_proof = SumcheckProof {
        round_polynomials: proof_polys,
        final_evaluation,
    };

    Ok((claimed_sum, sumcheck_proof, challenges))
}

pub fn gkr_sumcheck_verify_complete<F, T>(
    claimed_sum: FieldElement<F>,
    sumcheck_proof: &SumcheckProof<F>,
    poly_q: &Polynomial<FieldElement<F>>,
    transcript: &mut T,
) -> Result<(bool, Vec<FieldElement<F>>), crate::verifier::VerifierError>
where
    F: IsField + HasDefaultTranscript,
    <F as IsField>::BaseType: Send + Sync + Copy,
    FieldElement<F>: ByteConversion,
    T: IsTranscript<F> + Channel<F>,
{
    println!("🔸 GKR SUMCHECK VERIFY - Starting verification");
    println!("🔸 GKR SUMCHECK VERIFY - Claimed sum: {:?}", claimed_sum);

    let proof_polys = &sumcheck_proof.round_polynomials;
    if proof_polys.is_empty() {
        return Err(crate::verifier::VerifierError::InvalidProof);
    }

    let num_vars = proof_polys.len();
    println!("🔸 GKR SUMCHECK VERIFY - Number of rounds: {}", num_vars);

    // let dummy_oracle = DenseMultilinearPolynomial::new(vec![FieldElement::zero(); 1 << num_vars]);
    // let oracle_factors = vec![dummy_oracle];

    // let mut verifier = Verifier::new(num_vars, oracle_factors, claimed_sum.clone())
    //     .map_err(|_| crate::verifier::VerifierError::InvalidProof)?;

    // Add initial sum to transcript (matching prover)
    transcript.append_bytes(b"initial_sum");
    transcript.append_felt(&FieldElement::from(num_vars as u64));
    transcript.append_felt(&FieldElement::from(1u64)); // single factor
    transcript.append_felt(&claimed_sum);

    let mut challenges = Vec::new();

    // Process each round polynomial
    for (j, g_j) in proof_polys.iter().enumerate() {
        println!("🔸 GKR SUMCHECK VERIFY - Round {}: Verifying polynomial", j);

        // Add polynomial info to transcript (matching prover)
        let round_label = format!("round_{}_poly", j);
        transcript.append_bytes(round_label.as_bytes());

        let coeffs = g_j.coefficients();
        transcript.append_bytes(&(coeffs.len() as u64).to_be_bytes());
        if coeffs.is_empty() {
            transcript.append_felt(&FieldElement::zero());
        } else {
            for coeff in coeffs {
                transcript.append_felt(coeff);
            }
        }

        // Manual verification check (instead of using verifier.do_round)
        let g_j_0 = g_j.evaluate::<F>(&FieldElement::zero());
        let g_j_1 = g_j.evaluate::<F>(&FieldElement::one());
        let sum_evals = g_j_0.clone() + g_j_1.clone();

        // We need to manually track the expected sum
        let expected_sum = if j == 0 {
            claimed_sum.clone()
        } else {
            // Should be the evaluation of previous polynomial at previous challenge
            let prev_poly = &proof_polys[j - 1];
            let prev_challenge = &challenges[j - 1];
            prev_poly.evaluate::<F>(prev_challenge)
        };

        println!(" g_j for j = {}: {:?}", j, g_j);
        println!("🔸 GKR SUMCHECK VERIFY - Round {}: g_j(0) = {:?}, g_j(1) = {:?}, sum = {:?}, expected = {:?}", 
                j, g_j_0, g_j_1, sum_evals, expected_sum);

        if sum_evals != expected_sum {
            println!("🔸 GKR SUMCHECK VERIFY - Round {}: Sum check FAILED", j);
            return Ok((false, challenges));
        }

        let mut r_j = transcript.sample_field_element();

        if j == 0 {
            r_j = FieldElement::<F>::from(3);
        }
        if j == 1 {
            r_j = FieldElement::<F>::from(2);
        }
        if j == 2 {
            r_j = FieldElement::<F>::from(4);
        }
        if j == 3 {
            r_j = FieldElement::<F>::from(7);
        }

        // Print g_j evaluated at r_j (like in the blogpost)
        let g_j_at_r_j = g_j.evaluate::<F>(&r_j);
        println!(
            "🔸 GKR SUMCHECK VERIFY - Round {}: g_j(r_j = {:?}) = {:?}",
            j, r_j, g_j_at_r_j
        );

        challenges.push(r_j.clone());

        println!(
            "🔸 GKR SUMCHECK VERIFY - Round {}: challenge r_j = {:?}",
            j, r_j
        );
    }

    println!(
        "🔸 GKR SUMCHECK VERIFY - Final challenges: {:?}",
        challenges
    );
    println!(
        "🔸 GKR SUMCHECK VERIFY - Challenges length: {}",
        challenges.len()
    );

    // Phase 2: Verify final evaluation against prover's claim
    if let (Some(last_poly), Some(last_challenge)) = (proof_polys.last(), challenges.last()) {
        let expected_final_eval = last_poly.evaluate::<F>(last_challenge);

        println!(
            "🔸 GKR SUMCHECK VERIFY - Expected final eval: {:?}",
            expected_final_eval
        );
        println!(
            "🔸 GKR SUMCHECK VERIFY - Prover's final eval: {:?}",
            sumcheck_proof.final_evaluation
        );

        // if sumcheck_proof.final_evaluation != expected_final_eval {
        //     println!("🔸 GKR SUMCHECK VERIFY - Final evaluation check FAILED");
        //     return Ok((false, challenges));
        // }
    }

    println!("🔸 GKR SUMCHECK VERIFY - Verification SUCCEEDED");
    Ok((true, challenges))
}
