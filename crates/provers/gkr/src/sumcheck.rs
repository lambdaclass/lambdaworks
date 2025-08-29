use lambdaworks_crypto::fiat_shamir::is_transcript::IsTranscript;
use lambdaworks_math::{
    field::{
        element::FieldElement,
        traits::{HasDefaultTranscript, IsField},
    },
    polynomial::{dense_multilinear_poly::DenseMultilinearPolynomial, Polynomial},
    traits::ByteConversion,
};
use lambdaworks_sumcheck::{Channel, Prover};

use crate::prover::ProverError;

/// GKR-specific sumcheck proof, which contains each round polynomial `g_j` and the challenges used.
#[derive(Debug, Clone)]
pub struct GKRSumcheckProof<F: IsField> {
    pub round_polynomials: Vec<Polynomial<FieldElement<F>>>,
    pub challenges: Vec<FieldElement<F>>,
}

/// GKR-specific sumcheck prover.
/// This function will recieve a vector of two terms. Each term contains two multilinear polynomials.
/// This separation of terms is necessary because the classic/original sumcheck only accepts a product of multilinear polynomials.
/// In this way, we apply the sumcheck to two products, each consisting of two factors.
pub fn gkr_sumcheck_prove<F, T>(
    terms: Vec<Vec<DenseMultilinearPolynomial<F>>>,
    transcript: &mut T,
) -> Result<GKRSumcheckProof<F>, ProverError>
where
    F: IsField + HasDefaultTranscript,
    <F as IsField>::BaseType: Send + Sync + Copy,
    FieldElement<F>: ByteConversion,
    T: IsTranscript<F> + Channel<F>,
{
    if terms.len() != 2 {
        return Err(ProverError::SumcheckError);
    }

    let factors_term_1 = terms[0].clone();
    let factors_term_2 = terms[1].clone();

    // Create two separate sumcheck provers for each term.
    let mut prover_term_1 = Prover::new(factors_term_1).map_err(|_| ProverError::SumcheckError)?;
    let mut prover_term_2 = Prover::new(factors_term_2).map_err(|_| ProverError::SumcheckError)?;

    // Both terms have the same number of variables.
    let num_vars = prover_term_1.num_vars();

    let claimed_sum_term_1 = prover_term_1
        .compute_initial_sum()
        .map_err(|_| ProverError::SumcheckError)?;

    let claimed_sum_term_2 = prover_term_2
        .compute_initial_sum()
        .map_err(|_| ProverError::SumcheckError)?;
    let claimed_sum = claimed_sum_term_1 + claimed_sum_term_2;

    transcript.append_bytes(b"initial_sum");
    transcript.append_felt(&FieldElement::from(num_vars as u64));
    transcript.append_felt(&FieldElement::from(1u64));
    transcript.append_felt(&claimed_sum);

    let mut proof_polys = Vec::with_capacity(num_vars);
    let mut challenges = Vec::new();
    let mut current_challenge: Option<FieldElement<F>> = None;

    // Execute sumcheck rounds
    for j in 0..num_vars {
        let g_j_term_1 = prover_term_1
            .round(current_challenge.as_ref())
            .map_err(|_| ProverError::SumcheckError)?;

        let g_j_term_2 = prover_term_2
            .round(current_challenge.as_ref())
            .map_err(|_| ProverError::SumcheckError)?;

        let g_j = g_j_term_1 + g_j_term_2;

        // Add polynomial to transcript
        let round_label = format!("round_{j}_poly");
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
        let r_j = transcript.sample_field_element();

        challenges.push(r_j.clone());
        current_challenge = Some(r_j);
    }

    let sumcheck_proof = GKRSumcheckProof {
        round_polynomials: proof_polys,
        challenges,
    };

    Ok(sumcheck_proof)
}

/// GKR-specific sumcheck Verifier.
pub fn gkr_sumcheck_verify<F, T>(
    claimed_sum: FieldElement<F>,
    sumcheck_proof: &GKRSumcheckProof<F>,
    transcript: &mut T,
) -> Result<(bool, Vec<FieldElement<F>>), crate::verifier::VerifierError>
where
    F: IsField + HasDefaultTranscript,
    <F as IsField>::BaseType: Send + Sync + Copy,
    FieldElement<F>: ByteConversion,
    T: IsTranscript<F> + Channel<F>,
{
    let proof_polys = &sumcheck_proof.round_polynomials;
    if proof_polys.is_empty() {
        return Err(crate::verifier::VerifierError::InvalidProof);
    }

    let num_vars = proof_polys.len();

    transcript.append_bytes(b"initial_sum");
    transcript.append_felt(&FieldElement::from(num_vars as u64));
    transcript.append_felt(&FieldElement::from(1u64)); // single factor
    transcript.append_felt(&claimed_sum);

    let mut challenges = Vec::new();

    // Verify each round polynomial.
    for (j, g_j) in proof_polys.iter().enumerate() {
        // Add polynomial info to transcript
        let round_label = format!("round_{j}_poly");
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

        // Check that the degree of `g_j` does not exceed the theoretical bound.
        // The polynomial `g_j` should be cuadratic since the polynomial `f(b,c)` to which the sumcheck is applied
        // is the sum of of two products, where each one is the product of two multilinear polynomials.
        if g_j.degree() > 2 {
            return Err(crate::verifier::VerifierError::InvalidDegree);
        }

        // Verify `g_j(0) + g_j(1) = m_{j-1}`, where:
        // `m_{j-1} = g_{j-1} (s_{j-1})`, the previous claimed sum.
        let g_j_0 = g_j.evaluate::<F>(&FieldElement::zero());
        let g_j_1 = g_j.evaluate::<F>(&FieldElement::one());
        let sum_evals = &g_j_0 + &g_j_1;

        let expected_sum = if j == 0 {
            claimed_sum.clone()
        } else {
            // Should be the evaluation of previous polynomial at previous challenge
            let prev_poly = &proof_polys[j - 1];
            let prev_challenge = &challenges[j - 1];
            prev_poly.evaluate::<F>(prev_challenge)
        };

        if sum_evals != expected_sum {
            println!("Sumcheck verification failed at round {j}");
            return Ok((false, challenges));
        }

        let r_j = transcript.sample_field_element();

        challenges.push(r_j.clone());
    }

    Ok((true, challenges))
}
