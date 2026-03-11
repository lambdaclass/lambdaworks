//! Spartan verifier.
//!
//! Verifies a Spartan proof by replaying the Fiat-Shamir transcript and
//! checking the sumcheck and PCS proofs.

use std::marker::PhantomData;

use lambdaworks_crypto::fiat_shamir::default_transcript::DefaultTranscript;
use lambdaworks_crypto::fiat_shamir::is_transcript::IsTranscript;
use lambdaworks_math::field::{
    element::FieldElement,
    traits::{HasDefaultTranscript, IsField},
};
use lambdaworks_math::polynomial::Polynomial;
use lambdaworks_math::traits::ByteConversion;

use crate::errors::SpartanError;
use crate::mle::{eq_poly, matrix_mle_eval, next_power_of_two};
use crate::pcs::IsMultilinearPCS;
use crate::prover::SpartanProof;
use crate::r1cs::R1CS;
use crate::transcript::{
    append_public_inputs, append_r1cs_instance, append_round_poly_to_transcript, draw_challenges,
};

/// The Spartan verifier.
pub struct SpartanVerifier<F, PCS>
where
    F: IsField,
    F::BaseType: Send + Sync,
    PCS: IsMultilinearPCS<F>,
{
    pcs: PCS,
    _f: PhantomData<F>,
}

impl<F, PCS> SpartanVerifier<F, PCS>
where
    F: IsField + HasDefaultTranscript,
    F::BaseType: Send + Sync,
    FieldElement<F>: ByteConversion,
    PCS: IsMultilinearPCS<F>,
    PCS::Error: 'static,
{
    /// Creates a new SpartanVerifier with the given PCS.
    pub fn new(pcs: PCS) -> Self {
        Self {
            pcs,
            _f: PhantomData,
        }
    }

    /// Verifies a Spartan proof.
    ///
    /// Returns Ok(true) if the proof is valid, Ok(false) if it's invalid but
    /// structurally sound, or Err if there's a structural error.
    pub fn verify(
        &self,
        r1cs: &R1CS<F>,
        public_inputs: &[FieldElement<F>],
        proof: &SpartanProof<F, PCS>,
    ) -> Result<bool, SpartanError> {
        // -----------------------------------------------------------------------
        // Step 1: Initialize transcript identically to prover
        // -----------------------------------------------------------------------
        let mut transcript = DefaultTranscript::<F>::default();
        transcript.append_bytes(b"lambdaworks-spartan-v1");
        append_r1cs_instance(&mut transcript, r1cs);
        append_public_inputs(&mut transcript, public_inputs);
        // Absorb the witness commitment (same step as prover).
        transcript.append_bytes(b"witness_commitment");
        transcript.append_bytes(&PCS::serialize_commitment(&proof.witness_commitment));

        // -----------------------------------------------------------------------
        // Step 2: Draw tau (same as prover)
        // -----------------------------------------------------------------------
        let num_constraints_padded = next_power_of_two(r1cs.num_constraints).max(2);
        let log_constraints = {
            let mut k = 0;
            let mut n = num_constraints_padded;
            while n > 1 {
                k += 1;
                n >>= 1;
            }
            k
        };
        let num_cols_padded = next_power_of_two(r1cs.num_variables).max(2);

        transcript.append_bytes(b"tau_challenge");
        let tau = draw_challenges(&mut transcript, log_constraints);

        // -----------------------------------------------------------------------
        // Step 3: Verify outer sumcheck
        //
        // Claimed sum = 0 (R1CS is satisfied).
        // The outer sumcheck uses a two-term GKR-style sumcheck.
        // Round polynomials have degree ≤ 3 (term1 is cubic, term2 is quadratic).
        // -----------------------------------------------------------------------
        transcript.append_bytes(b"outer_sumcheck");

        let outer_claimed_sum = FieldElement::<F>::zero();

        let (outer_ok, r_x) = verify_two_term_sumcheck_round_polys(
            outer_claimed_sum,
            &proof.outer_sumcheck_polys,
            3, // max degree for cubic sumcheck
            &mut transcript,
        )?;

        if !outer_ok {
            return Ok(false);
        }

        // Verify that the claimed outer challenges match the transcript-derived ones
        if r_x != proof.outer_challenges {
            return Ok(false);
        }

        // -----------------------------------------------------------------------
        // Step 4: Oracle check for outer sumcheck
        //
        // The last round poly at last challenge should equal:
        //   eq(τ, r_x) · v_a · v_b - eq(τ, r_x) · v_c
        //   = eq(τ, r_x) · (v_a · v_b - v_c)
        //
        // This is valid because we use a PRODUCT sumcheck:
        // term1 oracle = eq(τ,r_x) · v_a · v_b
        // term2 oracle = eq(τ,r_x) · (-v_c)
        // -----------------------------------------------------------------------
        let eq_tau_rx = eq_poly(&tau)
            .evaluate(r_x.clone())
            .map_err(|e| SpartanError::VerificationFailed(format!("eq eval error: {e:?}")))?;

        let outer_oracle = &eq_tau_rx * &(&proof.v_a * &proof.v_b - &proof.v_c);

        // The final evaluation from the last round polynomial at the last challenge
        let last_outer_poly = proof.outer_sumcheck_polys.last().ok_or_else(|| {
            SpartanError::VerificationFailed("Empty outer sumcheck polys".to_string())
        })?;
        let last_challenge = r_x.last().ok_or_else(|| {
            SpartanError::VerificationFailed("Empty outer challenges".to_string())
        })?;
        let last_outer_eval = last_outer_poly.evaluate(last_challenge);

        if last_outer_eval != outer_oracle {
            return Ok(false);
        }

        // -----------------------------------------------------------------------
        // Step 5: Append v_a, v_b, v_c to transcript
        // -----------------------------------------------------------------------
        transcript.append_bytes(b"v_a");
        transcript.append_field_element(&proof.v_a);
        transcript.append_bytes(b"v_b");
        transcript.append_field_element(&proof.v_b);
        transcript.append_bytes(b"v_c");
        transcript.append_field_element(&proof.v_c);

        // -----------------------------------------------------------------------
        // Step 6: Draw batching challenges
        // -----------------------------------------------------------------------
        transcript.append_bytes(b"inner_challenges");
        let rho = draw_challenges(&mut transcript, 3);
        let (rho_a, rho_b, rho_c) = (&rho[0], &rho[1], &rho[2]);

        let inner_claimed_sum = rho_a * &proof.v_a + rho_b * &proof.v_b + rho_c * &proof.v_c;

        // -----------------------------------------------------------------------
        // Step 7: Verify inner sumcheck
        // -----------------------------------------------------------------------
        transcript.append_bytes(b"inner_sumcheck");

        let (inner_ok, r_y) = verify_sumcheck_round_polys(
            inner_claimed_sum,
            &proof.inner_sumcheck_polys,
            2, // max degree for quadratic sumcheck (2 factors)
            &mut transcript,
        )?;

        if !inner_ok {
            return Ok(false);
        }

        if r_y != proof.inner_challenges {
            return Ok(false);
        }

        // -----------------------------------------------------------------------
        // Step 8: Oracle check for inner sumcheck
        //
        // Compute the combined matrix evaluation at (r_x, r_y):
        //   rho_a * A(r_x, r_y) + rho_b * B(r_x, r_y) + rho_c * C(r_x, r_y)
        // Then check: combined_matrix_eval * z̃(r_y) = last inner poly eval at r_y[last]
        // -----------------------------------------------------------------------
        let a_eval = matrix_mle_eval(&r1cs.a, num_constraints_padded, num_cols_padded, &r_x, &r_y);
        let b_eval = matrix_mle_eval(&r1cs.b, num_constraints_padded, num_cols_padded, &r_x, &r_y);
        let c_eval = matrix_mle_eval(&r1cs.c, num_constraints_padded, num_cols_padded, &r_x, &r_y);

        let combined_matrix_eval = rho_a * &a_eval + rho_b * &b_eval + rho_c * &c_eval;

        let inner_oracle_eval = &combined_matrix_eval * &proof.witness_eval;

        let last_inner_poly = proof.inner_sumcheck_polys.last().ok_or_else(|| {
            SpartanError::VerificationFailed("Empty inner sumcheck polys".to_string())
        })?;
        let last_inner_challenge = r_y.last().ok_or_else(|| {
            SpartanError::VerificationFailed("Empty inner challenges".to_string())
        })?;
        let last_inner_eval = last_inner_poly.evaluate(last_inner_challenge);

        if last_inner_eval != inner_oracle_eval {
            return Ok(false);
        }

        // Absorb witness_eval into transcript for composability (mirrors prover).
        transcript.append_bytes(b"witness_eval");
        transcript.append_field_element(&proof.witness_eval);

        // -----------------------------------------------------------------------
        // Step 9: Verify PCS opening
        // -----------------------------------------------------------------------
        let pcs_ok = self
            .pcs
            .verify(
                &proof.witness_commitment,
                &r_y,
                &proof.witness_eval,
                &proof.witness_proof,
            )
            .map_err(|e| SpartanError::PcsError(e.to_string()))?;

        if !pcs_ok {
            return Ok(false);
        }

        Ok(true)
    }
}

/// Verifies the two-term (GKR-style) sumcheck round polynomials.
///
/// Uses the same transcript format as `run_two_term_sumcheck` in the prover.
fn verify_two_term_sumcheck_round_polys<F>(
    claimed_sum: FieldElement<F>,
    round_polys: &[Polynomial<FieldElement<F>>],
    max_degree: usize,
    transcript: &mut DefaultTranscript<F>,
) -> Result<(bool, Vec<FieldElement<F>>), SpartanError>
where
    F: IsField + HasDefaultTranscript,
    F::BaseType: Send + Sync,
    FieldElement<F>: ByteConversion,
{
    let num_vars = round_polys.len();

    // Append initial sum to transcript (same as prover)
    transcript.append_bytes(b"initial_sum");
    transcript.append_field_element(&FieldElement::from(num_vars as u64));
    transcript.append_field_element(&FieldElement::from(1u64));
    transcript.append_field_element(&claimed_sum);

    let mut current_claim = claimed_sum;
    let mut challenges = Vec::with_capacity(num_vars);

    for (round, g_j) in round_polys.iter().enumerate() {
        // Check degree bound
        if g_j.degree() > max_degree {
            return Ok((false, challenges));
        }

        // Check g_j(0) + g_j(1) == current_claim
        let eval_0 = g_j.evaluate(&FieldElement::<F>::zero());
        let eval_1 = g_j.evaluate(&FieldElement::<F>::one());
        let sum_check = eval_0 + eval_1;

        if sum_check != current_claim {
            return Ok((false, challenges));
        }

        // Append to transcript (same GKR format as prover)
        let round_label = format!("round_{round}_poly");
        transcript.append_bytes(round_label.as_bytes());
        let coeffs = g_j.coefficients();
        transcript.append_bytes(&(coeffs.len() as u64).to_be_bytes());
        if coeffs.is_empty() {
            transcript.append_field_element(&FieldElement::zero());
        } else {
            for coeff in coeffs {
                transcript.append_field_element(coeff);
            }
        }

        let r = transcript.sample_field_element();

        // Update claim for next round
        current_claim = g_j.evaluate(&r);
        challenges.push(r);
    }

    Ok((true, challenges))
}

/// Verifies sumcheck round polynomials using the external transcript.
///
/// Uses the standard round polynomial transcript format.
fn verify_sumcheck_round_polys<F>(
    mut claimed_sum: FieldElement<F>,
    round_polys: &[Polynomial<FieldElement<F>>],
    max_degree: usize,
    transcript: &mut DefaultTranscript<F>,
) -> Result<(bool, Vec<FieldElement<F>>), SpartanError>
where
    F: IsField + HasDefaultTranscript,
    F::BaseType: Send + Sync,
    FieldElement<F>: ByteConversion,
{
    let num_vars = round_polys.len();
    let mut challenges = Vec::with_capacity(num_vars);

    // Bind the claimed sum to the transcript before rounds (mirrors prover).
    transcript.append_bytes(b"initial_sum");
    transcript.append_field_element(&FieldElement::from(num_vars as u64));
    transcript.append_field_element(&FieldElement::from(1u64));
    transcript.append_field_element(&claimed_sum);

    for (round, g_j) in round_polys.iter().enumerate() {
        // Check degree bound
        if g_j.degree() > max_degree {
            return Ok((false, challenges));
        }

        // Check g_j(0) + g_j(1) == claimed_sum
        let eval_0 = g_j.evaluate(&FieldElement::<F>::zero());
        let eval_1 = g_j.evaluate(&FieldElement::<F>::one());
        let sum_check = eval_0 + eval_1;

        if sum_check != claimed_sum {
            return Ok((false, challenges));
        }

        // Append to transcript and draw challenge (same format as prover)
        append_round_poly_to_transcript(transcript, round, g_j);
        let r = transcript.sample_field_element();

        // Update claimed sum for next round
        claimed_sum = g_j.evaluate(&r);
        challenges.push(r);
    }

    Ok((true, challenges))
}
