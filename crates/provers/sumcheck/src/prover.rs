use crate::sum_product_over_suffix;
use crate::Channel;
use lambdaworks_crypto::fiat_shamir::default_transcript::DefaultTranscript;
use lambdaworks_crypto::fiat_shamir::is_transcript::IsTranscript;
use lambdaworks_math::{
    field::{
        element::FieldElement,
        traits::{HasDefaultTranscript, IsField},
    },
    polynomial::{
        dense_multilinear_poly::DenseMultilinearPolynomial, InterpolateError, Polynomial,
    },
    traits::ByteConversion,
};
use std::ops::Mul;

pub type ProofResult<F> = (FieldElement<F>, Vec<Polynomial<FieldElement<F>>>);
pub type ProverOutput<F> = Result<ProofResult<F>, ProverError>;

#[derive(Debug)]
pub enum ProverError {
    /// Error when the input factors are inconsistent
    FactorMismatch(String),
    /// Error occurring during the calculation within a specific round.
    RoundError(String),
    /// Error during the polynomial interpolation step in a round.
    InterpolationError(InterpolateError),
    /// Error during the initial sum calculation.
    SummationError(String),
    /// Error indicating the Prover is in an invalid state
    InvalidState(String),
}

impl From<InterpolateError> for ProverError {
    fn from(e: InterpolateError) -> Self {
        ProverError::InterpolationError(e)
    }
}

/// Represents the Prover for the Sum-Check protocol operating on a product of DenseMultilinearPolynomials.
pub struct Prover<F: IsField>
where
    F::BaseType: Send + Sync,
{
    num_vars: usize,
    /// The factors \( P_i \) of the product being summed.
    factors: Vec<DenseMultilinearPolynomial<F>>,
    /// Challenges \( r_1, r_2, ... \) received from the Verifier in previous rounds.
    challenges: Vec<FieldElement<F>>,
}

impl<F: IsField> Prover<F>
where
    F::BaseType: Send + Sync,
    FieldElement<F>: Clone + Mul<Output = FieldElement<F>>,
{
    /// Creates a new Prover instance for the given factors.
    pub fn new(factors: Vec<DenseMultilinearPolynomial<F>>) -> Result<Self, ProverError> {
        if factors.is_empty() {
            return Err(ProverError::FactorMismatch(
                "At least one polynomial factor is required.".to_string(),
            ));
        }
        let num_vars = factors[0].num_vars();
        if factors.iter().any(|p| p.num_vars() != num_vars) {
            return Err(ProverError::FactorMismatch(
                "All factors must have the same number of variables.".to_string(),
            ));
        }
        Ok(Self {
            num_vars,
            factors,
            challenges: Vec::with_capacity(num_vars),
        })
    }

    /// Returns the number of variables in the polynomials handled by this prover.
    pub fn num_vars(&self) -> usize {
        self.num_vars
    }

    /// Computes the initial claimed sum \( C = \\sum_{x \\in \\{0,1\\}^n} \\prod_i P_i(x) \).
    pub fn compute_initial_sum(&self) -> Result<FieldElement<F>, ProverError> {
        sum_product_over_suffix(&self.factors, &[])
            .map_err(|e| ProverError::SummationError(format!("Error computing initial sum: {}", e)))
    }

    /// Executes a round of the Sum-Check protocol.
    ///
    /// Given the challenge `r_prev` from the previous round (if any), this function
    /// computes the round polynomial \( g_j(X_j) \).
    /// \( g_j(X_j) = \\sum_{x_{j+1}, ..., x_n \\in \\{0,1\\}} \\prod_i P_i(r_1, ..., r_{j-1}, X_j, x_{j+1}, ..., x_n) \)
    /// This is achieved by evaluating the sum at `deg(g_j) + 1` points and interpolating.
    pub fn round(
        &mut self,
        r_prev: Option<&FieldElement<F>>,
    ) -> Result<Polynomial<FieldElement<F>>, ProverError> {
        // Store the challenge from the previous round
        if let Some(r) = r_prev {
            if self.challenges.len() >= self.num_vars {
                return Err(ProverError::InvalidState(
                    "Received challenge when all variables are already fixed.".to_string(),
                ));
            }
            self.challenges.push(r.clone());
        }

        // Check if all rounds are completed
        let current_round_idx = self.challenges.len();
        if current_round_idx >= self.num_vars {
            return Err(ProverError::InvalidState(
                "All variables already fixed, no more rounds to run.".to_string(),
            ));
        }

        // Calculate the polynomial g_j(X_j) by interpolation.
        // The degree of g_j is at most the number of factors.
        let num_eval_points = self.factors.len() + 1;
        let mut evaluation_points_x = Vec::with_capacity(num_eval_points);
        let mut evaluations_y = Vec::with_capacity(num_eval_points);

        // Prefix for evaluation points: (r1, r2, ..., r_{j-1}, eval_point_x)
        let mut current_point_prefix = self.challenges.clone();
        current_point_prefix.push(FieldElement::zero());

        for i in 0..num_eval_points {
            // Point at which to evaluate X_j
            let eval_point_x = FieldElement::from(i as u64);
            evaluation_points_x.push(eval_point_x.clone());

            // Set the actual value for X_j in the prefix
            *current_point_prefix.last_mut().unwrap() = eval_point_x;

            let g_j_at_eval_point = sum_product_over_suffix(&self.factors, &current_point_prefix)
                .map_err(|e| {
                ProverError::RoundError(format!("Error in sum for g_j({}): {}", i, e))
            })?;
            evaluations_y.push(g_j_at_eval_point);
        }

        let poly_g_j = Polynomial::interpolate(&evaluation_points_x, &evaluations_y)?;

        Ok(poly_g_j)
    }
}

pub fn prove<F>(factors: Vec<DenseMultilinearPolynomial<F>>) -> ProverOutput<F>
where
    F: IsField + HasDefaultTranscript,
    F::BaseType: Send + Sync,
    FieldElement<F>: Clone + Mul<Output = FieldElement<F>> + ByteConversion,
{
    // Initialize the prover
    let mut prover = Prover::new(factors.clone())?;
    let num_vars = prover.num_vars();
    // Compute the claimed sum C
    let claimed_sum = prover.compute_initial_sum()?;

    // Initialize Fiat-Shamir transcript
    let mut transcript = DefaultTranscript::<F>::default();
    transcript.append_bytes(b"initial_sum");
    transcript.append_felt(&FieldElement::from(num_vars as u64));
    transcript.append_felt(&FieldElement::from(factors.len() as u64));
    transcript.append_felt(&claimed_sum);

    let mut proof_polys = Vec::with_capacity(num_vars);
    let mut current_challenge: Option<FieldElement<F>> = None;

    // Execute rounds
    for j in 0..num_vars {
        // Prover computes the round polynomial g_j
        let g_j = prover.round(current_challenge.as_ref())?;

        // Append g_j information to transcript for the verifier to derive challenge
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

        // Derive challenge for the next round from transcript (if not the last round)
        if j < num_vars - 1 {
            current_challenge = Some(transcript.draw_felt());
        } else {
            // No challenge needed after the last round polynomial is sent
            current_challenge = None;
        }
    }

    Ok((claimed_sum, proof_polys))
}
