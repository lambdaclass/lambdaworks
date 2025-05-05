use crate::{evaluate_product_at_point, Channel}; // Use helper from lib.rs
use lambdaworks_crypto::fiat_shamir::default_transcript::DefaultTranscript;
use lambdaworks_crypto::fiat_shamir::is_transcript::IsTranscript; // Import trait
use lambdaworks_math::{
    field::{
        element::FieldElement,
        traits::{HasDefaultTranscript, IsField},
    },
    polynomial::{dense_multilinear_poly::DenseMultilinearPolynomial, Polynomial},
    traits::ByteConversion,
};
use std::ops::Mul;
use std::vec::Vec;

/// Represents the result of a single round of verification.
#[derive(Debug)]
pub enum VerifierRoundResult<F: IsField> {
    /// The round was successful, providing the challenge for the next round.
    NextRound(FieldElement<F>),
    /// This was the final round, indicating the overall success or failure of the verification.
    Final(bool),
}

#[derive(Debug)]
pub enum VerifierError<F: IsField>
where
    F::BaseType: Send + Sync,
{
    /// The sum check g(0) + g(1) == expected failed.
    InconsistentSum {
        round: usize,
        s0: FieldElement<F>,
        s1: FieldElement<F>,
        expected: FieldElement<F>,
    },
    /// Error evaluating the product of oracle polynomials at the claimed point.
    OracleEvaluationError(String),
    /// The degree of the polynomial sent by the prover is invalid for the current round.
    InvalidDegree {
        round: usize,
        actual_degree: usize,
        max_allowed: usize,
    },
    /// The proof contains an incorrect number of polynomials.
    IncorrectProofLength { expected: usize, actual: usize },
    /// The list of oracle factors provided was empty.
    MissingOracle,
    /// Error indicating the Verifier is in an invalid state (e.g., inconsistent factors, unexpected round result).
    InvalidState(String),
}

/// Represents the Verifier for the Sum-Check protocol operating on a product of DenseMultilinearPolynomials.

#[derive(Debug)]
pub struct Verifier<F: IsField>
where
    F::BaseType: Send + Sync,
    FieldElement<F>: Clone + Mul<Output = FieldElement<F>>,
{
    num_vars: usize,
    round: usize,
    /// The claimed factors \( P_i \) of the product, used for the final check.
    oracle_factors: Vec<DenseMultilinearPolynomial<F>>,
    current_sum: FieldElement<F>,
    challenges: Vec<FieldElement<F>>,
}

impl<F: IsField> Verifier<F>
where
    F::BaseType: Send + Sync,
    FieldElement<F>: Clone + Mul<Output = FieldElement<F>>,
{
    /// Creates a new Verifier instance.
    pub fn new(
        num_vars: usize,
        oracle_factors: Vec<DenseMultilinearPolynomial<F>>,
        claimed_sum: FieldElement<F>,
    ) -> Result<Self, VerifierError<F>> {
        if oracle_factors.is_empty() {
            // Need at least one factor for the product evaluation.
            return Err(VerifierError::MissingOracle);
        }
        if oracle_factors.iter().any(|p| p.num_vars() != num_vars) {
            // All factors must operate on the same number of variables.
            return Err(VerifierError::InvalidState(
                "Oracle factors have inconsistent number of variables.".to_string(),
            ));
        }

        Ok(Self {
            num_vars,
            round: 0,
            oracle_factors,
            current_sum: claimed_sum, // Starts with the initial claimed sum C
            challenges: Vec::with_capacity(num_vars),
        })
    }

    /// Executes a verification round based on the polynomial \( g_j \) received from the prover.
    pub fn do_round<C: Channel<F>>(
        &mut self,
        g_j: Polynomial<FieldElement<F>>,
        transcript: &mut C, // Transcript is now primarily used *here* to draw the challenge
    ) -> Result<VerifierRoundResult<F>, VerifierError<F>> {
        // Check if we are past the expected number of rounds.
        if self.round >= self.num_vars {
            return Err(VerifierError::InvalidState(
                "Round number exceeds number of variables.".to_string(),
            ));
        }

        // 1. Check degree of g_j.
        // The degree of g_j(X_j) = sum_{...} prod P_i(...) can be at most the number of factors.
        let max_degree = self.oracle_factors.len();
        if g_j.degree() > max_degree {
            return Err(VerifierError::InvalidDegree {
                round: self.round,
                actual_degree: g_j.degree(),
                max_allowed: max_degree,
            });
        }

        // 2. Check consistency: g_j(0) + g_j(1) == expected_sum (current_sum)
        let zero = FieldElement::<F>::zero();
        let one = FieldElement::<F>::one();
        let eval_0 = g_j.evaluate(&zero);
        let eval_1 = g_j.evaluate(&one);
        let sum_evals = eval_0.clone() + eval_1.clone();

        if sum_evals != self.current_sum {
            // The prover's polynomial g_j does not match the expected sum from the previous round (or initial C).
            return Err(VerifierError::InconsistentSum {
                round: self.round,
                s0: eval_0,
                s1: eval_1,
                expected: self.current_sum.clone(),
            });
        }

        // 3. Obtain challenge r_j for this round from the transcript.
        let r_j = transcript.draw_felt();
        self.challenges.push(r_j.clone());

        // 4. Update the expected sum for the *next* round: current_sum = g_j(r_j)
        self.current_sum = g_j.evaluate(&r_j);
        self.round += 1;

        // 5. Check if this is the final round.
        if self.round == self.num_vars {
            // Perform the final check: evaluate prod P_i(r1, ..., rn)
            match evaluate_product_at_point(&self.oracle_factors, &self.challenges) {
                Ok(expected_final_eval) => {
                    // Compare with the value computed from the last round polynomial: g_{n-1}(r_{n-1})
                    let success = expected_final_eval == self.current_sum;
                    Ok(VerifierRoundResult::Final(success))
                }
                Err(e) => Err(VerifierError::OracleEvaluationError(e)), // Pass the specific evaluation error
            }
        } else {
            // Not the final round, return the challenge for the Prover's next step (though prover derives it too).
            Ok(VerifierRoundResult::NextRound(r_j))
        }
    }
}

pub fn verify<F>(
    num_vars: usize,
    claimed_sum: FieldElement<F>,
    proof_polys: Vec<Polynomial<FieldElement<F>>>,
    oracle_factors: Vec<DenseMultilinearPolynomial<F>>,
) -> Result<bool, VerifierError<F>>
where
    F: IsField + HasDefaultTranscript,
    F::BaseType: Send + Sync,
    FieldElement<F>: Clone + Mul<Output = FieldElement<F>> + ByteConversion,
{
    // Basic check: ensure the number of polynomials matches the number of variables.
    if proof_polys.len() != num_vars {
        return Err(VerifierError::IncorrectProofLength {
            expected: num_vars,
            actual: proof_polys.len(),
        });
    }

    // Initialize Verifier
    let mut verifier = Verifier::new(num_vars, oracle_factors, claimed_sum.clone())?;

    // Initialize Fiat-Shamir transcript - must match the prover's initialization.
    let mut transcript = DefaultTranscript::<F>::default();
    transcript.append_bytes(b"initial_sum");
    transcript.append_felt(&claimed_sum);

    // Process each round polynomial from the proof.
    for (j, g_j) in proof_polys.into_iter().enumerate() {
        // Reconstruct the transcript state before drawing the challenge.
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

        match verifier.do_round(g_j, &mut transcript)? {
            VerifierRoundResult::NextRound(_) => {
                // Consistency checks passed, challenge r_j generated and stored.
                // Continue to the next round.
                continue;
            }
            VerifierRoundResult::Final(result) => {
                // This was the last round (j == num_vars - 1).
                if j == num_vars - 1 {
                    // Return the final result from the last round check.
                    return Ok(result);
                } else {
                    // Should not get Final result before the last round.
                    return Err(VerifierError::InvalidState(
                        "Final result obtained before the last round.".to_string(),
                    ));
                }
            }
        }
    }

    Err(VerifierError::InvalidState(
        "Verification loop finished unexpectedly.".to_string(),
    ))
}
