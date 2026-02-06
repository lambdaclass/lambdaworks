use crate::evaluate_product_at_point;
use crate::Channel;
use crate::EvaluationError;
use lambdaworks_crypto::fiat_shamir::default_transcript::DefaultTranscript;
use lambdaworks_math::{
    field::{
        element::FieldElement,
        traits::{HasDefaultTranscript, IsField},
    },
    polynomial::{dense_multilinear_poly::DenseMultilinearPolynomial, Polynomial},
    traits::ByteConversion,
};
use std::ops::Mul;
use thiserror::Error;

/// Represents the result of a single round of verification.
#[derive(Debug)]
pub enum VerifierRoundResult<F: IsField> {
    /// The round was successful, providing the challenge for the next round.
    NextRound(FieldElement<F>),
    Final(bool),
}

#[derive(Debug, Error)]
pub enum VerifierError<F: IsField>
where
    F::BaseType: Send + Sync,
{
    #[error("Inconsistent sum at round {round}: g({round})(0)+g({round})(1) = {s0:?}+{s1:?}, expected {expected:?}")]
    InconsistentSum {
        round: usize,
        s0: FieldElement<F>,
        s1: FieldElement<F>,
        expected: FieldElement<F>,
    },
    #[error("Oracle evaluation error: {0}")]
    OracleEvaluationError(EvaluationError),
    #[error("Invalid degree at round {round}: got {actual_degree}, max allowed {max_allowed}")]
    InvalidDegree {
        round: usize,
        actual_degree: usize,
        max_allowed: usize,
    },
    #[error("Incorrect proof length: expected {expected}, got {actual}")]
    IncorrectProofLength { expected: usize, actual: usize },
    #[error("Missing oracle: at least one polynomial factor is required")]
    MissingOracle,
    #[error("Invalid verifier state: {0}")]
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
            // Starts with the initial claimed sum C
            current_sum: claimed_sum,
            challenges: Vec::with_capacity(num_vars),
        })
    }

    /// Executes a verification round based on the polynomial \( g_j \) received from the prover.
    pub fn do_round<C: Channel<F>>(
        &mut self,
        g_j: Polynomial<FieldElement<F>>,
        transcript: &mut C,
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
                    let success = expected_final_eval == self.current_sum;
                    Ok(VerifierRoundResult::Final(success))
                }
                Err(e) => Err(VerifierError::OracleEvaluationError(e)),
            }
        } else {
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
    if proof_polys.len() != num_vars {
        return Err(VerifierError::IncorrectProofLength {
            expected: num_vars,
            actual: proof_polys.len(),
        });
    }

    let num_factors = oracle_factors.len();
    let mut transcript = DefaultTranscript::<F>::default();
    crate::init_transcript(&mut transcript, num_vars, num_factors, &claimed_sum);

    let mut verifier = Verifier::new(num_vars, oracle_factors, claimed_sum)?;

    for (j, g_j) in proof_polys.into_iter().enumerate() {
        crate::append_round_poly(&mut transcript, j, &g_j);

        match verifier.do_round(g_j, &mut transcript)? {
            VerifierRoundResult::NextRound(_) => continue,
            VerifierRoundResult::Final(result) => {
                if j == num_vars - 1 {
                    return Ok(result);
                } else {
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
