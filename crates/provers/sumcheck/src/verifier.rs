use crate::evaluate_product_at_point;
use crate::Channel;
use lambdaworks_crypto::fiat_shamir::default_transcript::DefaultTranscript;
use lambdaworks_crypto::fiat_shamir::is_transcript::IsTranscript;
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

        // Debug: Print what the verifier is checking
        println!(
            "Sumcheck verifier round {}: g_j(0) = {:?}, g_j(1) = {:?}, sum = {:?}, expected = {:?}",
            self.round, eval_0, eval_1, sum_evals, self.current_sum
        );

        if sum_evals != self.current_sum {
            // The prover's polynomial g_j does not match the expected sum from the previous round (or initial C).
            return Err(VerifierError::InconsistentSum {
                round: self.round,
                s0: eval_0,
                s1: eval_1,
                expected: self.current_sum.clone(),
            });
        }

        // 3. Check if this is the final round.
        if self.round == self.num_vars - 1 {
            // Última ronda: obtener challenge, evaluar, agregarlo y luego verificar
            let r_j = transcript.draw_felt();
            self.challenges.push(r_j.clone());
            println!(
                "Sumcheck verifier round {}: Generated challenge {:?} (final round)",
                self.round, r_j
            );
            self.current_sum = g_j.evaluate(&r_j);
            self.round += 1;
            // Verificación final
            match evaluate_product_at_point(&self.oracle_factors, &self.challenges) {
                Ok(expected_final_eval) => {
                    let success = expected_final_eval == self.current_sum;
                    Ok(VerifierRoundResult::Final(success))
                }
                Err(e) => Err(VerifierError::OracleEvaluationError(e)),
            }
        } else {
            // Not the final round, obtain challenge r_j for this round from the transcript.
            let r_j = transcript.draw_felt();
            self.challenges.push(r_j.clone());
            println!(
                "Sumcheck verifier round {}: Generated challenge {:?}",
                self.round, r_j
            );

            // 4. Update the expected sum for the *next* round: current_sum = g_j(r_j)
            self.current_sum = g_j.evaluate(&r_j);
            self.round += 1;
            Ok(VerifierRoundResult::NextRound(r_j))
        }
    }
}

/// Verifier function that uses an external transcript (e.g., for GKR protocol)
/// This function delegates to verify_backend using the provided transcript
pub fn verify_with_transcript<F, T>(
    num_vars: usize,
    claimed_sum: FieldElement<F>,
    proof_polys: Vec<Polynomial<FieldElement<F>>>,
    oracle_polys: Vec<DenseMultilinearPolynomial<F>>,
    transcript: &mut T,
) -> Result<(bool, FieldElement<F>), VerifierError<F>>
where
    F: IsField + HasDefaultTranscript,
    F::BaseType: Send + Sync,
    FieldElement<F>: Clone + Mul<Output = FieldElement<F>> + ByteConversion,
    T: Channel<F> + IsTranscript<F>,
{
    verify_backend(num_vars, claimed_sum, proof_polys, oracle_polys, transcript)
}

/// Verifier function for standalone sumcheck (creates its own transcript)
/// This function delegates to verify_backend with a fresh transcript
pub fn verify<F>(
    num_vars: usize,
    claimed_sum: FieldElement<F>,
    proof_polys: Vec<Polynomial<FieldElement<F>>>,
    oracle_polys: Vec<DenseMultilinearPolynomial<F>>,
) -> Result<bool, VerifierError<F>>
where
    F: IsField + HasDefaultTranscript,
    F::BaseType: Send + Sync,
    FieldElement<F>: Clone + Mul<Output = FieldElement<F>> + ByteConversion,
{
    use lambdaworks_crypto::fiat_shamir::default_transcript::DefaultTranscript;
    let mut tr = DefaultTranscript::<F>::default();
    let (result, _) = verify_backend(num_vars, claimed_sum, proof_polys, oracle_polys, &mut tr)?;
    Ok(result)
}

pub fn verify_backend<F, T>(
    num_vars: usize,
    claimed_sum: FieldElement<F>,
    proof_polys: Vec<Polynomial<FieldElement<F>>>,
    oracle_factors: Vec<DenseMultilinearPolynomial<F>>,
    transcript: &mut T,
) -> Result<(bool, FieldElement<F>), VerifierError<F>>
where
    F: IsField + HasDefaultTranscript,
    F::BaseType: Send + Sync,
    FieldElement<F>: Clone + Mul<Output = FieldElement<F>> + ByteConversion,
    T: Channel<F> + IsTranscript<F>,
{
    // ensure the number of polynomials matches the number of variables.
    if proof_polys.len() != num_vars {
        return Err(VerifierError::IncorrectProofLength {
            expected: num_vars,
            actual: proof_polys.len(),
        });
    }

    let mut verifier = Verifier::new(num_vars, oracle_factors.clone(), claimed_sum.clone())?;

    // Use the provided transcript
    transcript.append_bytes(b"initial_sum");
    transcript.append_felt(&FieldElement::from(num_vars as u64));
    transcript.append_felt(&FieldElement::from(oracle_factors.len() as u64));
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

        match verifier.do_round(g_j, transcript)? {
            VerifierRoundResult::NextRound(_) => {
                // Consistency checks passed, challenge r_j generated and stored.
                // Continue to the next round.
                continue;
            }
            VerifierRoundResult::Final(result) => {
                // This was the last round (j == num_vars - 1).
                if j == num_vars - 1 {
                    // Return the final result from the last round check along with the claimed sum.
                    return Ok((result, claimed_sum));
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
