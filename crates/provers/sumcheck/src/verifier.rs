use crate::Channel;
use lambdaworks_crypto::fiat_shamir::default_transcript::DefaultTranscript;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::HasDefaultTranscript;
use lambdaworks_math::field::traits::IsField;
use lambdaworks_math::polynomial::{
    dense_multilinear_poly::DenseMultilinearPolynomial, Polynomial,
};
use lambdaworks_math::traits::ByteConversion;
use std::vec::Vec;

/// Result of a verifier round
#[derive(Debug)]
pub enum VerifierRoundResult<F: IsField> {
    /// Next round with challenge
    NextRound(FieldElement<F>),
    /// Final result
    Final(bool),
}

#[derive(Debug)]
pub enum VerifierError<F: IsField>
where
    <F as IsField>::BaseType: Send + Sync,
{
    /// The sum of evaluations at 0 and 1 does not match the expected value.
    InconsistentSum {
        round: usize,
        s0: FieldElement<F>,
        s1: FieldElement<F>,
        expected: FieldElement<F>,
    },
    /// Error when evaluating the oracle polynomial in the final round.
    OracleEvaluationError,
    /// Error when the degree of the univariate polynomial is greater than 1.
    InvalidDegree {
        round: usize,
        actual_degree: usize,
        max_allowed: usize,
    },
    /// Error when the proof contains an incorrect number of polynomials.
    IncorrectProofLength { expected: usize, actual: usize },
    /// This represents a code path that should never be reached.
    Unreachable,
}

/// Verifier for the Sum-Check protocol
pub struct Verifier<F: IsField>
where
    <F as IsField>::BaseType: Send + Sync,
{
    pub n: usize,
    pub round: usize,
    pub poly: Option<DenseMultilinearPolynomial<F>>,
    pub current_sum: FieldElement<F>,
    pub challenges: Vec<FieldElement<F>>,
}

impl<F: IsField> Verifier<F>
where
    <F as IsField>::BaseType: Send + Sync,
{
    pub fn new(
        n: usize,
        poly: Option<DenseMultilinearPolynomial<F>>,
        claimed_sum: FieldElement<F>,
    ) -> Self {
        Self {
            n,
            round: 0,
            poly,
            current_sum: claimed_sum,
            challenges: Vec::with_capacity(n),
        }
    }

    /// Executes a single round of verification
    pub fn do_round<C: Channel<F>>(
        &mut self,
        g: Polynomial<FieldElement<F>>,
        transcript: &mut C,
    ) -> Result<VerifierRoundResult<F>, VerifierError<F>> {
        // Check if we've exceeded the number of rounds
        if self.round >= self.n {
            return Err(VerifierError::Unreachable);
        }

        // Check if the polynomial is of the correct degree
        if g.degree() > 1 {
            return Err(VerifierError::InvalidDegree {
                round: self.round,
                actual_degree: g.degree(),
                max_allowed: 1,
            });
        }

        // Check if g(0) + g(1) = current_sum
        let zero = FieldElement::<F>::zero();
        let one = FieldElement::<F>::one();
        let eval_0 = g.evaluate(&zero);
        let eval_1 = g.evaluate(&one);
        let sum = &eval_0 + &eval_1;

        if sum != self.current_sum {
            return Err(VerifierError::InconsistentSum {
                round: self.round,
                s0: eval_0,
                s1: eval_1,
                expected: self.current_sum.clone(),
            });
        }

        // Generate challenge for next round
        let r = transcript.draw_felt();
        self.challenges.push(r.clone());
        self.current_sum = g.evaluate(&r);
        self.round += 1;

        // If this is the final round, check the final evaluation
        if self.round == self.n {
            if let Some(poly) = &self.poly {
                let full_point = self.challenges.clone();
                match poly.evaluate(full_point) {
                    Ok(expected) => {
                        return Ok(VerifierRoundResult::Final(expected == self.current_sum));
                    }
                    Err(_) => return Err(VerifierError::OracleEvaluationError),
                }
            }
            return Ok(VerifierRoundResult::Final(true));
        }

        Ok(VerifierRoundResult::NextRound(r))
    }
}

/// Main verification function for the sumcheck protocol
pub fn verify<F>(
    num_vars: usize,
    claimed_sum: FieldElement<F>,
    proof_polys: Vec<Polynomial<FieldElement<F>>>,
    oracle: Option<DenseMultilinearPolynomial<F>>,
) -> Result<bool, VerifierError<F>>
where
    F: IsField + HasDefaultTranscript,
    F::BaseType: Send + Sync,
    FieldElement<F>: ByteConversion,
{
    if proof_polys.len() != num_vars {
        return Err(VerifierError::IncorrectProofLength {
            expected: num_vars,
            actual: proof_polys.len(),
        });
    }

    let mut verifier = Verifier::new(num_vars, oracle, claimed_sum.clone());
    let mut transcript = DefaultTranscript::<F>::default();

    // Initialize channel with claim
    transcript.append_felt(&claimed_sum);

    for poly in proof_polys {
        // Re-absorb message
        for coeff in &poly.coefficients {
            transcript.append_felt(coeff);
        }
        match verifier.do_round(poly, &mut transcript)? {
            VerifierRoundResult::NextRound(_) => continue,
            VerifierRoundResult::Final(result) => return Ok(result),
        }
    }

    Ok(false)
}
