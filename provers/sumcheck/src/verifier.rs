use super::Channel;
use lambdaworks_crypto::fiat_shamir::default_transcript::DefaultTranscript;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::IsField;
use lambdaworks_math::polynomial::{
    dense_multilinear_poly::DenseMultilinearPolynomial, Polynomial,
};
use lambdaworks_math::traits::ByteConversion;
use std::vec::Vec;

pub enum VerifierRoundResult<F: IsField>
where
    <F as IsField>::BaseType: Send + Sync,
{
    NextRound(FieldElement<F>),
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
}

pub struct Verifier<F: IsField>
where
    <F as IsField>::BaseType: Send + Sync,
{
    pub n: usize,
    pub c_1: FieldElement<F>,
    pub round: usize,
    pub poly: Option<DenseMultilinearPolynomial<F>>,
    pub last_val: FieldElement<F>,
    pub challenges: Vec<FieldElement<F>>,
}

impl<F: IsField> Verifier<F>
where
    <F as IsField>::BaseType: Send + Sync,
{
    pub fn new(
        n: usize,
        poly: Option<DenseMultilinearPolynomial<F>>,
        c_1: FieldElement<F>,
    ) -> Self {
        Self {
            n,
            c_1,
            round: 0,
            poly,
            last_val: FieldElement::zero(),
            challenges: Vec::with_capacity(n),
        }
    }

    /// Executes round `j` of the verifier.
    pub fn do_round<C: Channel<F>>(
        &mut self,
        univar: Polynomial<FieldElement<F>>,
        channel: &mut C,
    ) -> Result<VerifierRoundResult<F>, VerifierError<F>> {
        // Evaluate polynomial at 0 and 1 once, reusing the values.
        let eval_0 = univar.evaluate(&FieldElement::<F>::zero());
        let eval_1 = univar.evaluate(&FieldElement::<F>::one());

        if self.round == 0 {
            // Check intermediate consistency for round 0: s0 + s1 must equal c_1.
            if &eval_0 + &eval_1 != self.c_1 {
                return Err(VerifierError::InconsistentSum {
                    round: self.round,
                    s0: eval_0,
                    s1: eval_1,
                    expected: self.c_1.clone(),
                });
            }
        } else {
            let sum = &eval_0 + &eval_1;
            // Check intermediate consistency: s0 + s1 must equal last_val.
            if sum != self.last_val {
                return Err(VerifierError::InconsistentSum {
                    round: self.round,
                    s0: eval_0,
                    s1: eval_1,
                    expected: self.last_val.clone(),
                });
            }
        }

        // Append the field element to the channel.
        channel.append_felt(&univar.coefficients[0]);

        // Draw a random challenge for the round.
        let base_challenge = channel.draw_felt();
        let r_j = &base_challenge + FieldElement::<F>::from(self.round as u64);

        self.challenges.push(r_j.clone());
        // Evaluate polynomial at the challenge.
        let val = univar.evaluate(&r_j);
        self.last_val = val;
        self.round += 1;

        if self.round == self.n {
            // Final round
            if let Some(ref poly) = self.poly {
                let full_point = self.challenges.clone();
                if let Ok(real_val) = poly.evaluate(full_point) {
                    return Ok(VerifierRoundResult::Final(real_val == self.last_val));
                } else {
                    return Err(VerifierError::OracleEvaluationError);
                }
            }
            Ok(VerifierRoundResult::Final(true))
        } else {
            Ok(VerifierRoundResult::NextRound(r_j))
        }
    }
}

/// Verify a complete Sumcheck proof.
///
/// # Arguments
/// * `n` - Number of variables in the polynomial
/// * `claimed_sum` - The claimed sum of all evaluations
/// * `proof_polys` - A vector of univariate polynomials, one for each round
/// * `oracle_poly` - Optional original polynomial used for the final verification
///
/// # Returns
/// * `true` if the proof is valid, `false` otherwise
pub fn verify<F: IsField>(
    n: usize,
    claimed_sum: FieldElement<F>,
    proof_polys: Vec<Polynomial<FieldElement<F>>>,
    oracle_poly: Option<DenseMultilinearPolynomial<F>>,
) -> Result<bool, VerifierError<F>>
where
    <F as IsField>::BaseType: Send + Sync,
    FieldElement<F>: ByteConversion,
{
    let mut verifier = Verifier::new(n, oracle_poly, claimed_sum);
    let mut transcript = DefaultTranscript::<F>::default();

    for (i, univar) in proof_polys.into_iter().enumerate() {
        match verifier.do_round(univar, &mut transcript)? {
            VerifierRoundResult::NextRound(_) => {
                // Continue to next round
                if i == n - 1 {
                    return Err(VerifierError::OracleEvaluationError);
                }
            }
            VerifierRoundResult::Final(result) => {
                return Ok(result);
            }
        }
    }

    // This should not happen if the proof has the correct number of rounds
    Err(VerifierError::OracleEvaluationError)
}
