use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::{HasDefaultTranscript, IsField};
use lambdaworks_math::polynomial::Polynomial;
use lambdaworks_math::traits::ByteConversion;

use lambdaworks_crypto::fiat_shamir::is_transcript::IsTranscript;

/// Max degree of polynomials the verifier accepts in each sumcheck round.
pub const MAX_DEGREE: usize = 3;

/// An oracle for a multivariate polynomial that supports the sumcheck protocol.
///
/// Unlike the sumcheck in `crates/provers/sumcheck/` which works with explicit
/// `Vec<DenseMultilinearPolynomial>`, this trait uses an oracle abstraction that
/// lazily computes the sum polynomial from `eq_evals` + layer data + `lambda`.
pub trait SumcheckOracle<F: IsField>: Sized {
    /// Number of remaining variables.
    fn n_variables(&self) -> usize;

    /// Computes the univariate polynomial `f(t) = sum_x g(t, x)` for all `x` in `{0,1}^(n-1)`.
    ///
    /// `claim` equals `f(0) + f(1)`, which can be used to derive `f(1)` from `f(0)`.
    fn sum_as_poly_in_first_variable(&self, claim: &FieldElement<F>)
        -> Polynomial<FieldElement<F>>;

    /// Fixes the first variable to `challenge`, returning a new oracle with one fewer variable.
    fn fix_first_variable(self, challenge: &FieldElement<F>) -> Self;
}

/// Proof for the sumcheck protocol: one round polynomial per variable.
#[derive(Debug, Clone)]
pub struct SumcheckProof<F: IsField> {
    pub round_polys: Vec<Polynomial<FieldElement<F>>>,
}

/// Sumcheck protocol error.
#[derive(Debug)]
pub enum SumcheckError<F: IsField> {
    /// Polynomial degree exceeds MAX_DEGREE.
    DegreeInvalid { round: usize },
    /// `f(0) + f(1) != claim` in the given round.
    SumInvalid {
        claim: FieldElement<F>,
        sum: FieldElement<F>,
        round: usize,
    },
}

impl<F: IsField> core::fmt::Display for SumcheckError<F> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::DegreeInvalid { round } => {
                write!(f, "degree of polynomial in round {round} is too high")
            }
            Self::SumInvalid { round, .. } => {
                write!(f, "sum does not match claim in round {round}")
            }
        }
    }
}

/// Proves `sum_{x in {0,1}^n} g(x) = claim` using the sumcheck protocol.
///
/// Returns `(proof, variable_assignment, constant_oracle)` where `constant_oracle`
/// has all variables fixed and can be used to extract the mask.
pub fn prove<F, O, T>(
    mut claim: FieldElement<F>,
    mut oracle: O,
    channel: &mut T,
) -> (SumcheckProof<F>, Vec<FieldElement<F>>, O)
where
    F: IsField + HasDefaultTranscript,
    FieldElement<F>: ByteConversion,
    O: SumcheckOracle<F>,
    T: IsTranscript<F>,
{
    let n_variables = oracle.n_variables();
    let mut round_polys = Vec::with_capacity(n_variables);
    let mut assignment = Vec::with_capacity(n_variables);

    for _round in 0..n_variables {
        let round_poly = oracle.sum_as_poly_in_first_variable(&claim);

        // Sanity check: f(0) + f(1) == claim
        debug_assert_eq!(
            round_poly.evaluate(&FieldElement::<F>::zero())
                + round_poly.evaluate(&FieldElement::<F>::one()),
            claim
        );
        debug_assert!(round_poly.degree() <= MAX_DEGREE);

        // Send round polynomial to verifier (via channel)
        for coeff in round_poly.coefficients() {
            channel.append_field_element(coeff);
        }

        let challenge: FieldElement<F> = channel.sample_field_element();

        claim = round_poly.evaluate(&challenge);
        oracle = oracle.fix_first_variable(&challenge);

        round_polys.push(round_poly);
        assignment.push(challenge);
    }

    let proof = SumcheckProof { round_polys };
    (proof, assignment, oracle)
}

/// Result of partial sumcheck verification: `(variable_assignment, claimed_eval)`.
pub type SumcheckResult<F> = (Vec<FieldElement<F>>, FieldElement<F>);

/// Partially verifies a sumcheck proof.
///
/// Checks that `f(0) + f(1) == claim` at each round and that the degree is valid.
/// Returns `(variable_assignment, claimed_eval)` — the claimed evaluation must be
/// checked externally against the actual gate computation.
pub fn partially_verify<F, T>(
    mut claim: FieldElement<F>,
    proof: &SumcheckProof<F>,
    channel: &mut T,
) -> Result<SumcheckResult<F>, SumcheckError<F>>
where
    F: IsField + HasDefaultTranscript,
    FieldElement<F>: ByteConversion,
    T: IsTranscript<F>,
{
    let mut assignment = Vec::new();

    for (round, round_poly) in proof.round_polys.iter().enumerate() {
        if round_poly.degree() > MAX_DEGREE {
            return Err(SumcheckError::DegreeInvalid { round });
        }

        let sum = round_poly.evaluate(&FieldElement::<F>::zero())
            + round_poly.evaluate(&FieldElement::<F>::one());

        if claim != sum {
            return Err(SumcheckError::SumInvalid { claim, sum, round });
        }

        for coeff in round_poly.coefficients() {
            channel.append_field_element(coeff);
        }
        let challenge: FieldElement<F> = channel.sample_field_element();

        claim = round_poly.evaluate(&challenge);
        assignment.push(challenge);
    }

    Ok((assignment, claim))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mle::Mle;
    use lambdaworks_crypto::fiat_shamir::default_transcript::DefaultTranscript;
    use lambdaworks_math::field::fields::u64_prime_field::U64PrimeField;

    const MODULUS: u64 = 2013265921;
    type F = U64PrimeField<MODULUS>;
    type FE = FieldElement<F>;

    struct TestMleOracle {
        mle: Mle<F>,
    }

    impl SumcheckOracle<F> for TestMleOracle {
        fn n_variables(&self) -> usize {
            self.mle.n_variables()
        }

        fn sum_as_poly_in_first_variable(&self, _claim: &FE) -> Polynomial<FE> {
            let n = self.mle.n_variables();
            if n == 0 {
                return Polynomial::new(&[self.mle[0].clone()]);
            }
            let half = self.mle.len() / 2;
            let sum_at_0: FE = (0..half).map(|j| self.mle[j].clone()).sum();
            let sum_at_1: FE = (half..self.mle.len()).map(|j| self.mle[j].clone()).sum();
            let diff = &sum_at_1 - &sum_at_0;
            Polynomial::new(&[sum_at_0, diff])
        }

        fn fix_first_variable(mut self, challenge: &FE) -> Self {
            self.mle.fix_first_variable(challenge);
            self
        }
    }

    #[test]
    fn prove_verify_roundtrip() {
        let evals: Vec<FE> = (1u64..=8).map(FE::from).collect();
        let claim: FE = evals.iter().cloned().sum();
        let oracle = TestMleOracle {
            mle: Mle::new(evals.clone()),
        };

        let mut prover_channel = DefaultTranscript::<F>::new(&[]);
        let (proof, assignment, _constant_oracle) =
            prove(claim.clone(), oracle, &mut prover_channel);

        assert_eq!(proof.round_polys.len(), 3);
        assert_eq!(assignment.len(), 3);

        let mut verifier_channel = DefaultTranscript::<F>::new(&[]);
        let result = partially_verify(claim, &proof, &mut verifier_channel);
        assert!(result.is_ok());

        let (v_assignment, _claimed_eval) = result.unwrap();
        assert_eq!(v_assignment, assignment);
    }

    #[test]
    fn corrupt_proof_rejected() {
        let evals: Vec<FE> = (1u64..=8).map(FE::from).collect();
        let claim: FE = evals.iter().cloned().sum();
        let oracle = TestMleOracle {
            mle: Mle::new(evals),
        };

        let mut prover_channel = DefaultTranscript::<F>::new(&[]);
        let (mut proof, _, _) = prove(claim.clone(), oracle, &mut prover_channel);

        proof.round_polys[0] = Polynomial::new(&[FE::from(999), FE::from(1)]);

        let mut verifier_channel = DefaultTranscript::<F>::new(&[]);
        let result = partially_verify(claim, &proof, &mut verifier_channel);
        assert!(result.is_err());
    }
}
