use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::{HasDefaultTranscript, IsField};
use lambdaworks_math::polynomial::Polynomial;
use lambdaworks_math::traits::ByteConversion;

use lambdaworks_crypto::fiat_shamir::is_transcript::IsTranscript;

use crate::utils::random_linear_combination;

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

/// Proves a batch of sumcheck instances with a single shared proof.
///
/// Combines multiple oracles into one via random linear combination with `alpha`.
/// Oracles can have different numbers of variables — smaller ones are scaled by a
/// "doubling factor" `2^(max_vars - oracle_vars)` and produce a constant polynomial
/// `claim / 2` until their variables begin.
///
/// Returns `(proof, assignment, constant_oracles, final_claims)`.
#[allow(clippy::type_complexity)]
pub fn prove_batch<F, O, T>(
    mut claims: Vec<FieldElement<F>>,
    mut oracles: Vec<O>,
    alpha: &FieldElement<F>,
    channel: &mut T,
) -> (
    SumcheckProof<F>,
    Vec<FieldElement<F>>,
    Vec<O>,
    Vec<FieldElement<F>>,
)
where
    F: IsField + HasDefaultTranscript,
    FieldElement<F>: ByteConversion,
    O: SumcheckOracle<F>,
    T: IsTranscript<F>,
{
    let n_variables = oracles.iter().map(|o| o.n_variables()).max().unwrap();
    assert_eq!(claims.len(), oracles.len());

    let mut round_polys = Vec::new();
    let mut assignment = Vec::new();

    // Scale claims by doubling factor for smaller instances.
    let two = &FieldElement::<F>::one() + &FieldElement::<F>::one();
    for (claim, oracle) in claims.iter_mut().zip(oracles.iter()) {
        let n_unused = n_variables - oracle.n_variables();
        for _ in 0..n_unused {
            *claim = &*claim * &two;
        }
    }

    for round in 0..n_variables {
        let n_remaining = n_variables - round;

        // Compute per-oracle round polynomials.
        let this_round_polys: Vec<Polynomial<FieldElement<F>>> = oracles
            .iter()
            .zip(claims.iter())
            .map(|(oracle, claim)| {
                if n_remaining == oracle.n_variables() {
                    oracle.sum_as_poly_in_first_variable(claim)
                } else {
                    // Oracle hasn't started yet: constant polynomial = claim / 2.
                    let half_claim = claim * two.inv().unwrap();
                    Polynomial::new(&[half_claim])
                }
            })
            .collect();

        // Combine with alpha via random linear combination of polynomials.
        let combined = poly_random_linear_combination(&this_round_polys, alpha);

        // Sanity check.
        debug_assert_eq!(
            combined.evaluate(&FieldElement::<F>::zero())
                + combined.evaluate(&FieldElement::<F>::one()),
            random_linear_combination(&claims, alpha)
        );
        debug_assert!(combined.degree() <= MAX_DEGREE);

        // Send combined polynomial to verifier.
        for coeff in combined.coefficients() {
            channel.append_field_element(coeff);
        }
        let challenge: FieldElement<F> = channel.sample_field_element();

        // Update per-oracle claims.
        claims = this_round_polys
            .iter()
            .map(|p| p.evaluate(&challenge))
            .collect();

        // Fix first variable on active oracles.
        oracles = oracles
            .into_iter()
            .map(|oracle| {
                if n_remaining != oracle.n_variables() {
                    return oracle;
                }
                oracle.fix_first_variable(&challenge)
            })
            .collect();

        round_polys.push(combined);
        assignment.push(challenge);
    }

    let proof = SumcheckProof { round_polys };
    (proof, assignment, oracles, claims)
}

/// Random linear combination of polynomials: `p_0 + alpha * p_1 + ... + alpha^(n-1) * p_{n-1}`.
fn poly_random_linear_combination<F: IsField>(
    polys: &[Polynomial<FieldElement<F>>],
    alpha: &FieldElement<F>,
) -> Polynomial<FieldElement<F>> {
    polys.iter().rev().fold(
        Polynomial::new(&[FieldElement::<F>::zero()]),
        |acc, poly| acc * Polynomial::new(std::slice::from_ref(alpha)) + poly.clone(),
    )
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
                return Polynomial::new(&[self.mle[0]]);
            }
            let half = self.mle.len() / 2;
            let sum_at_0: FE = (0..half).map(|j| self.mle[j]).sum();
            let sum_at_1: FE = (half..self.mle.len()).map(|j| self.mle[j]).sum();
            let diff = sum_at_1 - sum_at_0;
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
        let (proof, assignment, _constant_oracle) = prove(claim, oracle, &mut prover_channel);

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
        let (mut proof, _, _) = prove(claim, oracle, &mut prover_channel);

        proof.round_polys[0] = Polynomial::new(&[FE::from(999), FE::from(1)]);

        let mut verifier_channel = DefaultTranscript::<F>::new(&[]);
        let result = partially_verify(claim, &proof, &mut verifier_channel);
        assert!(result.is_err());
    }

    #[test]
    fn prove_batch_single_oracle_roundtrip() {
        let evals: Vec<FE> = (1u64..=8).map(FE::from).collect();
        let claim: FE = evals.iter().cloned().sum();
        let oracle = TestMleOracle {
            mle: Mle::new(evals),
        };
        let alpha = FE::from(7);

        let mut prover_channel = DefaultTranscript::<F>::new(&[]);
        let (proof, assignment, _constant_oracles, _final_claims) =
            prove_batch(vec![claim], vec![oracle], &alpha, &mut prover_channel);

        assert_eq!(proof.round_polys.len(), 3);
        assert_eq!(assignment.len(), 3);

        // Verify: for a single oracle, the combined claim equals the original claim.
        let mut verifier_channel = DefaultTranscript::<F>::new(&[]);
        let result = partially_verify(claim, &proof, &mut verifier_channel);
        assert!(result.is_ok());

        let (v_assignment, _) = result.unwrap();
        assert_eq!(v_assignment, assignment);
    }

    #[test]
    fn prove_batch_two_oracles_same_size() {
        let evals0: Vec<FE> = (1u64..=8).map(FE::from).collect();
        let evals1: Vec<FE> = (11u64..=18).map(FE::from).collect();
        let claim0: FE = evals0.iter().cloned().sum();
        let claim1: FE = evals1.iter().cloned().sum();
        let alpha = FE::from(13);

        let oracle0 = TestMleOracle {
            mle: Mle::new(evals0),
        };
        let oracle1 = TestMleOracle {
            mle: Mle::new(evals1),
        };

        let mut prover_channel = DefaultTranscript::<F>::new(&[]);
        let (proof, _assignment, _constant_oracles, _final_claims) = prove_batch(
            vec![claim0, claim1],
            vec![oracle0, oracle1],
            &alpha,
            &mut prover_channel,
        );

        assert_eq!(proof.round_polys.len(), 3);

        // Combined claim = claim0 + alpha * claim1
        let combined_claim = claim0 + (alpha * claim1);
        let mut verifier_channel = DefaultTranscript::<F>::new(&[]);
        let result = partially_verify(combined_claim, &proof, &mut verifier_channel);
        assert!(result.is_ok());
    }

    #[test]
    fn prove_batch_different_sizes() {
        // oracle0 has 3 variables (8 evals), oracle1 has 2 variables (4 evals)
        let evals0: Vec<FE> = (1u64..=8).map(FE::from).collect();
        let evals1: Vec<FE> = (1u64..=4).map(FE::from).collect();
        let claim0: FE = evals0.iter().cloned().sum();
        let claim1: FE = evals1.iter().cloned().sum();
        let alpha = FE::from(5);

        let oracle0 = TestMleOracle {
            mle: Mle::new(evals0),
        };
        let oracle1 = TestMleOracle {
            mle: Mle::new(evals1),
        };

        let mut prover_channel = DefaultTranscript::<F>::new(&[]);
        let (proof, assignment, _constant_oracles, _final_claims) = prove_batch(
            vec![claim0, claim1],
            vec![oracle0, oracle1],
            &alpha,
            &mut prover_channel,
        );

        // Max variables = 3, so 3 rounds.
        assert_eq!(proof.round_polys.len(), 3);
        assert_eq!(assignment.len(), 3);

        // Doubling factor for oracle1: 2^(3-2) = 2
        let two = FE::from(2);
        let combined_claim = claim0 + (alpha * (claim1 * two));
        let mut verifier_channel = DefaultTranscript::<F>::new(&[]);
        let result = partially_verify(combined_claim, &proof, &mut verifier_channel);
        assert!(result.is_ok());
    }
}
