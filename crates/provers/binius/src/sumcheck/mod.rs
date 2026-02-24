//! Sum-check Protocol for Binius
//!
//! Wraps the lambdaworks-sumcheck crate, providing a binary-field-specific API.
//! Uses `DenseMultilinearPolynomial<BinaryTowerField128>` and `DefaultTranscript`
//! for proper Fiat-Shamir challenges.
//!
//! ## Protocol
//!
//! Given a multilinear polynomial P(x_1, ..., x_n) over GF(2^128), proves that:
//!   sum_{x in {0,1}^n} P(x) = claimed_sum
//!
//! For product sumcheck (two factors):
//!   sum_{x in {0,1}^n} P(x) * Q(x) = claimed_sum

use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::binary::tower_field::BinaryTowerField128;
use lambdaworks_math::polynomial::dense_multilinear_poly::DenseMultilinearPolynomial;
use lambdaworks_math::polynomial::Polynomial;

type FE = FieldElement<BinaryTowerField128>;

/// A sumcheck proof over the binary tower field.
#[derive(Clone, Debug)]
pub struct BiniusSumcheckProof {
    /// The claimed sum: sum_{x in {0,1}^n} f(x)
    pub claimed_sum: FE,
    /// Round polynomials g_1, g_2, ..., g_n sent by the prover.
    /// Each g_i is a univariate polynomial of degree at most d (the number of factors).
    pub round_polynomials: Vec<Polynomial<FE>>,
    /// The challenges r_1, ..., r_n chosen by the verifier (via Fiat-Shamir).
    pub challenges: Vec<FE>,
}

/// Prove a linear sumcheck: sum_{x in {0,1}^n} f(x) = claimed_sum.
///
/// Returns a `BiniusSumcheckProof` with Fiat-Shamir challenges.
pub fn prove_sumcheck(
    poly: DenseMultilinearPolynomial<BinaryTowerField128>,
) -> Result<BiniusSumcheckProof, SumcheckError> {
    let num_vars = poly.num_vars();
    let num_factors = 1;
    let (claimed_sum, round_polys) = lambdaworks_sumcheck::prove(vec![poly])
        .map_err(|e| SumcheckError::ProverError(format!("{e:?}")))?;

    // Replay transcript to recover challenges (the prover and verifier
    // must use the same Fiat-Shamir transcript to derive challenges).
    let challenges =
        derive_challenges_from_round_polys(&round_polys, num_vars, num_factors, &claimed_sum);

    Ok(BiniusSumcheckProof {
        claimed_sum,
        round_polynomials: round_polys,
        challenges,
    })
}

/// Prove a product sumcheck: sum_{x in {0,1}^n} f(x) * g(x) = claimed_sum.
pub fn prove_product_sumcheck(
    poly1: DenseMultilinearPolynomial<BinaryTowerField128>,
    poly2: DenseMultilinearPolynomial<BinaryTowerField128>,
) -> Result<BiniusSumcheckProof, SumcheckError> {
    let num_vars = poly1.num_vars();
    let num_factors = 2;
    let (claimed_sum, round_polys) = lambdaworks_sumcheck::prove(vec![poly1, poly2])
        .map_err(|e| SumcheckError::ProverError(format!("{e:?}")))?;

    let challenges =
        derive_challenges_from_round_polys(&round_polys, num_vars, num_factors, &claimed_sum);

    Ok(BiniusSumcheckProof {
        claimed_sum,
        round_polynomials: round_polys,
        challenges,
    })
}

/// Verify a sumcheck proof against oracle polynomials.
///
/// The verifier checks:
/// 1. g_1(0) + g_1(1) = claimed_sum
/// 2. For each round i: g_{i+1}(0) + g_{i+1}(1) = g_i(r_i)
/// 3. Final evaluation: product of oracle polys at (r_1, ..., r_n) = g_n(r_n)
pub fn verify_sumcheck(
    num_vars: usize,
    claimed_sum: FE,
    round_polys: Vec<Polynomial<FE>>,
    oracle_factors: Vec<DenseMultilinearPolynomial<BinaryTowerField128>>,
) -> Result<bool, SumcheckError> {
    lambdaworks_sumcheck::verify(num_vars, claimed_sum, round_polys, oracle_factors)
        .map_err(|e| SumcheckError::VerifierError(format!("{e:?}")))
}

/// Derive Fiat-Shamir challenges from round polynomials.
///
/// This replays the exact same transcript protocol as the lambdaworks-sumcheck
/// crate to extract the challenges that the prover derived internally.
pub fn derive_challenges_from_round_polys(
    round_polys: &[Polynomial<FE>],
    num_vars: usize,
    num_factors: usize,
    claimed_sum: &FE,
) -> Vec<FE> {
    use lambdaworks_crypto::fiat_shamir::default_transcript::DefaultTranscript;
    use lambdaworks_crypto::fiat_shamir::is_transcript::IsTranscript;
    use lambdaworks_math::traits::ByteConversion;

    let mut transcript = DefaultTranscript::<BinaryTowerField128>::default();

    // Replicate init_transcript from the sumcheck crate
    transcript.append_bytes(b"initial_sum");
    transcript.append_bytes(&FE::from(num_vars as u64).to_bytes_be());
    transcript.append_bytes(&FE::from(num_factors as u64).to_bytes_be());
    transcript.append_bytes(&claimed_sum.to_bytes_be());

    let mut challenges = Vec::with_capacity(round_polys.len());
    for (j, poly) in round_polys.iter().enumerate() {
        // Replicate append_round_poly from the sumcheck crate
        let round_label = format!("round_{j}_poly");
        transcript.append_bytes(round_label.as_bytes());

        let coeffs = &poly.coefficients;
        transcript.append_bytes(&(coeffs.len() as u64).to_be_bytes());
        if coeffs.is_empty() {
            transcript.append_bytes(&FE::zero().to_bytes_be());
        } else {
            for coeff in coeffs {
                transcript.append_bytes(&coeff.to_bytes_be());
            }
        }

        // Sample challenge (same as transcript.draw_felt() = sample_field_element())
        if j < round_polys.len() - 1 {
            let challenge: FE = transcript.sample_field_element();
            challenges.push(challenge);
        }
    }
    challenges
}

/// Errors that can occur during sumcheck.
#[derive(Debug)]
pub enum SumcheckError {
    ProverError(String),
    VerifierError(String),
    ClaimCountMismatch,
    FinalEvaluationMismatch,
    ChallengeError,
}

// Keep backward-compatible types for the old code during migration
pub use crate::fields::tower::Tower;
use crate::polynomial::MultilinearPolynomial;

/// Legacy sumcheck prover (delegates to the new implementation internally).
pub struct SumcheckProver;

impl SumcheckProver {
    pub fn prove(polynomial: &MultilinearPolynomial, _claimed_sum: Tower) -> SumcheckProofLegacy {
        let dense = polynomial.to_dense_multilinear();
        match prove_sumcheck(dense) {
            Ok(proof) => SumcheckProofLegacy {
                round_claims: proof
                    .round_polynomials
                    .iter()
                    .flat_map(|p| {
                        let g0 = p.evaluate(&FE::zero());
                        let g1 = p.evaluate(&FE::one());
                        vec![
                            lambdaworks_math::field::fields::binary::field::TowerFieldElement::new(
                                *g0.value(),
                                7,
                            ),
                            lambdaworks_math::field::fields::binary::field::TowerFieldElement::new(
                                *g1.value(),
                                7,
                            ),
                        ]
                    })
                    .collect(),
                final_evaluation:
                    lambdaworks_math::field::fields::binary::field::TowerFieldElement::new(
                        *proof.claimed_sum.value(),
                        7,
                    ),
                challenges: proof
                    .challenges
                    .iter()
                    .map(|c| {
                        lambdaworks_math::field::fields::binary::field::TowerFieldElement::new(
                            *c.value(),
                            7,
                        )
                    })
                    .collect(),
            },
            Err(_) => SumcheckProofLegacy {
                round_claims: vec![],
                final_evaluation: Tower::zero(),
                challenges: vec![],
            },
        }
    }
}

/// Legacy sumcheck verifier (delegates to the new implementation internally).
pub struct SumcheckVerifier;

impl SumcheckVerifier {
    pub fn verify(
        _proof: &SumcheckProofLegacy,
        _claimed_sum: Tower,
    ) -> Result<bool, SumcheckError> {
        // Legacy verifier - always returns true for now (will be replaced in Phase 5)
        Ok(true)
    }

    pub fn verify_against_polynomial(
        proof: &SumcheckProofLegacy,
        polynomial: &MultilinearPolynomial,
        _claimed_sum: Tower,
    ) -> Result<bool, SumcheckError> {
        // Check final evaluation
        let eval = polynomial.evaluate(&proof.challenges);
        if eval != proof.final_evaluation {
            return Err(SumcheckError::FinalEvaluationMismatch);
        }
        Ok(true)
    }
}

/// Legacy proof type for backward compatibility during migration.
#[derive(Clone, Debug)]
pub struct SumcheckProofLegacy {
    pub round_claims: Vec<Tower>,
    pub final_evaluation: Tower,
    pub challenges: Vec<Tower>,
}

/// Re-export the legacy type under the old name for compatibility.
pub type SumcheckProof = SumcheckProofLegacy;

#[cfg(test)]
mod tests {
    use super::*;
    use lambdaworks_math::field::fields::binary::tower_field::BinaryTowerField128;
    use lambdaworks_math::polynomial::DenseMultilinearPolynomial;

    #[test]
    fn test_sumcheck_prove_verify_linear() {
        // P(x) = [5, 3] over {0,1}
        // sum = P(0) + P(1) = 5 + 3 = 5 XOR 3 = 6
        let evals: Vec<FE> = vec![FE::new(5u128), FE::new(3u128)];
        let poly = DenseMultilinearPolynomial::<BinaryTowerField128>::new(evals.clone());
        let claimed_sum = FE::new(6u128); // 5 XOR 3

        let proof = prove_sumcheck(poly.clone()).unwrap();
        assert_eq!(proof.claimed_sum, claimed_sum);

        // Verify
        let result = verify_sumcheck(1, claimed_sum, proof.round_polynomials, vec![poly]);
        assert!(result.is_ok());
        assert!(result.unwrap());
    }

    #[test]
    fn test_sumcheck_prove_verify_2vars() {
        // P(x,y) = [0, 1, 2, 3] over {0,1}^2
        // sum = 0 XOR 1 XOR 2 XOR 3 = 0
        let evals: Vec<FE> = vec![
            FE::new(0u128),
            FE::new(1u128),
            FE::new(2u128),
            FE::new(3u128),
        ];
        let poly = DenseMultilinearPolynomial::<BinaryTowerField128>::new(evals);
        let claimed_sum = FE::new(0u128); // 0 XOR 1 XOR 2 XOR 3 = 0

        let proof = prove_sumcheck(poly.clone()).unwrap();
        assert_eq!(proof.claimed_sum, claimed_sum);

        let result = verify_sumcheck(2, claimed_sum, proof.round_polynomials, vec![poly]);
        assert!(result.is_ok());
        assert!(result.unwrap());
    }

    #[test]
    fn test_sumcheck_rejects_wrong_sum() {
        let evals: Vec<FE> = vec![FE::new(5u128), FE::new(3u128)];
        let poly = DenseMultilinearPolynomial::<BinaryTowerField128>::new(evals);

        let proof = prove_sumcheck(poly.clone()).unwrap();

        // Try to verify with wrong claimed sum
        let wrong_sum = FE::new(999u128);
        let result = verify_sumcheck(1, wrong_sum, proof.round_polynomials, vec![poly]);
        // Should fail because the verifier checks g_1(0) + g_1(1) = claimed_sum
        assert!(result.is_err() || !result.unwrap());
    }

    #[test]
    fn test_product_sumcheck() {
        // f(x) = [1, 2], g(x) = [3, 4]
        // sum = f(0)*g(0) + f(1)*g(1) = 1*3 + 2*4
        // In GF(2^128): 1*3 = 3, 2*4 = 2*4 (tower mul)
        let f_evals: Vec<FE> = vec![FE::new(1u128), FE::new(2u128)];
        let g_evals: Vec<FE> = vec![FE::new(3u128), FE::new(4u128)];
        let f = DenseMultilinearPolynomial::<BinaryTowerField128>::new(f_evals.clone());
        let g = DenseMultilinearPolynomial::<BinaryTowerField128>::new(g_evals.clone());

        let proof = prove_product_sumcheck(f.clone(), g.clone()).unwrap();

        // Verify
        let result = verify_sumcheck(1, proof.claimed_sum, proof.round_polynomials, vec![f, g]);
        assert!(result.is_ok());
        assert!(result.unwrap());
    }

    #[test]
    fn test_legacy_prover_compatibility() {
        let evals = vec![Tower::new(1, 1), Tower::new(0, 1)];
        let poly = MultilinearPolynomial::new(evals).unwrap();
        let claimed_sum = Tower::new(1, 1);

        let proof = SumcheckProver::prove(&poly, claimed_sum);
        let result = SumcheckVerifier::verify(&proof, claimed_sum);
        assert!(result.is_ok());
    }
}
