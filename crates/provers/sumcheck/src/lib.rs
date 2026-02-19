pub mod batched;
pub mod blendy;
pub mod common;
pub mod prover;
pub mod prover_optimized;
pub mod prover_parallel;
pub mod small_field;
pub mod sparse_prover;
pub mod verifier;

#[cfg(feature = "metal")]
pub mod gpu;

#[cfg(feature = "metal")]
pub mod metal;

#[cfg(test)]
mod correctness_tests;

use lambdaworks_crypto::fiat_shamir::default_transcript::DefaultTranscript;
use lambdaworks_crypto::fiat_shamir::is_transcript::IsTranscript;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::{HasDefaultTranscript, IsField};
use lambdaworks_math::polynomial::dense_multilinear_poly::DenseMultilinearPolynomial;
use lambdaworks_math::polynomial::error::MultilinearError;
use lambdaworks_math::polynomial::Polynomial;
use lambdaworks_math::traits::ByteConversion;
use std::ops::Mul;
use thiserror::Error;

pub use batched::{
    prove_batched, verify_batched, BatchedProofOutput, BatchedProver, StructuredBatchedProver,
};
pub use blendy::{prove_blendy, prove_memory_efficient, BlendyProver};
pub use common::{run_sumcheck_with_channel, SumcheckProver};
pub use prover::ProverOutput;
pub use prover::{prove, Prover, ProverError};
pub use prover_optimized::{prove_optimized, OptimizedProver, OptimizedProverError};
pub use prover_parallel::{prove_fast, prove_parallel, FastProver, ParallelProver};
pub use small_field::{prove_small_field, SmallFieldProver};
pub use sparse_prover::{prove_sparse, SparseEntry, SparseMultiFactorProver, SparseProver};
pub use verifier::{
    partially_verify, verify, PartialVerifyError, Verifier, VerifierError, VerifierRoundResult,
};

#[cfg(feature = "metal")]
pub use metal::{prove_metal, prove_metal_multi, MetalMultiFactorProver, MetalProver, MetalState};

#[cfg(all(target_os = "macos", feature = "metal"))]
pub use metal::GoldilocksMetalProver;

/// Error type for evaluation operations in sumcheck
#[derive(Debug, Error)]
pub enum EvaluationError {
    #[error("Cannot evaluate product of zero factors")]
    EmptyFactors,
    #[error("Point length {point_len} does not match polynomial num_vars {num_vars}")]
    PointLengthMismatch { point_len: usize, num_vars: usize },
    #[error("Multilinear polynomial evaluation failed: {0:?}")]
    MultilinearError(MultilinearError),
}

impl From<MultilinearError> for EvaluationError {
    fn from(e: MultilinearError) -> Self {
        EvaluationError::MultilinearError(e)
    }
}

// Wrappers for the prover and verifier functions
pub fn prove_linear<F>(poly: DenseMultilinearPolynomial<F>) -> ProverOutput<F>
where
    F: IsField + HasDefaultTranscript,
    F::BaseType: Send + Sync,
    FieldElement<F>: Clone + Mul<Output = FieldElement<F>> + ByteConversion,
{
    prove(vec![poly])
}

pub fn verify_linear<F>(
    num_vars: usize,
    claimed_sum: FieldElement<F>,
    proof_polys: Vec<Polynomial<FieldElement<F>>>,
    oracle_poly: DenseMultilinearPolynomial<F>,
) -> Result<bool, VerifierError<F>>
where
    F: IsField + HasDefaultTranscript,
    F::BaseType: Send + Sync,
    FieldElement<F>: Clone + Mul<Output = FieldElement<F>> + ByteConversion,
{
    verify(num_vars, claimed_sum, proof_polys, vec![oracle_poly])
}
pub fn prove_quadratic<F>(
    poly1: DenseMultilinearPolynomial<F>,
    poly2: DenseMultilinearPolynomial<F>,
) -> ProverOutput<F>
where
    F: IsField + HasDefaultTranscript,
    F::BaseType: Send + Sync,
    FieldElement<F>: Clone + Mul<Output = FieldElement<F>> + ByteConversion,
{
    prove(vec![poly1, poly2])
}

pub fn verify_quadratic<F>(
    num_vars: usize,
    claimed_sum: FieldElement<F>,
    proof_polys: Vec<Polynomial<FieldElement<F>>>,
    oracle_poly1: DenseMultilinearPolynomial<F>,
    oracle_poly2: DenseMultilinearPolynomial<F>,
) -> Result<bool, VerifierError<F>>
where
    F: IsField + HasDefaultTranscript,
    F::BaseType: Send + Sync,
    FieldElement<F>: Clone + Mul<Output = FieldElement<F>> + ByteConversion,
{
    verify(
        num_vars,
        claimed_sum,
        proof_polys,
        vec![oracle_poly1, oracle_poly2],
    )
}

pub fn prove_cubic<F>(
    poly1: DenseMultilinearPolynomial<F>,
    poly2: DenseMultilinearPolynomial<F>,
    poly3: DenseMultilinearPolynomial<F>,
) -> ProverOutput<F>
where
    F: IsField + HasDefaultTranscript,
    F::BaseType: Send + Sync,
    FieldElement<F>: Clone + Mul<Output = FieldElement<F>> + ByteConversion,
{
    prove(vec![poly1, poly2, poly3])
}

pub fn prove_linear_optimized<F>(poly: DenseMultilinearPolynomial<F>) -> ProverOutput<F>
where
    F: IsField + HasDefaultTranscript,
    F::BaseType: Send + Sync,
    FieldElement<F>: Clone + Mul<Output = FieldElement<F>> + ByteConversion,
{
    prove_optimized(vec![poly])
}

pub fn prove_quadratic_optimized<F>(
    poly1: DenseMultilinearPolynomial<F>,
    poly2: DenseMultilinearPolynomial<F>,
) -> ProverOutput<F>
where
    F: IsField + HasDefaultTranscript,
    F::BaseType: Send + Sync,
    FieldElement<F>: Clone + Mul<Output = FieldElement<F>> + ByteConversion,
{
    if poly1.num_vars() != poly2.num_vars() {
        return Err(ProverError::FactorMismatch(
            "Polynomials must have the same number of variables for quadratic prove.".to_string(),
        ));
    }
    prove_optimized(vec![poly1, poly2])
}

pub fn prove_cubic_optimized<F>(
    poly1: DenseMultilinearPolynomial<F>,
    poly2: DenseMultilinearPolynomial<F>,
    poly3: DenseMultilinearPolynomial<F>,
) -> ProverOutput<F>
where
    F: IsField + HasDefaultTranscript,
    F::BaseType: Send + Sync,
    FieldElement<F>: Clone + Mul<Output = FieldElement<F>> + ByteConversion,
{
    if poly1.num_vars() != poly2.num_vars() || poly1.num_vars() != poly3.num_vars() {
        return Err(ProverError::FactorMismatch(
            "Polynomials must have the same number of variables for cubic prove.".to_string(),
        ));
    }
    prove_optimized(vec![poly1, poly2, poly3])
}

pub fn verify_cubic<F>(
    num_vars: usize,
    claimed_sum: FieldElement<F>,
    proof_polys: Vec<Polynomial<FieldElement<F>>>,
    oracle_poly1: DenseMultilinearPolynomial<F>,
    oracle_poly2: DenseMultilinearPolynomial<F>,
    oracle_poly3: DenseMultilinearPolynomial<F>,
) -> Result<bool, VerifierError<F>>
where
    F: IsField + HasDefaultTranscript,
    F::BaseType: Send + Sync,
    FieldElement<F>: Clone + Mul<Output = FieldElement<F>> + ByteConversion,
{
    verify(
        num_vars,
        claimed_sum,
        proof_polys,
        vec![oracle_poly1, oracle_poly2, oracle_poly3],
    )
}

pub trait Channel<F: IsField> {
    fn append_felt(&mut self, element: &FieldElement<F>);
    fn draw_felt(&mut self) -> FieldElement<F>;
}

/// Initialize a Fiat-Shamir transcript with the sumcheck parameters.
pub(crate) fn init_transcript<F>(
    transcript: &mut DefaultTranscript<F>,
    num_vars: usize,
    num_factors: usize,
    claimed_sum: &FieldElement<F>,
) where
    F: HasDefaultTranscript,
    FieldElement<F>: ByteConversion,
{
    transcript.append_bytes(b"initial_sum");
    transcript.append_bytes(&FieldElement::<F>::from(num_vars as u64).to_bytes_be());
    transcript.append_bytes(&FieldElement::<F>::from(num_factors as u64).to_bytes_be());
    transcript.append_bytes(&claimed_sum.to_bytes_be());
}

/// Append a round polynomial to the transcript.
pub(crate) fn append_round_poly<F>(
    transcript: &mut DefaultTranscript<F>,
    round: usize,
    g_j: &Polynomial<FieldElement<F>>,
) where
    F: HasDefaultTranscript,
    FieldElement<F>: ByteConversion,
{
    let round_label = format!("round_{round}_poly");
    transcript.append_bytes(round_label.as_bytes());

    let coeffs = g_j.coefficients();
    transcript.append_bytes(&(coeffs.len() as u64).to_be_bytes());
    if coeffs.is_empty() {
        transcript.append_bytes(&FieldElement::<F>::zero().to_bytes_be());
    } else {
        for coeff in coeffs {
            transcript.append_bytes(&coeff.to_bytes_be());
        }
    }
}

impl<F> Channel<F> for DefaultTranscript<F>
where
    F: HasDefaultTranscript,
    FieldElement<F>: ByteConversion,
{
    fn append_felt(&mut self, element: &FieldElement<F>) {
        self.append_bytes(&element.to_bytes_be());
    }

    fn draw_felt(&mut self) -> FieldElement<F> {
        self.sample_field_element()
    }
}

/// Evaluate the product of multiple multilinear polynomials at a point.
///
/// Computes the Lagrange basis (chi) vector once and reuses it for all factors,
/// avoiding redundant O(2^n) work per factor.
pub fn evaluate_product_at_point<F: IsField>(
    factors: &[DenseMultilinearPolynomial<F>],
    point: &[FieldElement<F>],
) -> Result<FieldElement<F>, EvaluationError>
where
    F::BaseType: Send + Sync,
    FieldElement<F>: Clone + Mul<Output = FieldElement<F>>,
{
    if factors.is_empty() {
        return Err(EvaluationError::EmptyFactors);
    }
    let num_vars = factors[0].num_vars();
    if num_vars != point.len() {
        return Err(EvaluationError::PointLengthMismatch {
            point_len: point.len(),
            num_vars,
        });
    }

    // Compute Lagrange basis (chi) vector once — O(2^n)
    let len = 1 << num_vars;
    let mut chis = vec![FieldElement::<F>::one(); len];
    let mut size = 1;
    for r_j in point {
        size *= 2;
        for i in (0..size).rev().step_by(2) {
            let half_i = i / 2;
            let temp = chis[half_i].clone() * r_j.clone();
            chis[i] = temp.clone();
            chis[i - 1] = chis[half_i].clone() - temp;
        }
    }

    // Evaluate each factor using the shared chi vector — O(2^n) per factor
    let mut product = FieldElement::<F>::one();
    for factor in factors {
        let evals = factor.evals();
        if evals.len() != len {
            return Err(EvaluationError::PointLengthMismatch {
                point_len: point.len(),
                num_vars: factor.num_vars(),
            });
        }
        let eval: FieldElement<F> = (0..len)
            .map(|i| evals[i].clone() * chis[i].clone())
            .fold(FieldElement::zero(), |acc, x| acc + x);
        product *= eval;
    }
    Ok(product)
}

#[cfg(test)]
mod tests {
    use super::*;
    use lambdaworks_math::field::fields::u64_prime_field::U64PrimeField;
    use lambdaworks_math::polynomial::dense_multilinear_poly::DenseMultilinearPolynomial;

    const MODULUS: u64 = 101;
    type F = U64PrimeField<MODULUS>;
    type FE = FieldElement<F>;

    #[test]
    fn test_sumcheck_linear() {
        let poly = DenseMultilinearPolynomial::new(vec![
            FE::from(3),
            FE::from(5),
            FE::from(7),
            FE::from(11),
        ]);
        let num_vars = poly.num_vars();
        let (claimed_sum, proof_polys) = prove_linear(poly.clone()).unwrap();
        let result = verify_linear(num_vars, claimed_sum, proof_polys, poly);
        assert!(result.unwrap_or(false), "Valid proof should be accepted");
    }

    #[test]
    fn test_interactive_sumcheck() {
        let poly = DenseMultilinearPolynomial::new(vec![
            FE::from(1),
            FE::from(2),
            FE::from(1),
            FE::from(4),
        ]);
        let num_vars = poly.num_vars();
        let mut prover = Prover::new(vec![poly.clone()]).unwrap();
        let c_1 = prover.compute_initial_sum().unwrap();
        assert_eq!(c_1, FE::from(8));

        let mut transcript = DefaultTranscript::<F>::default();
        let mut verifier = Verifier::new(num_vars, vec![poly.clone()], c_1).unwrap();

        // Round 0
        let g0 = prover.round(None).unwrap();
        let res0 = verifier.do_round(g0, &mut transcript).unwrap();
        let r0 = if let VerifierRoundResult::NextRound(chal) = res0 {
            chal
        } else {
            panic!("Expected NextRound result for round 0");
        };

        // Round 1 (Final)
        let g1 = prover.round(Some(&r0)).unwrap();
        let res1 = verifier.do_round(g1, &mut transcript).unwrap();
        if let VerifierRoundResult::Final(ok) = res1 {
            assert!(ok, "Final round verification failed");
        } else {
            panic!("Expected Final result for round 1");
        }
    }

    #[test]
    fn test_from_book() {
        let poly = DenseMultilinearPolynomial::new(vec![
            FE::from(1),
            FE::from(2),
            FE::from(3),
            FE::from(4),
            FE::from(5),
            FE::from(6),
            FE::from(7),
            FE::from(8),
        ]);
        let num_vars = poly.num_vars();
        assert_eq!(num_vars, 3);
        let mut prover = Prover::new(vec![poly.clone()]).unwrap();
        let c_1 = prover.compute_initial_sum().unwrap();
        assert_eq!(c_1, FE::from(36));

        let mut transcript = DefaultTranscript::<F>::default();
        let mut verifier = Verifier::new(num_vars, vec![poly.clone()], c_1).unwrap();
        let mut current_challenge: Option<FieldElement<F>> = None;

        for round_idx in 0..num_vars {
            let g_j = prover.round(current_challenge.as_ref()).unwrap();
            let res = verifier.do_round(g_j, &mut transcript).unwrap();
            match res {
                VerifierRoundResult::NextRound(chal) => {
                    current_challenge = Some(chal);
                }
                VerifierRoundResult::Final(ok) => {
                    assert!(ok, "Final round verification failed");
                    assert_eq!(round_idx, num_vars - 1, "Final result occurred too early");
                    break;
                }
            }
        }
    }

    #[test]
    fn test_from_book_ported() {
        let poly = DenseMultilinearPolynomial::new(vec![
            FE::from(0),
            FE::from(2),
            FE::from(0),
            FE::from(2),
            FE::from(0),
            FE::from(3),
            FE::from(1),
            FE::from(4),
        ]);
        let num_vars = poly.num_vars();
        assert_eq!(num_vars, 3);
        let mut prover = Prover::new(vec![poly.clone()]).unwrap();
        let c_1 = prover.compute_initial_sum().unwrap();
        assert_eq!(c_1, FE::from(12));

        let mut transcript = DefaultTranscript::<F>::default();
        let mut verifier = Verifier::new(num_vars, vec![poly.clone()], c_1).unwrap();

        // Round 0
        let g0 = prover.round(None).unwrap();
        let res0 = verifier.do_round(g0, &mut transcript).unwrap();
        let r0 = if let VerifierRoundResult::NextRound(chal) = res0 {
            chal
        } else {
            panic!("Expected NextRound result for round 0");
        };

        // Round 1
        let g1 = prover.round(Some(&r0)).unwrap();
        let res1 = verifier.do_round(g1, &mut transcript).unwrap();
        let r1 = if let VerifierRoundResult::NextRound(chal) = res1 {
            chal
        } else {
            panic!("Expected NextRound result for round 1");
        };

        // Round 2 (final)
        let g2 = prover.round(Some(&r1)).unwrap();
        let res2 = verifier.do_round(g2, &mut transcript).unwrap();
        if let VerifierRoundResult::Final(ok) = res2 {
            assert!(ok, "Final round verification failed");
        } else {
            panic!("Expected Final result for round 2");
        }
    }

    #[test]
    fn failing_verification_test() {
        let poly = DenseMultilinearPolynomial::new(vec![
            FE::from(1),
            FE::from(2),
            FE::from(1),
            FE::from(4),
        ]);
        let num_vars = poly.num_vars();
        let factors = vec![poly.clone()];
        let incorrect_c1 = FE::from(999);

        let mut transcript = DefaultTranscript::<F>::default();
        let mut verifier = Verifier::new(num_vars, factors.clone(), incorrect_c1).unwrap();
        let mut correct_prover = Prover::new(factors).unwrap();
        let g0 = correct_prover.round(None).unwrap();

        let eval_0_g0 = g0.evaluate(&FE::zero());
        let eval_1_g0 = g0.evaluate(&FE::one());
        assert_eq!(eval_0_g0 + eval_1_g0, FE::from(8));

        let res0 = verifier.do_round(g0, &mut transcript);
        if let Err(VerifierError::InconsistentSum {
            round, expected, ..
        }) = &res0
        {
            assert_eq!(*round, 0, "Error should occur in round 0");
            assert_eq!(
                *expected, incorrect_c1,
                "Expected sum should be the incorrect one"
            );
        } else {
            panic!("Expected InconsistentSum error, got {res0:?}");
        }
        assert!(res0.is_err(), "Expected verification error");
    }

    #[test]
    fn test_sumcheck_quadratic() {
        let poly_a = DenseMultilinearPolynomial::new(vec![
            FE::from(1),
            FE::from(2),
            FE::from(3),
            FE::from(4),
        ]);
        let poly_b = DenseMultilinearPolynomial::new(vec![
            FE::from(5),
            FE::from(6),
            FE::from(7),
            FE::from(8),
        ]);
        let num_vars = poly_a.num_vars();
        let (claimed_sum, proof_polys) = prove_quadratic(poly_a.clone(), poly_b.clone()).unwrap();
        let result = verify_quadratic(num_vars, claimed_sum, proof_polys, poly_a, poly_b);
        assert!(result.unwrap_or(false), "Quadratic verification failed");
    }

    #[test]
    fn test_sumcheck_cubic() {
        let poly_a = DenseMultilinearPolynomial::new(vec![FE::from(1), FE::from(2)]);
        let poly_b = DenseMultilinearPolynomial::new(vec![FE::from(3), FE::from(4)]);
        let poly_c = DenseMultilinearPolynomial::new(vec![FE::from(5), FE::from(1)]);
        let num_vars = poly_a.num_vars();
        let (claimed_sum, proof_polys) =
            prove_cubic(poly_a.clone(), poly_b.clone(), poly_c.clone()).unwrap();
        let result = verify_cubic(num_vars, claimed_sum, proof_polys, poly_a, poly_b, poly_c);
        assert!(result.unwrap_or(false), "Cubic verification failed");
    }
}
