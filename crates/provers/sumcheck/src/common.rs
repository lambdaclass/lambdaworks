//! Common utilities for sumcheck provers
//!
//! This module provides shared functionality used across different prover implementations,
//! reducing code duplication and ensuring consistent behavior.

use crate::prover::{ProofResult, ProverError};
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

/// Validates that all polynomial factors have the same number of variables.
///
/// Returns the number of variables if valid, or an error if the factors are invalid.
pub fn validate_factors<F: IsField>(
    factors: &[DenseMultilinearPolynomial<F>],
) -> Result<usize, ProverError>
where
    F::BaseType: Send + Sync,
{
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

    Ok(num_vars)
}

/// Validates that num_vars is positive.
pub fn validate_num_vars(num_vars: usize) -> Result<(), ProverError> {
    if num_vars == 0 {
        return Err(ProverError::FactorMismatch(
            "Number of variables must be at least 1.".to_string(),
        ));
    }
    Ok(())
}

/// Checks if all rounds have been completed.
pub fn check_round_bounds(current_round: usize, num_vars: usize) -> Result<(), ProverError> {
    if current_round >= num_vars {
        return Err(ProverError::InvalidState(
            "All variables already fixed, no more rounds to run.".to_string(),
        ));
    }
    Ok(())
}

/// Trait for provers that can execute the sumcheck protocol.
///
/// This trait defines the common interface for all sumcheck prover implementations,
/// enabling shared proof generation logic.
pub trait SumcheckProver<F: IsField>
where
    F::BaseType: Send + Sync,
    FieldElement<F>: Clone + Mul<Output = FieldElement<F>>,
{
    /// Returns the number of variables in the polynomial(s).
    fn num_vars(&self) -> usize;

    /// Returns the number of polynomial factors (for product sumcheck).
    fn num_factors(&self) -> usize;

    /// Computes the initial claimed sum over the boolean hypercube.
    fn compute_initial_sum(&self) -> FieldElement<F>;

    /// Executes a single round of the sumcheck protocol.
    ///
    /// Takes the challenge from the previous round (None for the first round)
    /// and returns the round polynomial.
    fn round(
        &mut self,
        r_prev: Option<&FieldElement<F>>,
    ) -> Result<Polynomial<FieldElement<F>>, ProverError>;
}

/// Runs the sumcheck protocol using the provided prover and returns the proof.
///
/// This function handles all transcript operations, eliminating duplicate code
/// across different prover implementations.
pub fn run_sumcheck_protocol<F, P>(
    prover: &mut P,
    num_factors: usize,
) -> Result<ProofResult<F>, ProverError>
where
    F: IsField + HasDefaultTranscript,
    F::BaseType: Send + Sync,
    FieldElement<F>: Clone + Mul<Output = FieldElement<F>> + ByteConversion,
    P: SumcheckProver<F>,
{
    let num_vars = prover.num_vars();
    let claimed_sum = prover.compute_initial_sum();

    let mut transcript = DefaultTranscript::<F>::default();
    initialize_transcript(&mut transcript, num_vars, num_factors, &claimed_sum);

    let mut proof_polys = Vec::with_capacity(num_vars);
    let mut current_challenge: Option<FieldElement<F>> = None;

    for j in 0..num_vars {
        let g_j = prover.round(current_challenge.as_ref())?;

        append_round_poly_to_transcript(&mut transcript, j, &g_j);

        proof_polys.push(g_j);

        if j < num_vars - 1 {
            current_challenge = Some(transcript.draw_felt());
        } else {
            current_challenge = None;
        }
    }

    Ok((claimed_sum, proof_polys))
}

/// Initializes the transcript with the protocol parameters.
fn initialize_transcript<F>(
    transcript: &mut DefaultTranscript<F>,
    num_vars: usize,
    num_factors: usize,
    claimed_sum: &FieldElement<F>,
) where
    F: IsField + HasDefaultTranscript,
    FieldElement<F>: ByteConversion,
{
    transcript.append_bytes(b"initial_sum");
    transcript.append_felt(&FieldElement::from(num_vars as u64));
    transcript.append_felt(&FieldElement::from(num_factors as u64));
    transcript.append_felt(claimed_sum);
}

/// Appends a round polynomial to the transcript.
fn append_round_poly_to_transcript<F>(
    transcript: &mut DefaultTranscript<F>,
    round: usize,
    poly: &Polynomial<FieldElement<F>>,
) where
    F: IsField + HasDefaultTranscript,
    FieldElement<F>: ByteConversion,
{
    let round_label = format!("round_{round}_poly");
    transcript.append_bytes(round_label.as_bytes());

    let coeffs = poly.coefficients();
    transcript.append_bytes(&(coeffs.len() as u64).to_be_bytes());
    if coeffs.is_empty() {
        transcript.append_felt(&FieldElement::zero());
    } else {
        for coeff in coeffs {
            transcript.append_felt(coeff);
        }
    }
}

/// Applies a challenge to evaluation vectors, performing the "fix first variable" operation.
///
/// This computes: new[k] = (1-r) * old[k] + r * old[k + half]
/// which is equivalent to fixing the first variable to r.
pub fn apply_challenge_to_evals<F: IsField>(evals: &mut Vec<FieldElement<F>>, r: &FieldElement<F>)
where
    F::BaseType: Send + Sync,
    FieldElement<F>: Clone + Mul<Output = FieldElement<F>>,
{
    let half = evals.len() / 2;
    let one_minus_r = FieldElement::<F>::one() - r;

    for k in 0..half {
        let v0 = &evals[k];
        let v1 = &evals[k + half];
        evals[k] = &one_minus_r * v0 + r * v1;
    }

    evals.truncate(half);
}

/// Computes the sums at X=0 and X=1 for the round polynomial (single factor).
///
/// For a single polynomial, g(0) is the sum of the first half of evaluations,
/// and g(1) is the sum of the second half.
pub fn compute_round_sums_single<F: IsField>(
    evals: &[FieldElement<F>],
) -> (FieldElement<F>, FieldElement<F>)
where
    F::BaseType: Send + Sync,
    FieldElement<F>: Clone,
{
    let half = evals.len() / 2;
    let mut sum_0 = FieldElement::zero();
    let mut sum_1 = FieldElement::zero();

    for k in 0..half {
        sum_0 += evals[k].clone();
        sum_1 += evals[k + half].clone();
    }

    (sum_0, sum_1)
}

/// Computes the round polynomial evaluations for a product of factors.
///
/// The degree of the round polynomial is at most the number of factors,
/// so we need num_factors + 1 evaluation points.
pub fn compute_round_poly_product<F: IsField>(
    factor_evals: &[Vec<FieldElement<F>>],
) -> Vec<FieldElement<F>>
where
    F::BaseType: Send + Sync,
    FieldElement<F>: Clone + Mul<Output = FieldElement<F>>,
{
    let num_factors = factor_evals.len();
    let num_eval_points = num_factors + 1;
    let half = factor_evals[0].len() / 2;

    let mut evaluations = vec![FieldElement::zero(); num_eval_points];

    for (t, eval) in evaluations.iter_mut().enumerate().take(num_eval_points) {
        let t_fe: FieldElement<F> = FieldElement::from(t as u64);
        let mut sum = FieldElement::zero();

        for k in 0..half {
            let mut product = FieldElement::<F>::one();

            for factor_eval in factor_evals {
                let p0 = &factor_eval[k];
                let p1 = &factor_eval[k + half];
                let interpolated = p0 + &t_fe * &(p1 - p0);
                product *= interpolated;
            }

            sum += product;
        }

        *eval = sum;
    }

    evaluations
}

/// Computes the initial sum of products over the boolean hypercube.
pub fn compute_initial_sum_product<F: IsField>(
    factor_evals: &[Vec<FieldElement<F>>],
) -> FieldElement<F>
where
    F::BaseType: Send + Sync,
    FieldElement<F>: Clone + Mul<Output = FieldElement<F>>,
{
    let size = factor_evals[0].len();
    let mut sum = FieldElement::zero();

    for j in 0..size {
        let mut product: FieldElement<F> = FieldElement::one();
        for factor_eval in factor_evals {
            product *= factor_eval[j].clone();
        }
        sum += product;
    }

    sum
}

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Parallel threshold - use parallel algorithms when size exceeds this
#[cfg(feature = "parallel")]
pub const PARALLEL_THRESHOLD: usize = 1024;

/// Applies a challenge to evaluation vectors in parallel.
#[cfg(feature = "parallel")]
pub fn apply_challenge_to_evals_parallel<F: IsField>(
    evals: &mut Vec<FieldElement<F>>,
    r: &FieldElement<F>,
) where
    F::BaseType: Send + Sync,
    FieldElement<F>: Clone + Mul<Output = FieldElement<F>> + Send + Sync,
{
    let half = evals.len() / 2;

    if half < PARALLEL_THRESHOLD {
        apply_challenge_to_evals(evals, r);
        return;
    }

    let one_minus_r = FieldElement::<F>::one() - r;

    let new_evals: Vec<FieldElement<F>> = (0..half)
        .into_par_iter()
        .map(|k| {
            let v0 = &evals[k];
            let v1 = &evals[k + half];
            &one_minus_r * v0 + r * v1
        })
        .collect();

    *evals = new_evals;
}

/// Computes round sums in parallel (single factor).
#[cfg(feature = "parallel")]
pub fn compute_round_sums_single_parallel<F: IsField>(
    evals: &[FieldElement<F>],
) -> (FieldElement<F>, FieldElement<F>)
where
    F::BaseType: Send + Sync,
    FieldElement<F>: Clone + Send + Sync,
{
    let half = evals.len() / 2;

    if half < PARALLEL_THRESHOLD {
        return compute_round_sums_single(evals);
    }

    let sum_0: FieldElement<F> = (0..half)
        .into_par_iter()
        .map(|k| evals[k].clone())
        .reduce(FieldElement::zero, |a, b| a + b);

    let sum_1: FieldElement<F> = (0..half)
        .into_par_iter()
        .map(|k| evals[k + half].clone())
        .reduce(FieldElement::zero, |a, b| a + b);

    (sum_0, sum_1)
}

/// Computes the initial sum in parallel (single factor).
#[cfg(feature = "parallel")]
pub fn compute_initial_sum_parallel<F: IsField>(evals: &[FieldElement<F>]) -> FieldElement<F>
where
    F::BaseType: Send + Sync,
    FieldElement<F>: Clone + Send + Sync,
{
    if evals.len() < PARALLEL_THRESHOLD {
        return evals
            .iter()
            .cloned()
            .fold(FieldElement::zero(), |a, b| a + b);
    }

    evals
        .par_iter()
        .cloned()
        .reduce(FieldElement::zero, |a, b| a + b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use lambdaworks_math::field::fields::u64_prime_field::U64PrimeField;

    const MODULUS: u64 = 101;
    type F = U64PrimeField<MODULUS>;
    type FE = FieldElement<F>;

    #[test]
    fn test_validate_factors_empty() {
        let factors: Vec<DenseMultilinearPolynomial<F>> = vec![];
        assert!(validate_factors(&factors).is_err());
    }

    #[test]
    fn test_validate_factors_valid() {
        let poly1 = DenseMultilinearPolynomial::new(vec![FE::from(1), FE::from(2)]);
        let poly2 = DenseMultilinearPolynomial::new(vec![FE::from(3), FE::from(4)]);
        let factors = vec![poly1, poly2];
        assert_eq!(validate_factors(&factors).unwrap(), 1);
    }

    #[test]
    fn test_validate_factors_mismatch() {
        let poly1 = DenseMultilinearPolynomial::new(vec![FE::from(1), FE::from(2)]);
        let poly2 = DenseMultilinearPolynomial::new(vec![
            FE::from(1),
            FE::from(2),
            FE::from(3),
            FE::from(4),
        ]);
        let factors = vec![poly1, poly2];
        assert!(validate_factors(&factors).is_err());
    }

    #[test]
    fn test_apply_challenge_to_evals() {
        let mut evals = vec![FE::from(1), FE::from(2), FE::from(3), FE::from(4)];
        let r = FE::from(2);

        apply_challenge_to_evals(&mut evals, &r);

        // new[0] = (1-2)*1 + 2*3 = -1 + 6 = 5 (mod 101)
        // new[1] = (1-2)*2 + 2*4 = -2 + 8 = 6 (mod 101)
        assert_eq!(evals.len(), 2);
        assert_eq!(evals[0], FE::from(5));
        assert_eq!(evals[1], FE::from(6));
    }

    #[test]
    fn test_compute_round_sums_single() {
        let evals = vec![FE::from(1), FE::from(2), FE::from(3), FE::from(4)];
        let (sum_0, sum_1) = compute_round_sums_single(&evals);

        // sum_0 = 1 + 2 = 3
        // sum_1 = 3 + 4 = 7
        assert_eq!(sum_0, FE::from(3));
        assert_eq!(sum_1, FE::from(7));
    }

    #[test]
    fn test_compute_initial_sum_product() {
        let factor1 = vec![FE::from(1), FE::from(2)];
        let factor2 = vec![FE::from(3), FE::from(4)];
        let factor_evals = vec![factor1, factor2];

        let sum = compute_initial_sum_product(&factor_evals);

        // 1*3 + 2*4 = 3 + 8 = 11
        assert_eq!(sum, FE::from(11));
    }
}
