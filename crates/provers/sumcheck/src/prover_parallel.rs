//! Parallel Sumcheck Prover with optimizations
//!
//! # References
//!
//! ## Implementations Consulted
//!
//! - **arkworks/sumcheck**: <https://github.com/arkworks-rs/sumcheck>
//!   Parallel iterator patterns for multilinear evaluation
//!
//! - **microsoft/Nova**: <https://github.com/microsoft/Nova>
//!   Efficient parallelization strategies for recursive SNARKs
//!
//! - **rayon-rs/rayon**: <https://github.com/rayon-rs/rayon>
//!   Work-stealing parallelism patterns
//!
//! # Features
//!
//! 1. Rayon-based parallelization for hypercube summation
//! 2. Parallel challenge application
//! 3. Optimized memory access patterns
//!
//! Performance scales near-linearly with available CPU cores for large polynomials.

use crate::prover::{ProverError, ProverOutput};
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

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Parallel threshold - use parallel algorithms when size exceeds this
const PARALLEL_THRESHOLD: usize = 1024;

/// Parallel Prover for the Sum-Check protocol.
///
/// Uses rayon for parallel computation when the `parallel` feature is enabled.
/// Falls back to sequential computation for small inputs or when parallel is disabled.
pub struct ParallelProver<F: IsField>
where
    F::BaseType: Send + Sync,
{
    num_vars: usize,
    /// Working copies of factor evaluations, updated in-place each round
    factor_evals: Vec<Vec<FieldElement<F>>>,
    /// Current round (0-indexed)
    current_round: usize,
}

impl<F: IsField> ParallelProver<F>
where
    F::BaseType: Send + Sync,
    FieldElement<F>: Clone + Mul<Output = FieldElement<F>> + Send + Sync,
{
    /// Creates a new ParallelProver instance from the given factors.
    pub fn new(factors: Vec<DenseMultilinearPolynomial<F>>) -> Result<Self, ProverError> {
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

        let factor_evals: Vec<Vec<FieldElement<F>>> =
            factors.iter().map(|f| f.evals().clone()).collect();

        Ok(Self {
            num_vars,
            factor_evals,
            current_round: 0,
        })
    }

    /// Returns the number of variables in the polynomials.
    pub fn num_vars(&self) -> usize {
        self.num_vars
    }

    /// Computes the initial claimed sum over the boolean hypercube (parallelized).
    pub fn compute_initial_sum(&self) -> FieldElement<F> {
        let size = 1 << self.num_vars;

        #[cfg(feature = "parallel")]
        if size >= PARALLEL_THRESHOLD {
            return self.compute_initial_sum_parallel(size);
        }

        self.compute_initial_sum_sequential(size)
    }

    fn compute_initial_sum_sequential(&self, size: usize) -> FieldElement<F> {
        let mut sum = FieldElement::zero();
        for j in 0..size {
            let mut product = FieldElement::one();
            for factor_eval in &self.factor_evals {
                product *= factor_eval[j].clone();
            }
            sum += product;
        }
        sum
    }

    #[cfg(feature = "parallel")]
    fn compute_initial_sum_parallel(&self, size: usize) -> FieldElement<F> {
        (0..size)
            .into_par_iter()
            .map(|j| {
                let mut product = FieldElement::one();
                for factor_eval in &self.factor_evals {
                    product *= factor_eval[j].clone();
                }
                product
            })
            .reduce(FieldElement::zero, |a, b| a + b)
    }

    /// Executes a round of the Sum-Check protocol (parallelized).
    pub fn round(
        &mut self,
        r_prev: Option<&FieldElement<F>>,
    ) -> Result<Polynomial<FieldElement<F>>, ProverError> {
        // Apply previous challenge if provided
        if let Some(r) = r_prev {
            self.apply_challenge(r)?;
        }

        if self.current_round >= self.num_vars {
            return Err(ProverError::InvalidState(
                "All variables already fixed, no more rounds to run.".to_string(),
            ));
        }

        let current_size = 1 << (self.num_vars - self.current_round);
        let half = current_size / 2;
        let num_factors = self.factor_evals.len();
        let num_eval_points = num_factors + 1;

        #[cfg(feature = "parallel")]
        let evaluations = if half >= PARALLEL_THRESHOLD {
            self.compute_round_poly_parallel(half, num_eval_points)
        } else {
            self.compute_round_poly_sequential(half, num_eval_points)
        };

        #[cfg(not(feature = "parallel"))]
        let evaluations = self.compute_round_poly_sequential(half, num_eval_points);

        // Interpolate to get the polynomial
        let eval_points: Vec<FieldElement<F>> = (0..num_eval_points)
            .map(|i| FieldElement::from(i as u64))
            .collect();

        let poly = Polynomial::interpolate(&eval_points, &evaluations)?;

        self.current_round += 1;

        Ok(poly)
    }

    #[allow(clippy::needless_range_loop)]
    fn compute_round_poly_sequential(
        &self,
        half: usize,
        num_eval_points: usize,
    ) -> Vec<FieldElement<F>> {
        let mut evaluations = vec![FieldElement::zero(); num_eval_points];

        for t in 0..num_eval_points {
            let t_fe = FieldElement::from(t as u64);
            let mut sum = FieldElement::zero();

            for k in 0..half {
                let mut product = FieldElement::one();

                for factor_eval in &self.factor_evals {
                    let p0 = &factor_eval[k];
                    let p1 = &factor_eval[k + half];
                    let interpolated = p0 + &t_fe * &(p1 - p0);
                    product *= interpolated;
                }

                sum += product;
            }

            evaluations[t] = sum;
        }

        evaluations
    }

    #[cfg(feature = "parallel")]
    fn compute_round_poly_parallel(
        &self,
        half: usize,
        num_eval_points: usize,
    ) -> Vec<FieldElement<F>> {
        // For each evaluation point t, compute the sum in parallel
        (0..num_eval_points)
            .into_par_iter()
            .map(|t| {
                let t_fe = FieldElement::from(t as u64);

                (0..half)
                    .into_par_iter()
                    .map(|k| {
                        let mut product = FieldElement::one();

                        for factor_eval in &self.factor_evals {
                            let p0 = &factor_eval[k];
                            let p1 = &factor_eval[k + half];
                            let interpolated = p0 + &t_fe * &(p1 - p0);
                            product *= interpolated;
                        }

                        product
                    })
                    .reduce(FieldElement::zero, |a, b| a + b)
            })
            .collect()
    }

    /// Applies a challenge to update the working evaluations (parallelized).
    fn apply_challenge(&mut self, r: &FieldElement<F>) -> Result<(), ProverError> {
        if self.current_round == 0 {
            return Err(ProverError::InvalidState(
                "Cannot apply challenge before first round.".to_string(),
            ));
        }

        let current_size = 1 << (self.num_vars - self.current_round + 1);
        let half = current_size / 2;

        #[cfg(feature = "parallel")]
        if half >= PARALLEL_THRESHOLD {
            self.apply_challenge_parallel(r, half);
            return Ok(());
        }

        self.apply_challenge_sequential(r, half);
        Ok(())
    }

    fn apply_challenge_sequential(&mut self, r: &FieldElement<F>, half: usize) {
        for factor_eval in &mut self.factor_evals {
            for k in 0..half {
                let p0 = &factor_eval[k];
                let p1 = &factor_eval[k + half];
                factor_eval[k] = p0 + r * &(p1 - p0);
            }
            factor_eval.truncate(half);
        }
    }

    #[cfg(feature = "parallel")]
    fn apply_challenge_parallel(&mut self, r: &FieldElement<F>, half: usize) {
        for factor_eval in &mut self.factor_evals {
            // Compute new values in parallel
            let new_evals: Vec<FieldElement<F>> = (0..half)
                .into_par_iter()
                .map(|k| {
                    let p0 = &factor_eval[k];
                    let p1 = &factor_eval[k + half];
                    p0 + r * &(p1 - p0)
                })
                .collect();

            // Replace with new values
            factor_eval.clear();
            factor_eval.extend(new_evals);
        }
    }
}

/// Proves a sumcheck using parallel computation.
///
/// When the `parallel` feature is enabled, uses rayon for parallel computation.
/// This provides significant speedup for large polynomials (n >= 14 variables).
pub fn prove_parallel<F>(factors: Vec<DenseMultilinearPolynomial<F>>) -> ProverOutput<F>
where
    F: IsField + HasDefaultTranscript,
    F::BaseType: Send + Sync,
    FieldElement<F>: Clone + Mul<Output = FieldElement<F>> + ByteConversion + Send + Sync,
{
    let mut prover = ParallelProver::new(factors.clone())?;
    let num_vars = prover.num_vars();

    let claimed_sum = prover.compute_initial_sum();

    let mut transcript = DefaultTranscript::<F>::default();
    transcript.append_bytes(b"initial_sum");
    transcript.append_felt(&FieldElement::from(num_vars as u64));
    transcript.append_felt(&FieldElement::from(factors.len() as u64));
    transcript.append_felt(&claimed_sum);

    let mut proof_polys = Vec::with_capacity(num_vars);
    let mut current_challenge: Option<FieldElement<F>> = None;

    for j in 0..num_vars {
        let g_j = prover.round(current_challenge.as_ref())?;

        let round_label = format!("round_{j}_poly");
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

        proof_polys.push(g_j);

        if j < num_vars - 1 {
            current_challenge = Some(transcript.draw_felt());
        } else {
            current_challenge = None;
        }
    }

    Ok((claimed_sum, proof_polys))
}

// ============================================================================
// Optimized Algorithm with Precomputation
// ============================================================================

/// High-performance prover using precomputed interpolation coefficients.
///
/// This prover precomputes the interpolation differences for each factor,
/// reducing the number of field operations per hypercube point.
pub struct FastProver<F: IsField>
where
    F::BaseType: Send + Sync,
{
    num_vars: usize,
    num_factors: usize,
    /// Evaluations at x=0 for each factor: factor_evals_0[factor][k]
    factor_evals_0: Vec<Vec<FieldElement<F>>>,
    /// Differences (eval_1 - eval_0) for each factor: factor_diffs[factor][k]
    factor_diffs: Vec<Vec<FieldElement<F>>>,
    /// Current round
    current_round: usize,
}

impl<F: IsField> FastProver<F>
where
    F::BaseType: Send + Sync,
    FieldElement<F>: Clone + Mul<Output = FieldElement<F>> + Send + Sync,
{
    /// Creates a new FastProver with precomputed differences.
    pub fn new(factors: Vec<DenseMultilinearPolynomial<F>>) -> Result<Self, ProverError> {
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

        let num_factors = factors.len();
        let half = 1 << (num_vars - 1);

        // Precompute the differences for round 0
        let mut factor_evals_0 = Vec::with_capacity(num_factors);
        let mut factor_diffs = Vec::with_capacity(num_factors);

        for factor in &factors {
            let evals = factor.evals();
            let evals_0: Vec<FieldElement<F>> = (0..half).map(|k| evals[k].clone()).collect();
            let diffs: Vec<FieldElement<F>> = (0..half)
                .map(|k| &evals[k + half] - &evals[k])
                .collect();
            factor_evals_0.push(evals_0);
            factor_diffs.push(diffs);
        }

        Ok(Self {
            num_vars,
            num_factors,
            factor_evals_0,
            factor_diffs,
            current_round: 0,
        })
    }

    /// Returns the number of variables.
    pub fn num_vars(&self) -> usize {
        self.num_vars
    }

    /// Computes the initial claimed sum using precomputed values.
    pub fn compute_initial_sum(&self) -> FieldElement<F> {
        let half = 1 << (self.num_vars - 1);

        #[cfg(feature = "parallel")]
        if half >= PARALLEL_THRESHOLD {
            return self.compute_initial_sum_parallel(half);
        }

        self.compute_initial_sum_sequential(half)
    }

    fn compute_initial_sum_sequential(&self, half: usize) -> FieldElement<F> {
        let mut sum = FieldElement::zero();

        // Sum products at x_0 = 0
        for k in 0..half {
            let mut product = FieldElement::one();
            for f in 0..self.num_factors {
                product *= self.factor_evals_0[f][k].clone();
            }
            sum += product;
        }

        // Sum products at x_0 = 1
        for k in 0..half {
            let mut product = FieldElement::one();
            for f in 0..self.num_factors {
                // eval_1 = eval_0 + diff
                let eval_1 = &self.factor_evals_0[f][k] + &self.factor_diffs[f][k];
                product *= eval_1;
            }
            sum += product;
        }

        sum
    }

    #[cfg(feature = "parallel")]
    fn compute_initial_sum_parallel(&self, half: usize) -> FieldElement<F> {
        // Sum at x_0 = 0
        let sum_0: FieldElement<F> = (0..half)
            .into_par_iter()
            .map(|k| {
                let mut product = FieldElement::one();
                for f in 0..self.num_factors {
                    product *= self.factor_evals_0[f][k].clone();
                }
                product
            })
            .reduce(FieldElement::zero, |a, b| a + b);

        // Sum at x_0 = 1
        let sum_1: FieldElement<F> = (0..half)
            .into_par_iter()
            .map(|k| {
                let mut product = FieldElement::one();
                for f in 0..self.num_factors {
                    let eval_1 = &self.factor_evals_0[f][k] + &self.factor_diffs[f][k];
                    product *= eval_1;
                }
                product
            })
            .reduce(FieldElement::zero, |a, b| a + b);

        sum_0 + sum_1
    }

    /// Executes a round using the precomputed differences.
    pub fn round(
        &mut self,
        r_prev: Option<&FieldElement<F>>,
    ) -> Result<Polynomial<FieldElement<F>>, ProverError> {
        // Apply previous challenge
        if let Some(r) = r_prev {
            self.apply_challenge(r)?;
        }

        if self.current_round >= self.num_vars {
            return Err(ProverError::InvalidState(
                "All variables already fixed, no more rounds to run.".to_string(),
            ));
        }

        let half = 1 << (self.num_vars - self.current_round - 1);
        let num_eval_points = self.num_factors + 1;

        #[cfg(feature = "parallel")]
        let evaluations = if half >= PARALLEL_THRESHOLD {
            self.compute_round_parallel(half, num_eval_points)
        } else {
            self.compute_round_sequential(half, num_eval_points)
        };

        #[cfg(not(feature = "parallel"))]
        let evaluations = self.compute_round_sequential(half, num_eval_points);

        let eval_points: Vec<FieldElement<F>> = (0..num_eval_points)
            .map(|i| FieldElement::from(i as u64))
            .collect();

        let poly = Polynomial::interpolate(&eval_points, &evaluations)?;
        self.current_round += 1;

        Ok(poly)
    }

    #[allow(clippy::needless_range_loop)]
    fn compute_round_sequential(
        &self,
        half: usize,
        num_eval_points: usize,
    ) -> Vec<FieldElement<F>> {
        let mut evaluations = vec![FieldElement::zero(); num_eval_points];

        for t in 0..num_eval_points {
            let t_fe = FieldElement::from(t as u64);
            let mut sum = FieldElement::zero();

            for k in 0..half {
                let mut product = FieldElement::one();
                for f in 0..self.num_factors {
                    // P(t) = P(0) + t * (P(1) - P(0)) = eval_0 + t * diff
                    let interpolated = &self.factor_evals_0[f][k] + &t_fe * &self.factor_diffs[f][k];
                    product *= interpolated;
                }
                sum += product;
            }
            evaluations[t] = sum;
        }

        evaluations
    }

    #[cfg(feature = "parallel")]
    fn compute_round_parallel(&self, half: usize, num_eval_points: usize) -> Vec<FieldElement<F>> {
        (0..num_eval_points)
            .into_par_iter()
            .map(|t| {
                let t_fe = FieldElement::from(t as u64);
                (0..half)
                    .into_par_iter()
                    .map(|k| {
                        let mut product = FieldElement::one();
                        for f in 0..self.num_factors {
                            let interpolated =
                                &self.factor_evals_0[f][k] + &t_fe * &self.factor_diffs[f][k];
                            product *= interpolated;
                        }
                        product
                    })
                    .reduce(FieldElement::zero, |a, b| a + b)
            })
            .collect()
    }

    fn apply_challenge(&mut self, r: &FieldElement<F>) -> Result<(), ProverError> {
        if self.current_round == 0 {
            return Err(ProverError::InvalidState(
                "Cannot apply challenge before first round.".to_string(),
            ));
        }

        let current_half = 1 << (self.num_vars - self.current_round);
        let new_half = current_half / 2;

        #[cfg(feature = "parallel")]
        if new_half >= PARALLEL_THRESHOLD {
            self.apply_challenge_parallel(r, current_half, new_half);
            return Ok(());
        }

        self.apply_challenge_sequential(r, current_half, new_half);
        Ok(())
    }

    fn apply_challenge_sequential(&mut self, r: &FieldElement<F>, current_half: usize, new_half: usize) {
        for f in 0..self.num_factors {
            // First, update eval_0 values: new_eval_0[k] = eval_0[k] + r * diff[k]
            for k in 0..current_half {
                self.factor_evals_0[f][k] =
                    &self.factor_evals_0[f][k] + r * &self.factor_diffs[f][k];
            }

            // Then compute new diffs from the updated evals
            let new_diffs: Vec<FieldElement<F>> = (0..new_half)
                .map(|k| &self.factor_evals_0[f][k + new_half] - &self.factor_evals_0[f][k])
                .collect();

            self.factor_evals_0[f].truncate(new_half);
            self.factor_diffs[f] = new_diffs;
        }
    }

    #[cfg(feature = "parallel")]
    fn apply_challenge_parallel(&mut self, r: &FieldElement<F>, current_half: usize, new_half: usize) {
        for f in 0..self.num_factors {
            // Compute updated evals in parallel
            let new_evals: Vec<FieldElement<F>> = (0..current_half)
                .into_par_iter()
                .map(|k| &self.factor_evals_0[f][k] + r * &self.factor_diffs[f][k])
                .collect();

            // Compute new diffs from updated evals
            let new_diffs: Vec<FieldElement<F>> = (0..new_half)
                .into_par_iter()
                .map(|k| &new_evals[k + new_half] - &new_evals[k])
                .collect();

            self.factor_evals_0[f] = new_evals[..new_half].to_vec();
            self.factor_diffs[f] = new_diffs;
        }
    }
}

/// Proves a sumcheck using the fast prover with precomputation.
pub fn prove_fast<F>(factors: Vec<DenseMultilinearPolynomial<F>>) -> ProverOutput<F>
where
    F: IsField + HasDefaultTranscript,
    F::BaseType: Send + Sync,
    FieldElement<F>: Clone + Mul<Output = FieldElement<F>> + ByteConversion + Send + Sync,
{
    let mut prover = FastProver::new(factors.clone())?;
    let num_vars = prover.num_vars();

    let claimed_sum = prover.compute_initial_sum();

    let mut transcript = DefaultTranscript::<F>::default();
    transcript.append_bytes(b"initial_sum");
    transcript.append_felt(&FieldElement::from(num_vars as u64));
    transcript.append_felt(&FieldElement::from(factors.len() as u64));
    transcript.append_felt(&claimed_sum);

    let mut proof_polys = Vec::with_capacity(num_vars);
    let mut current_challenge: Option<FieldElement<F>> = None;

    for j in 0..num_vars {
        let g_j = prover.round(current_challenge.as_ref())?;

        let round_label = format!("round_{j}_poly");
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

        proof_polys.push(g_j);

        if j < num_vars - 1 {
            current_challenge = Some(transcript.draw_felt());
        } else {
            current_challenge = None;
        }
    }

    Ok((claimed_sum, proof_polys))
}

#[cfg(test)]
mod tests {
    use super::*;
    use lambdaworks_math::field::fields::u64_prime_field::U64PrimeField;

    const MODULUS: u64 = 101;
    type F = U64PrimeField<MODULUS>;
    type FE = FieldElement<F>;

    #[test]
    fn test_parallel_prover_linear() {
        let poly = DenseMultilinearPolynomial::new(vec![
            FE::from(3),
            FE::from(5),
            FE::from(7),
            FE::from(11),
        ]);
        let num_vars = poly.num_vars();

        let (claimed_sum, proof_polys) = prove_parallel(vec![poly.clone()]).unwrap();
        let result = crate::verify(num_vars, claimed_sum, proof_polys, vec![poly]);
        assert!(result.unwrap_or(false));
    }

    #[test]
    fn test_parallel_prover_quadratic() {
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

        let (claimed_sum, proof_polys) =
            prove_parallel(vec![poly_a.clone(), poly_b.clone()]).unwrap();
        let result = crate::verify(num_vars, claimed_sum, proof_polys, vec![poly_a, poly_b]);
        assert!(result.unwrap_or(false));
    }

    #[test]
    fn test_fast_prover_linear() {
        let poly = DenseMultilinearPolynomial::new(vec![
            FE::from(3),
            FE::from(5),
            FE::from(7),
            FE::from(11),
        ]);
        let num_vars = poly.num_vars();

        let (claimed_sum, proof_polys) = prove_fast(vec![poly.clone()]).unwrap();
        let result = crate::verify(num_vars, claimed_sum, proof_polys, vec![poly]);
        assert!(result.unwrap_or(false));
    }

    #[test]
    fn test_fast_prover_quadratic() {
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

        let (claimed_sum, proof_polys) = prove_fast(vec![poly_a.clone(), poly_b.clone()]).unwrap();
        let result = crate::verify(num_vars, claimed_sum, proof_polys, vec![poly_a, poly_b]);
        assert!(result.unwrap_or(false));
    }

    #[test]
    fn test_parallel_matches_optimized() {
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

        let (parallel_sum, _) = prove_parallel(vec![poly.clone()]).unwrap();
        let (optimized_sum, _) = crate::prove_optimized(vec![poly.clone()]).unwrap();
        let (fast_sum, _) = prove_fast(vec![poly]).unwrap();

        assert_eq!(parallel_sum, optimized_sum);
        assert_eq!(fast_sum, optimized_sum);
    }

    #[test]
    fn test_fast_prover_larger() {
        // Test with 8 variables
        let evals: Vec<FE> = (0..256).map(|i| FE::from(i as u64)).collect();
        let poly = DenseMultilinearPolynomial::new(evals);
        let num_vars = poly.num_vars();

        let (claimed_sum, proof_polys) = prove_fast(vec![poly.clone()]).unwrap();
        let result = crate::verify(num_vars, claimed_sum, proof_polys, vec![poly]);
        assert!(result.unwrap_or(false));
    }
}
