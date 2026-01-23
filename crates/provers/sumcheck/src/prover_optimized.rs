//! Optimized Sumcheck Prover using streaming algorithm
//!
//! This implementation uses the VSBW13 streaming algorithm which achieves O(2^n) time
//! complexity instead of the naive O(n · 2^2n) approach. The key insight is to maintain
//! working copies of polynomial evaluations and update them in-place after each round.

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

/// Optimized Prover for the Sum-Check protocol using streaming algorithm.
///
/// Instead of re-evaluating polynomials at every point in the hypercube for each round,
/// this prover maintains working copies of evaluations and updates them in-place.
///
/// Time complexity: O(d · 2^n) where d is the number of factors and n is the number of variables.
/// Space complexity: O(d · 2^n) for storing the working copies.
pub struct OptimizedProver<F: IsField>
where
    F::BaseType: Send + Sync,
{
    num_vars: usize,
    /// Working copies of factor evaluations, updated in-place each round
    factor_evals: Vec<Vec<FieldElement<F>>>,
    /// Current round (0-indexed)
    current_round: usize,
}

impl<F: IsField> OptimizedProver<F>
where
    F::BaseType: Send + Sync,
    FieldElement<F>: Clone + Mul<Output = FieldElement<F>>,
{
    /// Creates a new OptimizedProver instance from the given factors.
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

        // Clone evaluations into working buffers
        let factor_evals: Vec<Vec<FieldElement<F>>> = factors
            .iter()
            .map(|f| f.evals().clone())
            .collect();

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

    /// Computes the initial claimed sum over the boolean hypercube.
    /// Sum = Σ_{x ∈ {0,1}^n} Π_i P_i(x)
    pub fn compute_initial_sum(&self) -> FieldElement<F> {
        let size = 1 << self.num_vars;
        let mut sum = FieldElement::zero();

        for j in 0..size {
            let mut product = FieldElement::one();
            for factor_eval in &self.factor_evals {
                product = product * factor_eval[j].clone();
            }
            sum = sum + product;
        }

        sum
    }

    /// Executes a round of the Sum-Check protocol and returns the round polynomial.
    ///
    /// The round polynomial g_j(X) is computed by:
    /// g_j(X) = Σ_{x_{j+1},...,x_n ∈ {0,1}} Π_i P_i(r_1,...,r_{j-1}, X, x_{j+1},...,x_n)
    ///
    /// After computing g_j, if a challenge r_j is provided, the working evaluations
    /// are updated in-place to prepare for the next round.
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

        // Current size of evaluation vectors (halves each round)
        let current_size = 1 << (self.num_vars - self.current_round);
        let half = current_size / 2;
        let num_factors = self.factor_evals.len();

        // Compute round polynomial by evaluating at 0, 1, ..., num_factors
        // The degree is at most num_factors, so we need num_factors + 1 points
        let num_eval_points = num_factors + 1;
        let mut evaluations = vec![FieldElement::zero(); num_eval_points];

        // For each evaluation point t in {0, 1, ..., num_factors}
        for t in 0..num_eval_points {
            let t_fe = FieldElement::from(t as u64);
            let mut sum = FieldElement::zero();

            // Sum over all (b_{j+1}, ..., b_n) ∈ {0,1}^{n-j-1}
            for k in 0..half {
                let mut product = FieldElement::one();

                for factor_eval in &self.factor_evals {
                    // Interpolate: P(t) = P(0) + t * (P(1) - P(0))
                    // where P(0) = factor_eval[k] and P(1) = factor_eval[k + half]
                    let p0 = &factor_eval[k];
                    let p1 = &factor_eval[k + half];
                    let interpolated = p0 + &t_fe * &(p1 - p0);
                    product = product * interpolated;
                }

                sum = sum + product;
            }

            evaluations[t] = sum;
        }

        // Interpolate to get the polynomial
        let eval_points: Vec<FieldElement<F>> = (0..num_eval_points)
            .map(|i| FieldElement::from(i as u64))
            .collect();

        let poly = Polynomial::interpolate(&eval_points, &evaluations)?;

        self.current_round += 1;

        Ok(poly)
    }

    /// Applies a challenge to update the working evaluations in-place.
    ///
    /// This performs the "fix first variable" operation on all factor evaluations,
    /// reducing the evaluation count by half.
    fn apply_challenge(&mut self, r: &FieldElement<F>) -> Result<(), ProverError> {
        if self.current_round == 0 {
            return Err(ProverError::InvalidState(
                "Cannot apply challenge before first round.".to_string(),
            ));
        }

        let current_size = 1 << (self.num_vars - self.current_round + 1);
        let half = current_size / 2;

        for factor_eval in &mut self.factor_evals {
            // Update in-place: new[k] = old[k] + r * (old[k + half] - old[k])
            for k in 0..half {
                let p0 = &factor_eval[k];
                let p1 = &factor_eval[k + half];
                factor_eval[k] = p0 + r * &(p1 - p0);
            }
            // Truncate to the new size
            factor_eval.truncate(half);
        }

        Ok(())
    }

    /// Returns the final evaluations at the challenge point (for verification).
    pub fn get_final_evaluations(&self) -> Vec<FieldElement<F>> {
        self.factor_evals.iter().map(|e| e[0].clone()).collect()
    }
}

/// Proves a sumcheck using the optimized streaming algorithm.
///
/// This is a drop-in replacement for `prove` with significantly better performance.
pub fn prove_optimized<F>(factors: Vec<DenseMultilinearPolynomial<F>>) -> ProverOutput<F>
where
    F: IsField + HasDefaultTranscript,
    F::BaseType: Send + Sync,
    FieldElement<F>: Clone + Mul<Output = FieldElement<F>> + ByteConversion,
{
    let mut prover = OptimizedProver::new(factors.clone())?;
    let num_vars = prover.num_vars();

    // Compute the claimed sum
    let claimed_sum = prover.compute_initial_sum();

    // Initialize Fiat-Shamir transcript
    let mut transcript = DefaultTranscript::<F>::default();
    transcript.append_bytes(b"initial_sum");
    transcript.append_felt(&FieldElement::from(num_vars as u64));
    transcript.append_felt(&FieldElement::from(factors.len() as u64));
    transcript.append_felt(&claimed_sum);

    let mut proof_polys = Vec::with_capacity(num_vars);
    let mut current_challenge: Option<FieldElement<F>> = None;

    // Execute rounds
    for j in 0..num_vars {
        let g_j = prover.round(current_challenge.as_ref())?;

        // Append g_j to transcript
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

        // Derive challenge for the next round
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
    fn test_optimized_prover_linear() {
        let poly = DenseMultilinearPolynomial::new(vec![
            FE::from(3),
            FE::from(5),
            FE::from(7),
            FE::from(11),
        ]);
        let num_vars = poly.num_vars();

        let (claimed_sum, proof_polys) = prove_optimized(vec![poly.clone()]).unwrap();

        // Verify using the standard verifier
        let result = crate::verify(num_vars, claimed_sum, proof_polys, vec![poly]);
        assert!(result.unwrap_or(false), "Valid proof should be accepted");
    }

    #[test]
    fn test_optimized_prover_quadratic() {
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
            prove_optimized(vec![poly_a.clone(), poly_b.clone()]).unwrap();

        let result = crate::verify(num_vars, claimed_sum, proof_polys, vec![poly_a, poly_b]);
        assert!(result.unwrap_or(false), "Quadratic proof should be accepted");
    }

    #[test]
    fn test_optimized_prover_cubic() {
        let poly_a = DenseMultilinearPolynomial::new(vec![FE::from(1), FE::from(2)]);
        let poly_b = DenseMultilinearPolynomial::new(vec![FE::from(3), FE::from(4)]);
        let poly_c = DenseMultilinearPolynomial::new(vec![FE::from(5), FE::from(1)]);
        let num_vars = poly_a.num_vars();

        let (claimed_sum, proof_polys) =
            prove_optimized(vec![poly_a.clone(), poly_b.clone(), poly_c.clone()]).unwrap();

        let result =
            crate::verify(num_vars, claimed_sum, proof_polys, vec![poly_a, poly_b, poly_c]);
        assert!(result.unwrap_or(false), "Cubic proof should be accepted");
    }

    #[test]
    fn test_optimized_matches_original() {
        // Test that optimized prover produces the same claimed sum as original
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

        let (optimized_sum, _) = prove_optimized(vec![poly.clone()]).unwrap();
        let (original_sum, _) = crate::prove(vec![poly]).unwrap();

        assert_eq!(
            optimized_sum, original_sum,
            "Optimized and original should produce same sum"
        );
    }

    #[test]
    fn test_initial_sum_computation() {
        // Manual computation: sum over {0,1}^2 of polynomial with evals [1, 2, 3, 4]
        // = 1 + 2 + 3 + 4 = 10
        let poly = DenseMultilinearPolynomial::new(vec![
            FE::from(1),
            FE::from(2),
            FE::from(3),
            FE::from(4),
        ]);

        let prover = OptimizedProver::new(vec![poly]).unwrap();
        let sum = prover.compute_initial_sum();

        assert_eq!(sum, FE::from(10));
    }

    #[test]
    fn test_product_initial_sum() {
        // Sum over {0,1}^1 of P1 * P2 where P1 = [1, 2], P2 = [3, 4]
        // = P1(0)*P2(0) + P1(1)*P2(1) = 1*3 + 2*4 = 11
        let poly1 = DenseMultilinearPolynomial::new(vec![FE::from(1), FE::from(2)]);
        let poly2 = DenseMultilinearPolynomial::new(vec![FE::from(3), FE::from(4)]);

        let prover = OptimizedProver::new(vec![poly1, poly2]).unwrap();
        let sum = prover.compute_initial_sum();

        assert_eq!(sum, FE::from(11));
    }
}
