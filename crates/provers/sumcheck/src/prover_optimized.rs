//! Optimized Sumcheck Prover using streaming algorithm
//!
//! # References
//!
//! ## Paper
//!
//! **"Efficient RAM and Control Flow in Verifiable Outsourced Computation"**
//! Vu, Setty, Blumberg, and Walfish (VSBW13)
//! NDSS 2013 / ePrint 2012/303
//! <https://eprint.iacr.org/2012/303>
//!
//! ## Implementations Consulted
//!
//! - **arkworks/sumcheck**: <https://github.com/arkworks-rs/sumcheck>
//!   Reference implementation with clean API design
//!
//! - **microsoft/Spartan**: <https://github.com/microsoft/Spartan>
//!   Production-quality sumcheck with optimizations for R1CS
//!
//! - **HyperPlonk (EspressoSystems)**: <https://github.com/EspressoSystems/hyperplonk>
//!   Multilinear polynomial commitment and sumcheck integration
//!
//! # Algorithm
//!
//! The key insight is to maintain working copies of evaluations and update them
//! in-place after each round, avoiding the O(n*2^n) re-evaluation cost of the
//! naive algorithm.
//!
//! This implementation uses the VSBW13 streaming algorithm which achieves O(2^n) time
//! complexity instead of the naive O(n * 2^2n) approach. The key insight is to maintain
//! working copies of polynomial evaluations and update them in-place after each round.

use crate::common::{
    check_round_bounds, compute_initial_sum_product, compute_round_poly_product, run_sumcheck_protocol,
    validate_factors, SumcheckProver,
};
use crate::prover::{ProverError, ProverOutput};
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
        let num_vars = validate_factors(&factors)?;

        let factor_evals: Vec<Vec<FieldElement<F>>> =
            factors.iter().map(|f| f.evals().clone()).collect();

        Ok(Self {
            num_vars,
            factor_evals,
            current_round: 0,
        })
    }

    /// Returns the number of polynomial factors.
    pub fn num_factors(&self) -> usize {
        self.factor_evals.len()
    }

    /// Returns the final evaluations at the challenge point (for verification).
    pub fn get_final_evaluations(&self) -> Vec<FieldElement<F>> {
        self.factor_evals.iter().map(|e| e[0].clone()).collect()
    }

    /// Executes a round of the Sum-Check protocol and returns the round polynomial.
    ///
    /// The round polynomial g_j(X) is computed by:
    /// g_j(X) = Sum_{x_{j+1},...,x_n in {0,1}} Prod_i P_i(r_1,...,r_{j-1}, X, x_{j+1},...,x_n)
    ///
    /// After computing g_j, if a challenge r_j is provided, the working evaluations
    /// are updated in-place to prepare for the next round.
    fn round_impl(
        &mut self,
        r_prev: Option<&FieldElement<F>>,
    ) -> Result<Polynomial<FieldElement<F>>, ProverError> {
        if let Some(r) = r_prev {
            self.apply_challenge(r)?;
        }

        check_round_bounds(self.current_round, self.num_vars)?;

        let evaluations = compute_round_poly_product(&self.factor_evals);

        let num_eval_points = self.factor_evals.len() + 1;
        let eval_points: Vec<FieldElement<F>> = (0..num_eval_points)
            .map(|i| FieldElement::from(i as u64))
            .collect();

        let poly = Polynomial::interpolate(&eval_points, &evaluations)?;

        self.current_round += 1;

        Ok(poly)
    }

    /// Applies a challenge to update the working evaluations in-place.
    fn apply_challenge(&mut self, r: &FieldElement<F>) -> Result<(), ProverError> {
        if self.current_round == 0 {
            return Err(ProverError::InvalidState(
                "Cannot apply challenge before first round.".to_string(),
            ));
        }

        let current_size = 1 << (self.num_vars - self.current_round + 1);
        let half = current_size / 2;

        for factor_eval in &mut self.factor_evals {
            for k in 0..half {
                let p0 = &factor_eval[k];
                let p1 = &factor_eval[k + half];
                factor_eval[k] = p0 + r * &(p1 - p0);
            }
            factor_eval.truncate(half);
        }

        Ok(())
    }
}

impl<F: IsField> SumcheckProver<F> for OptimizedProver<F>
where
    F::BaseType: Send + Sync,
    FieldElement<F>: Clone + Mul<Output = FieldElement<F>>,
{
    fn num_vars(&self) -> usize {
        self.num_vars
    }

    fn num_factors(&self) -> usize {
        self.factor_evals.len()
    }

    fn compute_initial_sum(&self) -> FieldElement<F> {
        compute_initial_sum_product(&self.factor_evals)
    }

    fn round(
        &mut self,
        r_prev: Option<&FieldElement<F>>,
    ) -> Result<Polynomial<FieldElement<F>>, ProverError> {
        self.round_impl(r_prev)
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
    let num_factors = factors.len();
    let mut prover = OptimizedProver::new(factors)?;
    run_sumcheck_protocol(&mut prover, num_factors)
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
