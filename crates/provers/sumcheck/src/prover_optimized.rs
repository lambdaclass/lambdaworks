use crate::Channel;
use lambdaworks_crypto::fiat_shamir::default_transcript::DefaultTranscript;
use lambdaworks_crypto::fiat_shamir::is_transcript::IsTranscript;
use lambdaworks_math::{
    field::{
        element::FieldElement,
        traits::{HasDefaultTranscript, IsField},
    },
    polynomial::{
        dense_multilinear_poly::DenseMultilinearPolynomial, InterpolateError, Polynomial,
    },
    traits::ByteConversion,
};
use std::ops::Mul;
use thiserror::Error;

use crate::prover::ProverOutput;

#[derive(Debug, Error)]
pub enum OptimizedProverError {
    #[error("Factor mismatch: {0}")]
    FactorMismatch(String),
    #[error("Polynomial interpolation failed: {0}")]
    InterpolationError(#[from] InterpolateError),
    #[error("Invalid prover state: {0}")]
    InvalidState(String),
}

impl From<OptimizedProverError> for crate::prover::ProverError {
    fn from(e: OptimizedProverError) -> Self {
        match e {
            OptimizedProverError::FactorMismatch(msg) => {
                crate::prover::ProverError::FactorMismatch(msg)
            }
            OptimizedProverError::InterpolationError(e) => {
                crate::prover::ProverError::InterpolationError(e)
            }
            OptimizedProverError::InvalidState(msg) => {
                crate::prover::ProverError::InvalidState(msg)
            }
        }
    }
}

/// Optimized Sumcheck prover using table folding.
///
/// Instead of re-evaluating the full polynomial product from scratch each round
/// (O(d^2 * n * 2^n) total), this prover maintains mutable evaluation tables and
/// folds them in half after each challenge, achieving O(d^2 * 2^n) total work.
pub struct OptimizedProver<F: IsField>
where
    F::BaseType: Send + Sync,
{
    num_vars: usize,
    num_factors: usize,
    /// Mutable evaluation tables, one per factor polynomial.
    /// After round j, each table has 2^(n-j) entries.
    tables: Vec<Vec<FieldElement<F>>>,
    current_round: usize,
}

impl<F: IsField> OptimizedProver<F>
where
    F::BaseType: Send + Sync,
    FieldElement<F>: Clone + Mul<Output = FieldElement<F>>,
{
    /// Creates a new OptimizedProver from factor polynomials.
    pub fn new(
        factors: Vec<DenseMultilinearPolynomial<F>>,
    ) -> Result<Self, OptimizedProverError> {
        if factors.is_empty() {
            return Err(OptimizedProverError::FactorMismatch(
                "At least one polynomial factor is required.".to_string(),
            ));
        }
        let num_vars = factors[0].num_vars();
        if factors.iter().any(|p| p.num_vars() != num_vars) {
            return Err(OptimizedProverError::FactorMismatch(
                "All factors must have the same number of variables.".to_string(),
            ));
        }

        let tables: Vec<Vec<FieldElement<F>>> =
            factors.iter().map(|f| f.evals().clone()).collect();

        Ok(Self {
            num_vars,
            num_factors: factors.len(),
            tables,
            current_round: 0,
        })
    }

    pub fn num_vars(&self) -> usize {
        self.num_vars
    }

    /// Computes the initial claimed sum: C = sum_{x in {0,1}^n} prod_i P_i(x)
    pub fn compute_initial_sum(&self) -> FieldElement<F> {
        let table_len = self.tables[0].len(); // 2^n
        let mut sum = FieldElement::zero();
        for j in 0..table_len {
            let mut prod = self.tables[0][j].clone();
            for k in 1..self.num_factors {
                prod = prod * self.tables[k][j].clone();
            }
            sum = sum + prod;
        }
        sum
    }

    /// Computes the round polynomial g_j(X_j) using table folding.
    ///
    /// For each evaluation point t in {0, 1, ..., d}, computes:
    ///   g_j(t) = sum_{j=0}^{half-1} prod_k interpolated_factor_k(t)
    /// where interpolated_factor_k(t) = table_k[j] + t * (table_k[j+half] - table_k[j])
    ///
    /// Special cases: t=0 uses table[j] directly, t=1 uses table[j+half] directly.
    pub fn compute_round_polynomial(
        &self,
    ) -> Result<Polynomial<FieldElement<F>>, OptimizedProverError> {
        if self.current_round >= self.num_vars {
            return Err(OptimizedProverError::InvalidState(
                "All rounds completed.".to_string(),
            ));
        }

        let current_table_len = self.tables[0].len();
        let half = current_table_len / 2;
        let num_eval_points = self.num_factors + 1; // degree d = num_factors, need d+1 points

        let mut eval_points_x = Vec::with_capacity(num_eval_points);
        let mut eval_values_y = Vec::with_capacity(num_eval_points);

        for i in 0..num_eval_points {
            let t = FieldElement::from(i as u64);
            eval_points_x.push(t.clone());

            let mut sum = FieldElement::zero();
            for j in 0..half {
                let mut prod = FieldElement::one();
                for k in 0..self.num_factors {
                    let val = if i == 0 {
                        // t=0: interpolated value is table[j]
                        self.tables[k][j].clone()
                    } else if i == 1 {
                        // t=1: interpolated value is table[j+half]
                        self.tables[k][j + half].clone()
                    } else {
                        // General: table[j] + t * (table[j+half] - table[j])
                        let lo = &self.tables[k][j];
                        let hi = &self.tables[k][j + half];
                        let diff = hi.clone() - lo.clone();
                        lo.clone() + t.clone() * diff
                    };
                    prod = prod * val;
                }
                sum = sum + prod;
            }
            eval_values_y.push(sum);
        }

        let poly = Polynomial::interpolate(&eval_points_x, &eval_values_y)?;
        Ok(poly)
    }

    /// Folds all evaluation tables using the challenge r.
    /// table[j] = table[j] + r * (table[j+half] - table[j])
    /// Then truncates each table to half its size.
    pub fn receive_challenge(&mut self, r: &FieldElement<F>) -> Result<(), OptimizedProverError> {
        if self.current_round >= self.num_vars {
            return Err(OptimizedProverError::InvalidState(
                "All rounds completed, cannot receive more challenges.".to_string(),
            ));
        }

        let current_table_len = self.tables[0].len();
        let half = current_table_len / 2;

        for table in &mut self.tables {
            for j in 0..half {
                let lo = table[j].clone();
                let hi = table[j + half].clone();
                table[j] = lo.clone() + r.clone() * (hi - lo);
            }
            table.truncate(half);
        }

        self.current_round += 1;
        Ok(())
    }
}

/// Non-interactive optimized prover using Fiat-Shamir transform.
///
/// Uses identical transcript protocol as `prove()`, so output is verifiable
/// by the existing `verify()` function.
pub fn prove_optimized<F>(factors: Vec<DenseMultilinearPolynomial<F>>) -> ProverOutput<F>
where
    F: IsField + HasDefaultTranscript,
    F::BaseType: Send + Sync,
    FieldElement<F>: Clone + Mul<Output = FieldElement<F>> + ByteConversion,
{
    let num_factors = factors.len();
    let mut prover = OptimizedProver::new(factors)?;
    let num_vars = prover.num_vars();

    let claimed_sum = prover.compute_initial_sum();

    // Initialize Fiat-Shamir transcript (identical to prove())
    let mut transcript = DefaultTranscript::<F>::default();
    transcript.append_bytes(b"initial_sum");
    transcript.append_felt(&FieldElement::from(num_vars as u64));
    transcript.append_felt(&FieldElement::from(num_factors as u64));
    transcript.append_felt(&claimed_sum);

    let mut proof_polys = Vec::with_capacity(num_vars);

    for j in 0..num_vars {
        let g_j = prover.compute_round_polynomial()?;

        // Append g_j to transcript (identical to prove())
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

        // Derive challenge and fold tables (except after last round)
        if j < num_vars - 1 {
            let challenge: FieldElement<F> = transcript.draw_felt();
            prover.receive_challenge(&challenge)?;
        }
    }

    Ok((claimed_sum, proof_polys))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{verify_linear, verify_quadratic, verify_cubic};
    use lambdaworks_math::field::fields::u64_prime_field::U64PrimeField;
    use lambdaworks_math::polynomial::dense_multilinear_poly::DenseMultilinearPolynomial;

    const MODULUS: u64 = 101;
    type F = U64PrimeField<MODULUS>;
    type FE = FieldElement<F>;

    #[test]
    fn test_table_folding_matches_fix_first_variable() {
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
        let r = FE::from(42);

        // Reference: fix_first_variable
        let fixed = poly.fix_first_variable(&r);
        let expected_evals = fixed.evals().clone();

        // Table folding
        let mut prover = OptimizedProver::new(vec![poly]).unwrap();
        prover.compute_round_polynomial().unwrap();
        prover.receive_challenge(&r).unwrap();
        let folded_evals = prover.tables[0].clone();

        assert_eq!(folded_evals, expected_evals);
    }

    #[test]
    fn test_optimized_initial_sum_matches_naive() {
        let poly = DenseMultilinearPolynomial::new(vec![
            FE::from(1),
            FE::from(2),
            FE::from(3),
            FE::from(4),
        ]);
        let factors = vec![poly.clone()];

        let naive_prover = crate::prover::Prover::new(factors.clone()).unwrap();
        let naive_sum = naive_prover.compute_initial_sum().unwrap();

        let opt_prover = OptimizedProver::new(factors).unwrap();
        let opt_sum = opt_prover.compute_initial_sum();

        assert_eq!(naive_sum, opt_sum);
    }

    #[test]
    fn test_optimized_linear_verified() {
        let poly = DenseMultilinearPolynomial::new(vec![
            FE::from(3),
            FE::from(5),
            FE::from(7),
            FE::from(11),
        ]);
        let num_vars = poly.num_vars();

        let (claimed_sum, proof_polys) = prove_optimized(vec![poly.clone()]).unwrap();

        let result = verify_linear(num_vars, claimed_sum, proof_polys, poly);
        assert!(result.unwrap_or(false), "Optimized linear proof should verify");
    }

    #[test]
    fn test_optimized_quadratic_verified() {
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

        let result =
            verify_quadratic(num_vars, claimed_sum, proof_polys, poly_a, poly_b);
        assert!(
            result.unwrap_or(false),
            "Optimized quadratic proof should verify"
        );
    }

    #[test]
    fn test_optimized_cubic_verified() {
        let poly_a = DenseMultilinearPolynomial::new(vec![FE::from(1), FE::from(2)]);
        let poly_b = DenseMultilinearPolynomial::new(vec![FE::from(3), FE::from(4)]);
        let poly_c = DenseMultilinearPolynomial::new(vec![FE::from(5), FE::from(1)]);
        let num_vars = poly_a.num_vars();

        let (claimed_sum, proof_polys) =
            prove_optimized(vec![poly_a.clone(), poly_b.clone(), poly_c.clone()]).unwrap();

        let result = verify_cubic(
            num_vars, claimed_sum, proof_polys, poly_a, poly_b, poly_c,
        );
        assert!(
            result.unwrap_or(false),
            "Optimized cubic proof should verify"
        );
    }

    #[test]
    fn test_optimized_matches_naive_output() {
        let poly = DenseMultilinearPolynomial::new(vec![
            FE::from(1),
            FE::from(2),
            FE::from(1),
            FE::from(4),
        ]);
        let factors = vec![poly];

        let (naive_sum, naive_polys) = crate::prover::prove(factors.clone()).unwrap();
        let (opt_sum, opt_polys) = prove_optimized(factors).unwrap();

        assert_eq!(naive_sum, opt_sum, "Claimed sums should match");
        assert_eq!(
            naive_polys.len(),
            opt_polys.len(),
            "Number of round polys should match"
        );
        for (j, (np, op)) in naive_polys.iter().zip(opt_polys.iter()).enumerate() {
            assert_eq!(
                np.coefficients(),
                op.coefficients(),
                "Round {j} polynomial coefficients should match"
            );
        }
    }

    #[test]
    fn test_optimized_3var_verified() {
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

        let (claimed_sum, proof_polys) = prove_optimized(vec![poly.clone()]).unwrap();

        let result = verify_linear(num_vars, claimed_sum, proof_polys, poly);
        assert!(result.unwrap_or(false), "Optimized 3-var proof should verify");
    }

    #[test]
    fn test_optimized_quadratic_3var_matches_naive() {
        let poly_a = DenseMultilinearPolynomial::new(vec![
            FE::from(1),
            FE::from(2),
            FE::from(3),
            FE::from(4),
            FE::from(5),
            FE::from(6),
            FE::from(7),
            FE::from(8),
        ]);
        let poly_b = DenseMultilinearPolynomial::new(vec![
            FE::from(10),
            FE::from(20),
            FE::from(30),
            FE::from(40),
            FE::from(50),
            FE::from(60),
            FE::from(70),
            FE::from(80),
        ]);
        let factors = vec![poly_a, poly_b];

        let (naive_sum, naive_polys) = crate::prover::prove(factors.clone()).unwrap();
        let (opt_sum, opt_polys) = prove_optimized(factors).unwrap();

        assert_eq!(naive_sum, opt_sum);
        for (j, (np, op)) in naive_polys.iter().zip(opt_polys.iter()).enumerate() {
            assert_eq!(
                np.coefficients(),
                op.coefficients(),
                "Round {j} polynomial coefficients should match"
            );
        }
    }
}
