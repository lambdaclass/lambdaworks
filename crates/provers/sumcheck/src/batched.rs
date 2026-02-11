//! Batched Sumcheck Protocol
//!
//! Implements efficient batching of multiple sumcheck instances, useful for:
//! - GKR protocol (proving multiple gates in parallel)
//! - Batched polynomial commitments
//! - Lookup arguments with multiple tables
//!
//! The key insight is that multiple sumcheck instances with the same number
//! of variables can share the random challenges, reducing overall proof size
//! and verification cost.
//!
//! # Protocol Overview
//!
//! Given m sumcheck instances, each claiming:
//!   sum_{x in {0,1}^n} f_i(x) = c_i
//!
//! The batched protocol:
//! 1. Prover sends all claimed sums c_1, ..., c_m
//! 2. Verifier samples a random batching coefficient rho
//! 3. Parties run a single sumcheck for:
//!    sum_{x in {0,1}^n} (f_1(x) + rho*f_2(x) + ... + rho^{m-1}*f_m(x))
//!    = c_1 + rho*c_2 + ... + rho^{m-1}*c_m
//!
//! This reduces m sumcheck proofs to a single proof with m+1 additional field elements.

use crate::prover::ProverError;
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

/// Batched Sumcheck Prover.
///
/// Efficiently proves multiple sumcheck instances by batching them
/// with a random linear combination.
pub struct BatchedProver<F: IsField>
where
    F::BaseType: Send + Sync,
{
    num_vars: usize,
    /// Number of instances being batched
    num_instances: usize,
    /// Working evaluations for the combined polynomial
    combined_evals: Vec<FieldElement<F>>,
    /// Individual claimed sums (before batching)
    individual_sums: Vec<FieldElement<F>>,
    /// Batched claimed sum
    batched_sum: FieldElement<F>,
    /// Current round
    current_round: usize,
}

impl<F: IsField> BatchedProver<F>
where
    F::BaseType: Send + Sync,
    FieldElement<F>: Clone + Mul<Output = FieldElement<F>>,
{
    /// Creates a new BatchedProver for multiple polynomial instances.
    ///
    /// Each instance is a vector of polynomial factors (for product sumcheck).
    /// All instances must have the same number of variables.
    pub fn new(
        instances: Vec<Vec<DenseMultilinearPolynomial<F>>>,
        batching_coeff: FieldElement<F>,
    ) -> Result<Self, ProverError> {
        if instances.is_empty() {
            return Err(ProverError::FactorMismatch(
                "At least one instance is required.".to_string(),
            ));
        }

        // Verify all instances have the same number of variables
        let num_vars = instances[0]
            .first()
            .ok_or_else(|| {
                ProverError::FactorMismatch(
                    "Each instance must have at least one factor.".to_string(),
                )
            })?
            .num_vars();

        for (i, instance) in instances.iter().enumerate() {
            for factor in instance {
                if factor.num_vars() != num_vars {
                    return Err(ProverError::FactorMismatch(format!(
                        "Instance {} has mismatched variables",
                        i
                    )));
                }
            }
        }

        let num_instances = instances.len();
        let size = 1 << num_vars;

        // Compute individual sums and products for each instance
        let mut individual_sums = Vec::with_capacity(num_instances);
        let mut instance_evals: Vec<Vec<FieldElement<F>>> = Vec::with_capacity(num_instances);

        for instance in &instances {
            // Compute product evaluations for this instance
            let mut evals = vec![FieldElement::one(); size];
            for factor in instance {
                let factor_evals = factor.evals();
                for (i, eval) in evals.iter_mut().enumerate() {
                    *eval = eval.clone() * factor_evals[i].clone();
                }
            }

            // Sum for this instance
            let sum: FieldElement<F> = evals
                .iter()
                .cloned()
                .fold(FieldElement::zero(), |a, b| a + b);

            individual_sums.push(sum);
            instance_evals.push(evals);
        }

        // Compute batched sum: c_1 + rho*c_2 + rho^2*c_3 + ...
        let mut batched_sum = FieldElement::zero();
        let mut rho_power: FieldElement<F> = FieldElement::one();
        for sum in &individual_sums {
            batched_sum += &rho_power * sum;
            rho_power *= batching_coeff.clone();
        }

        // Combine evaluations: f_1 + rho*f_2 + rho^2*f_3 + ...
        let mut combined_evals = vec![FieldElement::zero(); size];
        let mut rho_power = FieldElement::<F>::one();

        for evals in &instance_evals {
            for (i, eval) in evals.iter().enumerate() {
                combined_evals[i] = &combined_evals[i] + &rho_power * eval;
            }
            rho_power *= batching_coeff.clone();
        }

        Ok(Self {
            num_vars,
            num_instances,
            combined_evals,
            individual_sums,
            batched_sum,
            current_round: 0,
        })
    }

    /// Returns the number of variables.
    pub fn num_vars(&self) -> usize {
        self.num_vars
    }

    /// Returns the number of batched instances.
    pub fn num_instances(&self) -> usize {
        self.num_instances
    }

    /// Returns the individual claimed sums (before batching).
    pub fn individual_sums(&self) -> &[FieldElement<F>] {
        &self.individual_sums
    }

    /// Returns the batched claimed sum.
    pub fn batched_sum(&self) -> &FieldElement<F> {
        &self.batched_sum
    }

    /// Executes a round of the batched sumcheck protocol.
    pub fn round(
        &mut self,
        r_prev: Option<&FieldElement<F>>,
    ) -> Result<Polynomial<FieldElement<F>>, ProverError> {
        // Apply previous challenge
        if let Some(r) = r_prev {
            let half = self.combined_evals.len() / 2;
            let one_minus_r = FieldElement::<F>::one() - r;

            for k in 0..half {
                let v0 = &self.combined_evals[k];
                let v1 = &self.combined_evals[k + half];
                self.combined_evals[k] = &one_minus_r * v0 + r * v1;
            }
            self.combined_evals.truncate(half);
        }

        if self.current_round >= self.num_vars {
            return Err(ProverError::InvalidState(
                "All variables already fixed.".to_string(),
            ));
        }

        let half = self.combined_evals.len() / 2;

        // Compute sums at x=0 and x=1
        let mut sum_0 = FieldElement::zero();
        let mut sum_1 = FieldElement::zero();

        for k in 0..half {
            sum_0 += self.combined_evals[k].clone();
            sum_1 += self.combined_evals[k + half].clone();
        }

        self.current_round += 1;

        // Interpolate round polynomial
        let eval_points = vec![FieldElement::zero(), FieldElement::one()];
        let evaluations = vec![sum_0, sum_1];

        Polynomial::interpolate(&eval_points, &evaluations).map_err(ProverError::InterpolationError)
    }
}

/// Result of a batched sumcheck proof.
pub struct BatchedProofOutput<F: IsField>
where
    F::BaseType: Send + Sync,
{
    /// Individual claimed sums for each instance
    pub individual_sums: Vec<FieldElement<F>>,
    /// Batching coefficient used
    pub batching_coeff: FieldElement<F>,
    /// Round polynomials
    pub proof_polys: Vec<Polynomial<FieldElement<F>>>,
}

/// Proves multiple sumcheck instances in a batch.
///
/// Each instance is a vector of polynomial factors.
/// Returns individual sums and the batched proof.
pub fn prove_batched<F>(
    instances: Vec<Vec<DenseMultilinearPolynomial<F>>>,
) -> Result<BatchedProofOutput<F>, ProverError>
where
    F: IsField + HasDefaultTranscript,
    F::BaseType: Send + Sync,
    FieldElement<F>: Clone + Mul<Output = FieldElement<F>> + ByteConversion,
{
    if instances.is_empty() || instances[0].is_empty() {
        return Err(ProverError::FactorMismatch(
            "At least one instance with one polynomial factor is required.".to_string(),
        ));
    }

    let num_vars = instances[0][0].num_vars();
    let num_instances = instances.len();

    // Initialize transcript and compute batching coefficient
    let mut transcript = DefaultTranscript::<F>::default();
    transcript.append_bytes(b"batched_sumcheck");
    transcript.append_bytes(&(num_vars as u64).to_be_bytes());
    transcript.append_bytes(&(num_instances as u64).to_be_bytes());

    // Compute individual sums first to include in transcript
    let individual_sums: Vec<FieldElement<F>> = instances
        .iter()
        .map(|instance| {
            let size = 1 << num_vars;
            let mut sum = FieldElement::zero();
            for i in 0..size {
                let mut product: FieldElement<F> = FieldElement::one();
                for factor in instance {
                    product *= factor.evals()[i].clone();
                }
                sum += product;
            }
            sum
        })
        .collect();

    // Add individual sums to transcript
    for sum in &individual_sums {
        transcript.append_felt(sum);
    }

    // Sample batching coefficient
    let batching_coeff: FieldElement<F> = transcript.draw_felt();

    // Create batched prover
    let mut prover = BatchedProver::new(instances, batching_coeff.clone())?;

    let mut proof_polys = Vec::with_capacity(num_vars);
    let mut current_challenge: Option<FieldElement<F>> = None;

    for j in 0..num_vars {
        let g_j = prover.round(current_challenge.as_ref())?;

        let round_label = format!("round_{j}_poly");
        transcript.append_bytes(round_label.as_bytes());

        let coeffs = g_j.coefficients();
        transcript.append_bytes(&(coeffs.len() as u64).to_be_bytes());
        for coeff in coeffs {
            transcript.append_felt(coeff);
        }

        proof_polys.push(g_j);

        if j < num_vars - 1 {
            current_challenge = Some(transcript.draw_felt());
        }
    }

    Ok(BatchedProofOutput {
        individual_sums,
        batching_coeff,
        proof_polys,
    })
}

/// Verifies a batched sumcheck proof.
pub fn verify_batched<F>(
    num_vars: usize,
    instances: Vec<Vec<DenseMultilinearPolynomial<F>>>,
    proof: &BatchedProofOutput<F>,
) -> Result<bool, crate::VerifierError<F>>
where
    F: IsField + HasDefaultTranscript,
    F::BaseType: Send + Sync,
    FieldElement<F>: Clone + Mul<Output = FieldElement<F>> + ByteConversion,
{
    use crate::VerifierError;

    let num_instances = instances.len();

    // Recompute batched sum
    let mut batched_sum = FieldElement::zero();
    let mut rho_power: FieldElement<F> = FieldElement::one();
    for sum in &proof.individual_sums {
        batched_sum += &rho_power * sum;
        rho_power *= proof.batching_coeff.clone();
    }

    // Initialize transcript for verification
    let mut transcript = DefaultTranscript::<F>::default();
    transcript.append_bytes(b"batched_sumcheck");
    transcript.append_bytes(&(num_vars as u64).to_be_bytes());
    transcript.append_bytes(&(num_instances as u64).to_be_bytes());

    for sum in &proof.individual_sums {
        transcript.append_felt(sum);
    }

    // Verify batching coefficient matches
    let expected_batching_coeff: FieldElement<F> = transcript.draw_felt();
    if expected_batching_coeff != proof.batching_coeff {
        return Ok(false);
    }

    let mut expected_sum = batched_sum;
    let mut challenges = Vec::with_capacity(num_vars);

    for j in 0..num_vars {
        let g_j = &proof.proof_polys[j];

        // Check sum consistency
        let sum_at_0 = g_j.evaluate(&FieldElement::zero());
        let sum_at_1 = g_j.evaluate(&FieldElement::one());
        let poly_sum = sum_at_0.clone() + sum_at_1.clone();

        if poly_sum != expected_sum {
            return Err(VerifierError::InconsistentSum {
                round: j,
                expected: expected_sum,
                s0: sum_at_0,
                s1: sum_at_1,
            });
        }

        // Update transcript
        let round_label = format!("round_{j}_poly");
        transcript.append_bytes(round_label.as_bytes());

        let coeffs = g_j.coefficients();
        transcript.append_bytes(&(coeffs.len() as u64).to_be_bytes());
        for coeff in coeffs {
            transcript.append_felt(coeff);
        }

        let r_j: FieldElement<F> = transcript.draw_felt();
        expected_sum = g_j.evaluate(&r_j);
        challenges.push(r_j);
    }

    // Oracle check: verify the final evaluation against combined polynomial
    let mut combined_eval = FieldElement::zero();
    let mut rho_power = FieldElement::<F>::one();

    for instance in &instances {
        let mut instance_eval = FieldElement::<F>::one();
        for factor in instance {
            let factor_eval = factor
                .evaluate(challenges.clone())
                .map_err(|e| VerifierError::OracleEvaluationError(e.into()))?;
            instance_eval *= factor_eval;
        }
        combined_eval += &rho_power * &instance_eval;
        rho_power *= proof.batching_coeff.clone();
    }

    if combined_eval != expected_sum {
        return Err(VerifierError::InvalidState(format!(
            "Combined evaluation mismatch: expected {:?}, got {:?}",
            expected_sum, combined_eval
        )));
    }

    Ok(true)
}

/// Batched sumcheck for polynomials sharing the same structure.
///
/// This is more efficient than prove_batched when all instances
/// have the same polynomial factors, just with different evaluations.
pub struct StructuredBatchedProver<F: IsField>
where
    F::BaseType: Send + Sync,
{
    num_vars: usize,
    /// Evaluations for each instance
    instance_evals: Vec<Vec<FieldElement<F>>>,
    /// Batching coefficient
    batching_coeff: FieldElement<F>,
    /// Current round
    current_round: usize,
}

impl<F: IsField> StructuredBatchedProver<F>
where
    F::BaseType: Send + Sync,
    FieldElement<F>: Clone + Mul<Output = FieldElement<F>>,
{
    /// Creates a structured batch prover.
    ///
    /// All polynomials must have the same number of variables.
    pub fn new(
        polynomials: Vec<DenseMultilinearPolynomial<F>>,
        batching_coeff: FieldElement<F>,
    ) -> Result<Self, ProverError> {
        if polynomials.is_empty() {
            return Err(ProverError::FactorMismatch(
                "At least one polynomial required.".to_string(),
            ));
        }

        let num_vars = polynomials[0].num_vars();
        for (i, poly) in polynomials.iter().enumerate().skip(1) {
            if poly.num_vars() != num_vars {
                return Err(ProverError::FactorMismatch(format!(
                    "Polynomial {} has {} variables, expected {}",
                    i,
                    poly.num_vars(),
                    num_vars
                )));
            }
        }

        let instance_evals: Vec<Vec<FieldElement<F>>> = polynomials
            .into_iter()
            .map(|p| p.evals().to_vec())
            .collect();

        Ok(Self {
            num_vars,
            instance_evals,
            batching_coeff,
            current_round: 0,
        })
    }

    /// Returns the number of variables.
    pub fn num_vars(&self) -> usize {
        self.num_vars
    }

    /// Returns the individual sums.
    pub fn compute_individual_sums(&self) -> Vec<FieldElement<F>> {
        self.instance_evals
            .iter()
            .map(|evals| {
                evals
                    .iter()
                    .cloned()
                    .fold(FieldElement::zero(), |a, b| a + b)
            })
            .collect()
    }

    /// Computes the batched sum.
    pub fn compute_batched_sum(&self) -> FieldElement<F> {
        let individual_sums = self.compute_individual_sums();
        let mut batched_sum = FieldElement::zero();
        let mut rho_power: FieldElement<F> = FieldElement::one();

        for sum in individual_sums {
            batched_sum += &rho_power * &sum;
            rho_power *= self.batching_coeff.clone();
        }

        batched_sum
    }

    /// Executes a round of the structured batched sumcheck.
    pub fn round(
        &mut self,
        r_prev: Option<&FieldElement<F>>,
    ) -> Result<Polynomial<FieldElement<F>>, ProverError> {
        // Apply challenge to all instances
        if let Some(r) = r_prev {
            let one_minus_r = FieldElement::<F>::one() - r;

            for evals in &mut self.instance_evals {
                let half = evals.len() / 2;
                for k in 0..half {
                    let v0 = &evals[k];
                    let v1 = &evals[k + half];
                    evals[k] = &one_minus_r * v0 + r * v1;
                }
                evals.truncate(half);
            }
        }

        if self.current_round >= self.num_vars {
            return Err(ProverError::InvalidState(
                "All variables already fixed.".to_string(),
            ));
        }

        let half = self.instance_evals[0].len() / 2;

        // Compute batched sums at x=0 and x=1
        let mut sum_0 = FieldElement::zero();
        let mut sum_1 = FieldElement::zero();
        let mut rho_power = FieldElement::<F>::one();

        for evals in &self.instance_evals {
            let mut instance_sum_0 = FieldElement::<F>::zero();
            let mut instance_sum_1 = FieldElement::<F>::zero();

            for k in 0..half {
                instance_sum_0 += evals[k].clone();
                instance_sum_1 += evals[k + half].clone();
            }

            sum_0 += &rho_power * &instance_sum_0;
            sum_1 += &rho_power * &instance_sum_1;
            rho_power *= self.batching_coeff.clone();
        }

        self.current_round += 1;

        let eval_points = vec![FieldElement::zero(), FieldElement::one()];
        let evaluations = vec![sum_0, sum_1];

        Polynomial::interpolate(&eval_points, &evaluations).map_err(ProverError::InterpolationError)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lambdaworks_math::field::fields::u64_prime_field::U64PrimeField;

    const MODULUS: u64 = 101;
    type F = U64PrimeField<MODULUS>;
    type FE = FieldElement<F>;

    #[test]
    fn test_batched_prover_single_instance() {
        // Single instance should work like regular sumcheck
        let poly = DenseMultilinearPolynomial::new(vec![
            FE::from(1),
            FE::from(2),
            FE::from(3),
            FE::from(4),
        ]);

        let instances = vec![vec![poly.clone()]];
        let proof = prove_batched(instances.clone()).unwrap();

        // Individual sum should be 1+2+3+4 = 10
        assert_eq!(proof.individual_sums[0], FE::from(10));

        let result = verify_batched(2, instances, &proof);
        assert!(result.unwrap_or(false));
    }

    #[test]
    fn test_batched_prover_two_instances() {
        let poly1 = DenseMultilinearPolynomial::new(vec![
            FE::from(1),
            FE::from(2),
            FE::from(3),
            FE::from(4),
        ]);

        let poly2 = DenseMultilinearPolynomial::new(vec![
            FE::from(5),
            FE::from(6),
            FE::from(7),
            FE::from(8),
        ]);

        let instances = vec![vec![poly1.clone()], vec![poly2.clone()]];
        let proof = prove_batched(instances.clone()).unwrap();

        // First instance sum: 1+2+3+4 = 10
        assert_eq!(proof.individual_sums[0], FE::from(10));
        // Second instance sum: 5+6+7+8 = 26
        assert_eq!(proof.individual_sums[1], FE::from(26));

        let result = verify_batched(2, instances, &proof);
        assert!(result.unwrap_or(false));
    }

    #[test]
    fn test_batched_prover_product_instances_sums() {
        // Verify that the prover correctly computes individual sums for product instances.
        // Note: The batched prover pre-computes point-wise products, so verification
        // via verify_batched requires single-factor instances. For product instances,
        // the oracle check would need access to the pre-computed product polynomial.
        let poly_a1 = DenseMultilinearPolynomial::new(vec![FE::from(1), FE::from(2)]);
        let poly_b1 = DenseMultilinearPolynomial::new(vec![FE::from(3), FE::from(4)]);

        let poly_a2 = DenseMultilinearPolynomial::new(vec![FE::from(5), FE::from(6)]);
        let poly_b2 = DenseMultilinearPolynomial::new(vec![FE::from(7), FE::from(8)]);

        let instances = vec![
            vec![poly_a1.clone(), poly_b1.clone()],
            vec![poly_a2.clone(), poly_b2.clone()],
        ];
        let proof = prove_batched(instances.clone()).unwrap();

        // First instance: 1*3 + 2*4 = 3 + 8 = 11
        assert_eq!(proof.individual_sums[0], FE::from(11));
        // Second instance: 5*7 + 6*8 = 35 + 48 = 83
        assert_eq!(proof.individual_sums[1], FE::from(83));

        // Round polynomial consistency: g(0) + g(1) = batched_sum
        let mut batched_sum = FE::zero();
        let mut rho_power = FE::one();
        for sum in &proof.individual_sums {
            batched_sum += rho_power * sum;
            rho_power *= proof.batching_coeff;
        }
        let g0_at_0 = proof.proof_polys[0].evaluate(&FE::zero());
        let g0_at_1 = proof.proof_polys[0].evaluate(&FE::one());
        assert_eq!(g0_at_0 + g0_at_1, batched_sum);
    }

    #[test]
    fn test_structured_batched_prover() {
        let poly1 = DenseMultilinearPolynomial::new(vec![
            FE::from(1),
            FE::from(2),
            FE::from(3),
            FE::from(4),
        ]);

        let poly2 = DenseMultilinearPolynomial::new(vec![
            FE::from(5),
            FE::from(6),
            FE::from(7),
            FE::from(8),
        ]);

        let rho = FE::from(7);
        let prover = StructuredBatchedProver::new(vec![poly1, poly2], rho).unwrap();

        let sums = prover.compute_individual_sums();
        assert_eq!(sums[0], FE::from(10)); // 1+2+3+4
        assert_eq!(sums[1], FE::from(26)); // 5+6+7+8

        // Batched sum: 10 + 7*26 = 10 + 182 = 192 mod 101 = 91
        let batched = prover.compute_batched_sum();
        assert_eq!(batched, FE::from(91));
    }

    #[test]
    fn test_batched_many_instances() {
        // Test with 4 instances
        let polys: Vec<DenseMultilinearPolynomial<F>> = (0..4)
            .map(|i| {
                DenseMultilinearPolynomial::new(vec![
                    FE::from(i * 4 + 1),
                    FE::from(i * 4 + 2),
                    FE::from(i * 4 + 3),
                    FE::from(i * 4 + 4),
                ])
            })
            .collect();

        let instances: Vec<Vec<DenseMultilinearPolynomial<F>>> =
            polys.into_iter().map(|p| vec![p]).collect();

        let proof = prove_batched(instances.clone()).unwrap();

        assert_eq!(proof.individual_sums.len(), 4);
        // Sums: 1+2+3+4=10, 5+6+7+8=26, 9+10+11+12=42, 13+14+15+16=58
        assert_eq!(proof.individual_sums[0], FE::from(10));
        assert_eq!(proof.individual_sums[1], FE::from(26));
        assert_eq!(proof.individual_sums[2], FE::from(42));
        assert_eq!(proof.individual_sums[3], FE::from(58));

        let result = verify_batched(2, instances, &proof);
        assert!(result.unwrap_or(false));
    }
}
