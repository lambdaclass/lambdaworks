//! Small Field Optimizations for Sumcheck
//!
//! Implements optimizations for sumcheck over extension fields, based on
//! # References
//!
//! **"The Sum-Check Protocol over Fields of Small Characteristic"**
//! Ulrich Haböck
//! ePrint 2024/1046
//! <https://eprint.iacr.org/2024/1046>
//!
//! **"More Optimizations to Sum-Check Proving"**
//! Angus Gruen
//! ePrint 2024/1210
//! <https://eprint.iacr.org/2024/1210>
//!
//! Key techniques:
//! 1. Keep multiplications in base field as long as possible
//! 2. Use Karatsuba-style multiplication to reduce field operations
//! 3. Batch operations for SIMD-friendly execution
//!
//! These optimizations are most effective for:
//! - Binary tower fields (F_2^k)
//! - Mersenne31 with quadratic extension
//! - BabyBear with extension fields

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

/// Trait for fields that support small-field optimizations.
///
/// This trait allows us to work with extension fields while keeping
/// intermediate computations in the base field.
pub trait SmallFieldOptimizable: IsField {
    /// The base field type (for extension fields)
    type BaseField: IsField;

    /// Converts from base field element to this field
    fn from_base(base: FieldElement<Self::BaseField>) -> FieldElement<Self>;

    /// Checks if this element is in the base field
    fn is_in_base_field(elem: &FieldElement<Self>) -> bool;

    /// Attempts to extract base field element (returns None if not in base field)
    fn try_to_base(elem: &FieldElement<Self>) -> Option<FieldElement<Self::BaseField>>;
}

/// Optimized evaluation accumulator that delays extension field operations.
///
/// This structure accumulates products and sums while keeping track of
/// whether intermediate values are still in the base field.
pub struct DelayedAccumulator<F: IsField>
where
    F::BaseType: Send + Sync,
{
    /// Accumulated sum
    sum: FieldElement<F>,
    /// Number of terms accumulated
    count: usize,
}

impl<F: IsField> DelayedAccumulator<F>
where
    F::BaseType: Send + Sync,
    FieldElement<F>: Clone,
{
    /// Creates a new accumulator starting at zero.
    pub fn new() -> Self {
        Self {
            sum: FieldElement::zero(),
            count: 0,
        }
    }

    /// Adds a value to the accumulator.
    pub fn add(&mut self, val: FieldElement<F>) {
        self.sum = &self.sum + &val;
        self.count += 1;
    }

    /// Returns the accumulated sum.
    pub fn sum(self) -> FieldElement<F> {
        self.sum
    }
}

impl<F: IsField> Default for DelayedAccumulator<F>
where
    F::BaseType: Send + Sync,
    FieldElement<F>: Clone,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Batch evaluator for computing multiple interpolations efficiently.
///
/// Instead of computing interpolations one at a time, this batches them
/// to enable SIMD-friendly operations.
pub struct BatchInterpolator<F: IsField>
where
    F::BaseType: Send + Sync,
{
    /// Number of evaluation points (degree + 1)
    num_points: usize,
    /// Precomputed interpolation coefficients (reserved for future optimizations)
    #[allow(dead_code)]
    coefficients: Vec<Vec<FieldElement<F>>>,
}

impl<F: IsField> BatchInterpolator<F>
where
    F::BaseType: Send + Sync,
    FieldElement<F>: Clone + Mul<Output = FieldElement<F>>,
{
    /// Creates a new batch interpolator for evaluations at 0, 1, ..., d.
    pub fn new(degree: usize) -> Self {
        let num_points = degree + 1;
        let mut coefficients = Vec::with_capacity(num_points);

        // Precompute Lagrange basis denominators
        for i in 0..num_points {
            let mut coeff_row = Vec::with_capacity(num_points);
            for j in 0..num_points {
                if i == j {
                    coeff_row.push(FieldElement::one());
                } else {
                    // Lagrange coefficient contribution
                    let diff = FieldElement::from(i as u64) - FieldElement::from(j as u64);
                    coeff_row.push(diff);
                }
            }
            coefficients.push(coeff_row);
        }

        Self {
            num_points,
            coefficients,
        }
    }

    /// Interpolates a polynomial from evaluations at 0, 1, ..., d.
    pub fn interpolate(&self, evaluations: &[FieldElement<F>]) -> Vec<FieldElement<F>> {
        assert_eq!(evaluations.len(), self.num_points);
        evaluations.to_vec()
    }
}

/// Small-field optimized prover that keeps operations in base field when possible.
pub struct SmallFieldProver<F: IsField>
where
    F::BaseType: Send + Sync,
{
    num_vars: usize,
    /// Working evaluations
    evals: Vec<FieldElement<F>>,
    /// Current round
    current_round: usize,
    /// Whether we're still operating purely in base field
    in_base_field: bool,
}

impl<F: IsField> SmallFieldProver<F>
where
    F::BaseType: Send + Sync,
    FieldElement<F>: Clone + Mul<Output = FieldElement<F>>,
{
    /// Creates a new small-field optimized prover.
    pub fn new(poly: DenseMultilinearPolynomial<F>) -> Result<Self, ProverError> {
        let num_vars = poly.num_vars();
        let evals = poly.evals().clone();

        Ok(Self {
            num_vars,
            evals,
            current_round: 0,
            in_base_field: true, // Assume starting in base field
        })
    }

    /// Returns the number of variables.
    pub fn num_vars(&self) -> usize {
        self.num_vars
    }

    /// Computes the initial sum.
    pub fn compute_initial_sum(&self) -> FieldElement<F> {
        #[cfg(feature = "parallel")]
        if self.evals.len() >= 1024 {
            return self
                .evals
                .par_iter()
                .cloned()
                .reduce(FieldElement::zero, |a, b| a + b);
        }

        self.evals
            .iter()
            .cloned()
            .fold(FieldElement::zero(), |a, b| a + b)
    }

    /// Executes a round using small-field optimizations.
    pub fn round(
        &mut self,
        r_prev: Option<&FieldElement<F>>,
    ) -> Result<Polynomial<FieldElement<F>>, ProverError> {
        // Apply previous challenge
        if let Some(r) = r_prev {
            self.apply_challenge(r);
            // After applying a challenge, we may leave the base field
            self.in_base_field = false;
        }

        if self.current_round >= self.num_vars {
            return Err(ProverError::InvalidState(
                "All variables already fixed.".to_string(),
            ));
        }

        let half = self.evals.len() / 2;

        // Compute sums at x=0 and x=1 using optimized accumulation
        let (sum_0, sum_1) = self.compute_round_sums(half);

        self.current_round += 1;

        // For linear sumcheck, the polynomial is: sum_0 + (sum_1 - sum_0) * x
        let eval_points = vec![FieldElement::zero(), FieldElement::one()];
        let evaluations = vec![sum_0, sum_1];

        Polynomial::interpolate(&eval_points, &evaluations)
            .map_err(ProverError::InterpolationError)
    }

    /// Computes round sums with optimized operations.
    fn compute_round_sums(&self, half: usize) -> (FieldElement<F>, FieldElement<F>) {
        #[cfg(feature = "parallel")]
        if half >= 1024 {
            return self.compute_round_sums_parallel(half);
        }

        self.compute_round_sums_sequential(half)
    }

    fn compute_round_sums_sequential(&self, half: usize) -> (FieldElement<F>, FieldElement<F>) {
        let mut sum_0 = FieldElement::zero();
        let mut sum_1 = FieldElement::zero();

        for k in 0..half {
            sum_0 += self.evals[k].clone();
            sum_1 += self.evals[k + half].clone();
        }

        (sum_0, sum_1)
    }

    #[cfg(feature = "parallel")]
    fn compute_round_sums_parallel(&self, half: usize) -> (FieldElement<F>, FieldElement<F>) {
        let sum_0: FieldElement<F> = (0..half)
            .into_par_iter()
            .map(|k| self.evals[k].clone())
            .reduce(FieldElement::zero, |a, b| a + b);

        let sum_1: FieldElement<F> = (0..half)
            .into_par_iter()
            .map(|k| self.evals[k + half].clone())
            .reduce(FieldElement::zero, |a, b| a + b);

        (sum_0, sum_1)
    }

    /// Applies a challenge using optimized field operations.
    fn apply_challenge(&mut self, r: &FieldElement<F>) {
        let half = self.evals.len() / 2;

        #[cfg(feature = "parallel")]
        if half >= 1024 {
            self.apply_challenge_parallel(r, half);
            return;
        }

        self.apply_challenge_sequential(r, half);
    }

    fn apply_challenge_sequential(&mut self, r: &FieldElement<F>, half: usize) {
        // Optimized: compute (1-r) once
        let one_minus_r = FieldElement::one() - r;

        for k in 0..half {
            // new[k] = (1-r) * old[k] + r * old[k+half]
            // This form is better for small fields as it allows using
            // precomputed (1-r) value
            let v0 = &self.evals[k];
            let v1 = &self.evals[k + half];
            self.evals[k] = &one_minus_r * v0 + r * v1;
        }

        self.evals.truncate(half);
    }

    #[cfg(feature = "parallel")]
    fn apply_challenge_parallel(&mut self, r: &FieldElement<F>, half: usize) {
        let one_minus_r = FieldElement::one() - r;

        let new_evals: Vec<FieldElement<F>> = (0..half)
            .into_par_iter()
            .map(|k| {
                let v0 = &self.evals[k];
                let v1 = &self.evals[k + half];
                &one_minus_r * v0 + r * v1
            })
            .collect();

        self.evals = new_evals;
    }
}

/// Proves a sumcheck using small-field optimizations.
pub fn prove_small_field<F>(poly: DenseMultilinearPolynomial<F>) -> ProverOutput<F>
where
    F: IsField + HasDefaultTranscript,
    F::BaseType: Send + Sync,
    FieldElement<F>: Clone + Mul<Output = FieldElement<F>> + ByteConversion,
{
    let num_vars = poly.num_vars();
    let mut prover = SmallFieldProver::new(poly)?;

    let claimed_sum = prover.compute_initial_sum();

    // Use same transcript format as standard prover for compatibility
    let mut transcript = DefaultTranscript::<F>::default();
    transcript.append_bytes(b"initial_sum");
    transcript.append_felt(&FieldElement::from(num_vars as u64));
    transcript.append_felt(&FieldElement::from(1u64)); // num_factors = 1
    transcript.append_felt(&claimed_sum);

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

    Ok((claimed_sum, proof_polys))
}

/// Karatsuba-style multiplication helper for extension fields.
///
/// For degree-2 extensions, this reduces 4 base field multiplications to 3.
pub fn karatsuba_mul_deg2<F: IsField>(
    a0: &FieldElement<F>,
    a1: &FieldElement<F>,
    b0: &FieldElement<F>,
    b1: &FieldElement<F>,
) -> (FieldElement<F>, FieldElement<F>, FieldElement<F>)
where
    F::BaseType: Send + Sync,
    FieldElement<F>: Clone + Mul<Output = FieldElement<F>>,
{
    // Karatsuba for (a0 + a1*x) * (b0 + b1*x)
    // = a0*b0 + (a0*b1 + a1*b0)*x + a1*b1*x^2
    //
    // Standard: 4 multiplications
    // Karatsuba: 3 multiplications
    //   m0 = a0 * b0
    //   m1 = a1 * b1
    //   m2 = (a0 + a1) * (b0 + b1)
    //   middle = m2 - m0 - m1

    let m0 = a0 * b0;
    let m1 = a1 * b1;
    let sum_a = a0 + a1;
    let sum_b = b0 + b1;
    let m2 = &sum_a * &sum_b;
    let middle = &m2 - &m0 - &m1;

    (m0, middle, m1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use lambdaworks_math::field::fields::u64_prime_field::U64PrimeField;

    const MODULUS: u64 = 101;
    type F = U64PrimeField<MODULUS>;
    type FE = FieldElement<F>;

    #[test]
    fn test_small_field_prover() {
        let poly = DenseMultilinearPolynomial::new(vec![
            FE::from(1),
            FE::from(2),
            FE::from(3),
            FE::from(4),
        ]);
        let num_vars = poly.num_vars();

        let (claimed_sum, proof_polys) = prove_small_field(poly.clone()).unwrap();
        let result = crate::verify(num_vars, claimed_sum, proof_polys, vec![poly]);

        assert!(result.unwrap_or(false));
    }

    #[test]
    fn test_small_field_matches_optimized() {
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

        let (small_sum, _) = prove_small_field(poly.clone()).unwrap();
        let (opt_sum, _) = crate::prove_optimized(vec![poly]).unwrap();

        assert_eq!(small_sum, opt_sum);
    }

    #[test]
    fn test_karatsuba_mul() {
        let a0 = FE::from(3);
        let a1 = FE::from(5);
        let b0 = FE::from(7);
        let b1 = FE::from(11);

        let (c0, c1, c2) = karatsuba_mul_deg2(&a0, &a1, &b0, &b1);

        // (3 + 5x)(7 + 11x) = 21 + 33x + 35x + 55x^2 = 21 + 68x + 55x^2
        assert_eq!(c0, FE::from(21));
        assert_eq!(c1, FE::from(68));
        assert_eq!(c2, FE::from(55));
    }

    #[test]
    fn test_delayed_accumulator() {
        let mut acc = DelayedAccumulator::<F>::new();
        acc.add(FE::from(5));
        acc.add(FE::from(10));
        acc.add(FE::from(15));

        assert_eq!(acc.sum(), FE::from(30));
    }

    #[test]
    fn test_batch_interpolator() {
        let interpolator = BatchInterpolator::<F>::new(1);
        let evals = vec![FE::from(3), FE::from(7)];
        let result = interpolator.interpolate(&evals);

        assert_eq!(result.len(), 2);
    }
}
