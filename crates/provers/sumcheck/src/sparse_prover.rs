//! Sparse Sumcheck Prover
//!
//! This module implements a sumcheck prover for sparse multilinear polynomials,
//! where the prover cost scales with the number of non-zero entries (k) rather
//! than the full polynomial size (2^n).
//!
//! Time complexity: O(n * k) where k is the number of non-zero evaluations
//! Space complexity: O(k)
//!
//! Based on the Spark approach from Lasso (a16z).

use crate::prover::{ProverError, ProverOutput};
use crate::Channel;
use lambdaworks_crypto::fiat_shamir::default_transcript::DefaultTranscript;
use lambdaworks_crypto::fiat_shamir::is_transcript::IsTranscript;
use lambdaworks_math::{
    field::{
        element::FieldElement,
        traits::{HasDefaultTranscript, IsField},
    },
    polynomial::Polynomial,
    traits::ByteConversion,
};
use std::collections::HashMap;
use std::ops::Mul;

#[cfg(feature = "parallel")]
#[allow(unused_imports)]
use rayon::prelude::*;

/// A sparse entry representing a non-zero evaluation at index `idx` with value `val`.
#[derive(Clone, Debug)]
pub struct SparseEntry<F: IsField>
where
    F::BaseType: Send + Sync,
{
    pub idx: usize,
    pub val: FieldElement<F>,
}

/// Sparse Sumcheck Prover that operates on sparse polynomial representations.
///
/// For polynomials with k << 2^n non-zero entries, this prover achieves
/// O(n * k) time complexity instead of O(n * 2^n).
///
/// The key insight is that we maintain a sparse representation throughout
/// the protocol. After each round, we update the sparse entries by "fixing"
/// the first remaining variable to the challenge value.
pub struct SparseProver<F: IsField>
where
    F::BaseType: Send + Sync,
{
    /// Number of variables remaining (decreases each round)
    remaining_vars: usize,
    /// Sparse entries, where idx is relative to remaining variables
    entries: Vec<SparseEntry<F>>,
    /// Current round (0-indexed)
    current_round: usize,
    /// Total number of original variables
    total_vars: usize,
}

impl<F: IsField> SparseProver<F>
where
    F::BaseType: Send + Sync,
    FieldElement<F>: Clone + Mul<Output = FieldElement<F>>,
{
    /// Creates a new SparseProver from sparse polynomial representations.
    ///
    /// Each entry is a (index, value) pair for non-zero evaluations.
    /// The index is interpreted as a binary string of length `num_vars`.
    pub fn new(
        num_vars: usize,
        entries: Vec<(usize, FieldElement<F>)>,
    ) -> Result<Self, ProverError> {
        if num_vars == 0 {
            return Err(ProverError::FactorMismatch(
                "Number of variables must be at least 1.".to_string(),
            ));
        }

        let sparse_entries: Vec<SparseEntry<F>> = entries
            .into_iter()
            .filter(|(_, val)| *val != FieldElement::zero())
            .map(|(idx, val)| SparseEntry { idx, val })
            .collect();

        Ok(Self {
            remaining_vars: num_vars,
            entries: sparse_entries,
            current_round: 0,
            total_vars: num_vars,
        })
    }

    /// Creates a SparseProver from a dense polynomial, extracting non-zero entries.
    pub fn from_dense(
        num_vars: usize,
        dense_evals: Vec<FieldElement<F>>,
    ) -> Result<Self, ProverError> {
        let entries: Vec<(usize, FieldElement<F>)> = dense_evals
            .into_iter()
            .enumerate()
            .filter(|(_, val)| *val != FieldElement::zero())
            .collect();

        Self::new(num_vars, entries)
    }

    /// Returns the total number of variables.
    pub fn num_vars(&self) -> usize {
        self.total_vars
    }

    /// Returns the number of non-zero entries.
    pub fn num_nonzero(&self) -> usize {
        self.entries.len()
    }

    /// Computes the initial claimed sum over the boolean hypercube.
    pub fn compute_initial_sum(&self) -> FieldElement<F> {
        self.entries
            .iter()
            .map(|e| e.val.clone())
            .fold(FieldElement::zero(), |a, b| a + b)
    }

    /// Executes a round of the sparse sumcheck protocol.
    ///
    /// The round polynomial g_j(X) is computed by grouping entries based on
    /// the value of the first variable (bit 0 vs bit 1), then interpolating.
    pub fn round(
        &mut self,
        r_prev: Option<&FieldElement<F>>,
    ) -> Result<Polynomial<FieldElement<F>>, ProverError> {
        // Apply previous challenge: collapse entries by fixing first variable
        if let Some(r) = r_prev {
            self.apply_challenge(r);
        }

        if self.current_round >= self.total_vars {
            return Err(ProverError::InvalidState(
                "All variables already fixed.".to_string(),
            ));
        }

        if self.remaining_vars == 0 {
            return Err(ProverError::InvalidState(
                "No remaining variables.".to_string(),
            ));
        }

        // Compute round polynomial by evaluating at t=0 and t=1
        // g(t) = sum over suffix of [val_0(suffix) + t * (val_1(suffix) - val_0(suffix))]
        //      = sum_0 + t * (sum_1 - sum_0)
        let (sum_0, sum_1) = self.compute_round_sums();

        self.current_round += 1;

        // Interpolate: polynomial is sum_0 + (sum_1 - sum_0) * X
        let eval_points = vec![FieldElement::zero(), FieldElement::one()];
        let evaluations = vec![sum_0, sum_1];

        Polynomial::interpolate(&eval_points, &evaluations)
            .map_err(ProverError::InterpolationError)
    }

    /// Computes the sums at X=0 and X=1 for the round polynomial.
    ///
    /// Groups entries by the most significant bit (first variable):
    /// - sum_0: sum of entries where first bit = 0
    /// - sum_1: sum of entries where first bit = 1
    fn compute_round_sums(&self) -> (FieldElement<F>, FieldElement<F>) {
        let half = 1 << (self.remaining_vars - 1);
        let mut sum_0 = FieldElement::zero();
        let mut sum_1 = FieldElement::zero();

        for entry in &self.entries {
            if entry.idx < half {
                // First bit is 0
                sum_0 += entry.val.clone();
            } else {
                // First bit is 1
                sum_1 += entry.val.clone();
            }
        }

        (sum_0, sum_1)
    }

    /// Applies a challenge by fixing the first variable to `r`.
    ///
    /// This collapses entries: for each pair (idx, idx+half), the new value is
    /// val_0 + r * (val_1 - val_0) = (1-r)*val_0 + r*val_1
    fn apply_challenge(&mut self, r: &FieldElement<F>) {
        if self.remaining_vars <= 1 {
            // No more variables to collapse
            if !self.entries.is_empty() {
                // Final evaluation at challenge point
                let (sum_0, sum_1) = self.compute_round_sums();
                let final_val = &sum_0 + r * &(&sum_1 - &sum_0);
                self.entries = vec![SparseEntry { idx: 0, val: final_val }];
            }
            self.remaining_vars = 0;
            return;
        }

        let half = 1 << (self.remaining_vars - 1);
        let one_minus_r = FieldElement::one() - r;

        // Group entries by their suffix (index with first bit stripped)
        let mut groups: HashMap<usize, (FieldElement<F>, FieldElement<F>)> = HashMap::new();

        for entry in &self.entries {
            let suffix = entry.idx & (half - 1);
            let is_high = entry.idx >= half;

            let (low, high) = groups
                .entry(suffix)
                .or_insert((FieldElement::zero(), FieldElement::zero()));

            if is_high {
                *high = high.clone() + entry.val.clone();
            } else {
                *low = low.clone() + entry.val.clone();
            }
        }

        // Compute new entries: new_val = (1-r)*low + r*high
        let new_entries: Vec<SparseEntry<F>> = groups
            .into_iter()
            .filter_map(|(suffix, (low, high))| {
                let new_val = &one_minus_r * &low + r * &high;
                if new_val != FieldElement::zero() {
                    Some(SparseEntry {
                        idx: suffix,
                        val: new_val,
                    })
                } else {
                    None
                }
            })
            .collect();

        self.entries = new_entries;
        self.remaining_vars -= 1;
    }

    /// Returns the final evaluation (after all rounds).
    pub fn get_final_evaluation(&self) -> Option<FieldElement<F>> {
        if self.remaining_vars == 0 && self.entries.len() == 1 {
            Some(self.entries[0].val.clone())
        } else if self.remaining_vars == 0 && self.entries.is_empty() {
            Some(FieldElement::zero())
        } else {
            None
        }
    }
}

/// Proves a sumcheck for a single sparse polynomial.
///
/// Time complexity: O(n * k) where k is the number of non-zero entries.
pub fn prove_sparse<F>(
    num_vars: usize,
    sparse_evals: Vec<(usize, FieldElement<F>)>,
) -> ProverOutput<F>
where
    F: IsField + HasDefaultTranscript,
    F::BaseType: Send + Sync,
    FieldElement<F>: Clone + Mul<Output = FieldElement<F>> + ByteConversion,
{
    let mut prover = SparseProver::new(num_vars, sparse_evals)?;

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

/// Sparse multi-factor prover for product sumcheck.
///
/// For products of sparse polynomials where the intersection of supports is small.
pub struct SparseMultiFactorProver<F: IsField>
where
    F::BaseType: Send + Sync,
{
    #[allow(dead_code)]
    remaining_vars: usize,
    total_vars: usize,
    /// Sparse entries for each factor
    factor_entries: Vec<Vec<SparseEntry<F>>>,
    #[allow(dead_code)]
    current_round: usize,
}

impl<F: IsField> SparseMultiFactorProver<F>
where
    F::BaseType: Send + Sync,
    FieldElement<F>: Clone + Mul<Output = FieldElement<F>>,
{
    /// Creates a new multi-factor sparse prover.
    pub fn new(
        num_vars: usize,
        factors: Vec<Vec<(usize, FieldElement<F>)>>,
    ) -> Result<Self, ProverError> {
        if factors.is_empty() {
            return Err(ProverError::FactorMismatch(
                "At least one factor required.".to_string(),
            ));
        }

        let factor_entries: Vec<Vec<SparseEntry<F>>> = factors
            .into_iter()
            .map(|entries| {
                entries
                    .into_iter()
                    .filter(|(_, val)| *val != FieldElement::zero())
                    .map(|(idx, val)| SparseEntry { idx, val })
                    .collect()
            })
            .collect();

        Ok(Self {
            remaining_vars: num_vars,
            total_vars: num_vars,
            factor_entries,
            current_round: 0,
        })
    }

    /// Returns the number of variables.
    pub fn num_vars(&self) -> usize {
        self.total_vars
    }

    /// Computes the initial sum of products over the hypercube.
    pub fn compute_initial_sum(&self) -> FieldElement<F> {
        // Build index maps for efficient lookup
        let maps: Vec<HashMap<usize, FieldElement<F>>> = self
            .factor_entries
            .iter()
            .map(|entries| {
                entries
                    .iter()
                    .map(|e| (e.idx, e.val.clone()))
                    .collect()
            })
            .collect();

        // Find all indices that appear in at least one factor
        let mut all_indices: Vec<usize> = maps
            .iter()
            .flat_map(|m| m.keys().cloned())
            .collect();
        all_indices.sort_unstable();
        all_indices.dedup();

        // Sum products over all indices
        let mut sum = FieldElement::zero();
        for idx in all_indices {
            let mut product = FieldElement::one();
            let mut all_present = true;

            for map in &maps {
                if let Some(val) = map.get(&idx) {
                    product *= val.clone();
                } else {
                    all_present = false;
                    break;
                }
            }

            if all_present {
                sum += product;
            }
        }

        sum
    }
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
    fn test_sparse_prover_all_nonzero() {
        // All entries non-zero - should match dense
        let entries: Vec<(usize, FE)> = vec![
            (0, FE::from(1)),
            (1, FE::from(2)),
            (2, FE::from(3)),
            (3, FE::from(4)),
        ];

        let prover = SparseProver::<F>::new(2, entries).unwrap();
        let sum = prover.compute_initial_sum();

        // Sum should be 1 + 2 + 3 + 4 = 10
        assert_eq!(sum, FE::from(10));
    }

    #[test]
    fn test_sparse_prover_some_zero() {
        // Only some entries non-zero
        let entries: Vec<(usize, FE)> = vec![(0, FE::from(5)), (3, FE::from(7))];

        let prover = SparseProver::<F>::new(2, entries).unwrap();
        let sum = prover.compute_initial_sum();

        // Sum should be 5 + 0 + 0 + 7 = 12
        assert_eq!(sum, FE::from(12));
    }

    #[test]
    fn test_sparse_prover_single_entry() {
        let entries: Vec<(usize, FE)> = vec![(5, FE::from(42))];

        let prover = SparseProver::<F>::new(3, entries).unwrap();
        let sum = prover.compute_initial_sum();

        assert_eq!(sum, FE::from(42));
    }

    #[test]
    fn test_sparse_from_dense() {
        let dense = vec![FE::from(0), FE::from(2), FE::from(0), FE::from(4)];

        let prover = SparseProver::<F>::from_dense(2, dense).unwrap();

        assert_eq!(prover.num_nonzero(), 2);
        assert_eq!(prover.compute_initial_sum(), FE::from(6));
    }

    #[test]
    fn test_sparse_round_sums() {
        // Entries at indices 0, 1, 2, 3 with values 1, 2, 3, 4
        // First bit 0: indices 0, 1 -> sum_0 = 1 + 2 = 3
        // First bit 1: indices 2, 3 -> sum_1 = 3 + 4 = 7
        let entries: Vec<(usize, FE)> = vec![
            (0, FE::from(1)),
            (1, FE::from(2)),
            (2, FE::from(3)),
            (3, FE::from(4)),
        ];

        let prover = SparseProver::<F>::new(2, entries).unwrap();
        let (sum_0, sum_1) = prover.compute_round_sums();

        assert_eq!(sum_0, FE::from(3));
        assert_eq!(sum_1, FE::from(7));
    }

    #[test]
    fn test_sparse_apply_challenge() {
        // Start with entries at all 4 positions with values 1, 2, 3, 4
        let entries: Vec<(usize, FE)> = vec![
            (0, FE::from(1)),
            (1, FE::from(2)),
            (2, FE::from(3)),
            (3, FE::from(4)),
        ];

        let mut prover = SparseProver::<F>::new(2, entries).unwrap();

        // Apply challenge r = 2
        // New entries should be: (1-r)*old_0 + r*old_high
        // For suffix 0: (1-2)*1 + 2*3 = -1 + 6 = 5 (mod 101)
        // For suffix 1: (1-2)*2 + 2*4 = -2 + 8 = 6 (mod 101)
        let r = FE::from(2);
        prover.apply_challenge(&r);

        assert_eq!(prover.remaining_vars, 1);
        assert_eq!(prover.num_nonzero(), 2);

        // Check the values
        let sum = prover.compute_initial_sum();
        // sum = 5 + 6 = 11 (mod 101)
        // Or: (1-2)*(1+2) + 2*(3+4) = -3 + 14 = 11
        assert_eq!(sum, FE::from(11));
    }

    #[test]
    fn test_sparse_proof_verifies() {
        // Create a sparse polynomial and verify the proof
        let entries: Vec<(usize, FE)> = vec![
            (0, FE::from(1)),
            (1, FE::from(2)),
            (2, FE::from(3)),
            (3, FE::from(4)),
        ];

        let (claimed_sum, proof_polys) = prove_sparse::<F>(2, entries.clone()).unwrap();

        // Claimed sum should be 1 + 2 + 3 + 4 = 10
        assert_eq!(claimed_sum, FE::from(10));

        // Verify using dense verification
        let dense_evals: Vec<FE> = (0..4)
            .map(|i| {
                entries
                    .iter()
                    .find(|(idx, _)| *idx == i)
                    .map(|(_, v)| v.clone())
                    .unwrap_or(FE::zero())
            })
            .collect();
        let dense_poly = DenseMultilinearPolynomial::new(dense_evals);

        let result = crate::verify(2, claimed_sum, proof_polys, vec![dense_poly]);
        assert!(result.unwrap(), "Sparse proof should verify");
    }

    #[test]
    fn test_sparse_proof_with_zeros() {
        // Sparse polynomial with only a few non-zero entries
        let entries: Vec<(usize, FE)> = vec![(1, FE::from(5)), (6, FE::from(7))];

        let (claimed_sum, proof_polys) = prove_sparse::<F>(3, entries.clone()).unwrap();

        // Sum should be 5 + 7 = 12
        assert_eq!(claimed_sum, FE::from(12));

        // Create dense version for verification
        let dense_evals: Vec<FE> = (0..8)
            .map(|i| {
                entries
                    .iter()
                    .find(|(idx, _)| *idx == i)
                    .map(|(_, v)| v.clone())
                    .unwrap_or(FE::zero())
            })
            .collect();
        let dense_poly = DenseMultilinearPolynomial::new(dense_evals);

        let result = crate::verify(3, claimed_sum, proof_polys, vec![dense_poly]);
        assert!(result.unwrap(), "Sparse proof with zeros should verify");
    }

    #[test]
    fn test_sparse_matches_dense() {
        // Test that sparse and dense provers produce the same claimed sum
        let entries: Vec<(usize, FE)> = vec![
            (0, FE::from(1)),
            (2, FE::from(3)),
            (5, FE::from(6)),
            (7, FE::from(8)),
        ];

        let (sparse_sum, _) = prove_sparse::<F>(3, entries.clone()).unwrap();

        // Create dense version
        let dense_evals: Vec<FE> = (0..8)
            .map(|i| {
                entries
                    .iter()
                    .find(|(idx, _)| *idx == i)
                    .map(|(_, v)| v.clone())
                    .unwrap_or(FE::zero())
            })
            .collect();
        let dense_poly = DenseMultilinearPolynomial::new(dense_evals);

        let (dense_sum, _) = crate::prove_optimized(vec![dense_poly]).unwrap();

        assert_eq!(sparse_sum, dense_sum, "Sparse and dense should have same sum");
    }

    #[test]
    fn test_multi_factor_initial_sum() {
        // Two sparse factors
        let factor1: Vec<(usize, FE)> = vec![(0, FE::from(1)), (1, FE::from(2))];

        let factor2: Vec<(usize, FE)> = vec![(0, FE::from(3)), (1, FE::from(4))];

        let prover = SparseMultiFactorProver::<F>::new(1, vec![factor1, factor2]).unwrap();
        let sum = prover.compute_initial_sum();

        // Product at index 0: 1 * 3 = 3
        // Product at index 1: 2 * 4 = 8
        // Sum = 3 + 8 = 11
        assert_eq!(sum, FE::from(11));
    }

    #[test]
    fn test_multi_factor_sparse_intersection() {
        // Factors with partial overlap
        let factor1: Vec<(usize, FE)> = vec![(0, FE::from(1)), (1, FE::from(2)), (2, FE::from(3))];

        let factor2: Vec<(usize, FE)> = vec![(1, FE::from(4)), (2, FE::from(5)), (3, FE::from(6))];

        let prover = SparseMultiFactorProver::<F>::new(2, vec![factor1, factor2]).unwrap();
        let sum = prover.compute_initial_sum();

        // Only indices 1 and 2 have both factors non-zero
        // Product at index 1: 2 * 4 = 8
        // Product at index 2: 3 * 5 = 15
        // Sum = 8 + 15 = 23
        assert_eq!(sum, FE::from(23));
    }
}
