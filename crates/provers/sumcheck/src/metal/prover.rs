//! Metal GPU Prover for Sumcheck
//!
//! Provides GPU-accelerated sumcheck proving using Apple's Metal API.

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

#[cfg(target_os = "macos")]
use super::shaders::SUMCHECK_SHADER_SOURCE;

/// Represents the Metal GPU state and resources.
///
/// This is a placeholder for the actual Metal implementation.
/// When the `metal` feature is enabled and running on macOS,
/// this will hold the Metal device, command queue, and compiled pipelines.
#[cfg(target_os = "macos")]
pub struct MetalState {
    /// Whether Metal is available on this system
    available: bool,
    /// Minimum polynomial size to use GPU (smaller uses CPU)
    min_gpu_size: usize,
}

#[cfg(not(target_os = "macos"))]
pub struct MetalState {
    available: bool,
    min_gpu_size: usize,
}

impl Default for MetalState {
    fn default() -> Self {
        Self::new()
    }
}

impl MetalState {
    /// Creates a new MetalState, initializing GPU resources if available.
    #[cfg(target_os = "macos")]
    pub fn new() -> Self {
        // In a full implementation, this would:
        // 1. Get the default Metal device
        // 2. Create command queue
        // 3. Compile shaders
        // 4. Create compute pipelines
        //
        // For now, we provide a stub that falls back to CPU
        Self {
            available: false, // Set to true when Metal crate is integrated
            min_gpu_size: 1 << 14, // 16K elements minimum for GPU benefit
        }
    }

    #[cfg(not(target_os = "macos"))]
    pub fn new() -> Self {
        Self {
            available: false,
            min_gpu_size: 1 << 14,
        }
    }

    /// Returns whether Metal GPU acceleration is available.
    pub fn is_available(&self) -> bool {
        self.available
    }

    /// Returns the minimum polynomial size for GPU to be beneficial.
    pub fn min_gpu_size(&self) -> usize {
        self.min_gpu_size
    }
}

/// Metal-accelerated Sumcheck Prover.
///
/// This prover automatically uses GPU acceleration when beneficial,
/// falling back to CPU for small polynomials or when Metal is unavailable.
pub struct MetalProver<F: IsField>
where
    F::BaseType: Send + Sync,
{
    num_vars: usize,
    /// Working evaluations (on CPU or GPU)
    evals: Vec<FieldElement<F>>,
    /// Current round
    current_round: usize,
    /// Metal state (if available)
    metal_state: MetalState,
    /// Whether we're using GPU for this proof
    using_gpu: bool,
}

impl<F: IsField> MetalProver<F>
where
    F::BaseType: Send + Sync,
    FieldElement<F>: Clone + Mul<Output = FieldElement<F>>,
{
    /// Creates a new MetalProver.
    ///
    /// Automatically determines whether to use GPU based on polynomial size
    /// and Metal availability.
    pub fn new(poly: DenseMultilinearPolynomial<F>) -> Result<Self, ProverError> {
        let num_vars = poly.num_vars();
        let evals = poly.evals().clone();
        let metal_state = MetalState::new();

        // Use GPU only if available and polynomial is large enough
        let using_gpu = metal_state.is_available() && evals.len() >= metal_state.min_gpu_size();

        Ok(Self {
            num_vars,
            evals,
            current_round: 0,
            metal_state,
            using_gpu,
        })
    }

    /// Creates a MetalProver with explicit GPU usage control.
    pub fn with_gpu_preference(
        poly: DenseMultilinearPolynomial<F>,
        prefer_gpu: bool,
    ) -> Result<Self, ProverError> {
        let num_vars = poly.num_vars();
        let evals = poly.evals().clone();
        let metal_state = MetalState::new();

        let using_gpu = prefer_gpu && metal_state.is_available();

        Ok(Self {
            num_vars,
            evals,
            current_round: 0,
            metal_state,
            using_gpu,
        })
    }

    /// Returns the number of variables.
    pub fn num_vars(&self) -> usize {
        self.num_vars
    }

    /// Returns whether GPU is being used.
    pub fn is_using_gpu(&self) -> bool {
        self.using_gpu
    }

    /// Computes the initial sum over the boolean hypercube.
    pub fn compute_initial_sum(&self) -> FieldElement<F> {
        if self.using_gpu {
            self.compute_initial_sum_gpu()
        } else {
            self.compute_initial_sum_cpu()
        }
    }

    fn compute_initial_sum_cpu(&self) -> FieldElement<F> {
        self.evals
            .iter()
            .cloned()
            .fold(FieldElement::zero(), |a, b| a + b)
    }

    fn compute_initial_sum_gpu(&self) -> FieldElement<F> {
        // GPU implementation would:
        // 1. Upload evals to GPU buffer
        // 2. Run parallel_sum kernel
        // 3. Download result

        // For now, fall back to CPU
        self.compute_initial_sum_cpu()
    }

    /// Executes a round of the sumcheck protocol.
    pub fn round(
        &mut self,
        r_prev: Option<&FieldElement<F>>,
    ) -> Result<Polynomial<FieldElement<F>>, ProverError> {
        // Apply previous challenge
        if let Some(r) = r_prev {
            if self.using_gpu {
                self.apply_challenge_gpu(r);
            } else {
                self.apply_challenge_cpu(r);
            }
        }

        if self.current_round >= self.num_vars {
            return Err(ProverError::InvalidState(
                "All variables already fixed.".to_string(),
            ));
        }

        let half = self.evals.len() / 2;

        // Compute round sums
        let (sum_0, sum_1) = if self.using_gpu {
            self.compute_round_sums_gpu(half)
        } else {
            self.compute_round_sums_cpu(half)
        };

        self.current_round += 1;

        // Interpolate round polynomial
        let eval_points = vec![FieldElement::zero(), FieldElement::one()];
        let evaluations = vec![sum_0, sum_1];

        Polynomial::interpolate(&eval_points, &evaluations)
            .map_err(|e| ProverError::InterpolationError(e))
    }

    fn compute_round_sums_cpu(&self, half: usize) -> (FieldElement<F>, FieldElement<F>) {
        let mut sum_0 = FieldElement::zero();
        let mut sum_1 = FieldElement::zero();

        for k in 0..half {
            sum_0 = sum_0 + self.evals[k].clone();
            sum_1 = sum_1 + self.evals[k + half].clone();
        }

        (sum_0, sum_1)
    }

    fn compute_round_sums_gpu(&self, half: usize) -> (FieldElement<F>, FieldElement<F>) {
        // GPU implementation would:
        // 1. Run compute_round_sums kernel
        // 2. Download sum_0 and sum_1

        // For now, fall back to CPU
        self.compute_round_sums_cpu(half)
    }

    fn apply_challenge_cpu(&mut self, r: &FieldElement<F>) {
        let half = self.evals.len() / 2;
        let one_minus_r = FieldElement::one() - r;

        for k in 0..half {
            let v0 = &self.evals[k];
            let v1 = &self.evals[k + half];
            self.evals[k] = &one_minus_r * v0 + r * v1;
        }

        self.evals.truncate(half);
    }

    fn apply_challenge_gpu(&mut self, r: &FieldElement<F>) {
        // GPU implementation would:
        // 1. Upload challenge to GPU
        // 2. Run apply_challenge kernel
        // 3. Update GPU buffer (no download needed until final round)

        // For now, fall back to CPU
        self.apply_challenge_cpu(r);
    }
}

/// Proves a sumcheck using Metal GPU acceleration when available.
///
/// Automatically falls back to optimized CPU implementation when:
/// - Running on non-macOS systems
/// - Metal is not available
/// - Polynomial is too small for GPU benefit
pub fn prove_metal<F>(poly: DenseMultilinearPolynomial<F>) -> ProverOutput<F>
where
    F: IsField + HasDefaultTranscript,
    F::BaseType: Send + Sync,
    FieldElement<F>: Clone + Mul<Output = FieldElement<F>> + ByteConversion,
{
    let num_vars = poly.num_vars();
    let mut prover = MetalProver::new(poly)?;

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

/// Multi-factor Metal prover for product sumcheck.
pub struct MetalMultiFactorProver<F: IsField>
where
    F::BaseType: Send + Sync,
{
    num_vars: usize,
    /// Working evaluations for each factor
    factor_evals: Vec<Vec<FieldElement<F>>>,
    /// Current round
    current_round: usize,
    /// Metal state
    metal_state: MetalState,
    /// Whether using GPU
    using_gpu: bool,
}

impl<F: IsField> MetalMultiFactorProver<F>
where
    F::BaseType: Send + Sync,
    FieldElement<F>: Clone + Mul<Output = FieldElement<F>>,
{
    /// Creates a new multi-factor Metal prover.
    pub fn new(factors: Vec<DenseMultilinearPolynomial<F>>) -> Result<Self, ProverError> {
        if factors.is_empty() {
            return Err(ProverError::FactorMismatch(
                "At least one polynomial factor is required.".to_string(),
            ));
        }

        let num_vars = factors[0].num_vars();
        for (i, f) in factors.iter().enumerate().skip(1) {
            if f.num_vars() != num_vars {
                return Err(ProverError::FactorMismatch(format!(
                    "Factor {} has {} variables, expected {}",
                    i,
                    f.num_vars(),
                    num_vars
                )));
            }
        }

        let factor_evals: Vec<Vec<FieldElement<F>>> =
            factors.into_iter().map(|p| p.evals().clone()).collect();

        let metal_state = MetalState::new();
        let using_gpu =
            metal_state.is_available() && factor_evals[0].len() >= metal_state.min_gpu_size();

        Ok(Self {
            num_vars,
            factor_evals,
            current_round: 0,
            metal_state,
            using_gpu,
        })
    }

    /// Returns the number of variables.
    pub fn num_vars(&self) -> usize {
        self.num_vars
    }

    /// Returns the number of factors.
    pub fn num_factors(&self) -> usize {
        self.factor_evals.len()
    }

    /// Computes the initial sum of products over the hypercube.
    pub fn compute_initial_sum(&self) -> FieldElement<F> {
        let size = self.factor_evals[0].len();
        let mut sum = FieldElement::zero();

        for i in 0..size {
            let mut product = FieldElement::one();
            for factor in &self.factor_evals {
                product = product * factor[i].clone();
            }
            sum = sum + product;
        }

        sum
    }

    /// Executes a round of the sumcheck protocol.
    pub fn round(
        &mut self,
        r_prev: Option<&FieldElement<F>>,
    ) -> Result<Polynomial<FieldElement<F>>, ProverError> {
        // Apply previous challenge to all factors
        if let Some(r) = r_prev {
            let one_minus_r = FieldElement::one() - r;
            for factor in &mut self.factor_evals {
                let half = factor.len() / 2;
                for k in 0..half {
                    let v0 = &factor[k];
                    let v1 = &factor[k + half];
                    factor[k] = &one_minus_r * v0 + r * v1;
                }
                factor.truncate(half);
            }
        }

        if self.current_round >= self.num_vars {
            return Err(ProverError::InvalidState(
                "All variables already fixed.".to_string(),
            ));
        }

        let half = self.factor_evals[0].len() / 2;
        let num_factors = self.factor_evals.len();
        let num_eval_points = num_factors + 1;

        // Compute evaluations at points 0, 1, ..., d
        let mut evaluations = vec![FieldElement::zero(); num_eval_points];

        for t in 0..num_eval_points {
            let t_fe = FieldElement::from(t as u64);
            let mut sum = FieldElement::zero();

            for k in 0..half {
                let mut product = FieldElement::one();
                for factor in &self.factor_evals {
                    let v0 = &factor[k];
                    let v1 = &factor[k + half];
                    // Interpolate: v(t) = v0 + t * (v1 - v0)
                    let v_t = v0 + &t_fe * &(v1 - v0);
                    product = product * v_t;
                }
                sum = sum + product;
            }

            evaluations[t] = sum;
        }

        self.current_round += 1;

        // Interpolate
        let eval_points: Vec<FieldElement<F>> = (0..num_eval_points)
            .map(|i| FieldElement::from(i as u64))
            .collect();

        Polynomial::interpolate(&eval_points, &evaluations)
            .map_err(|e| ProverError::InterpolationError(e))
    }
}

/// Proves a multi-factor sumcheck using Metal GPU acceleration.
pub fn prove_metal_multi<F>(factors: Vec<DenseMultilinearPolynomial<F>>) -> ProverOutput<F>
where
    F: IsField + HasDefaultTranscript,
    F::BaseType: Send + Sync,
    FieldElement<F>: Clone + Mul<Output = FieldElement<F>> + ByteConversion,
{
    let num_factors = factors.len();
    let num_vars = factors[0].num_vars();
    let mut prover = MetalMultiFactorProver::new(factors)?;

    let claimed_sum = prover.compute_initial_sum();

    let mut transcript = DefaultTranscript::<F>::default();
    transcript.append_bytes(b"initial_sum");
    transcript.append_felt(&FieldElement::from(num_vars as u64));
    transcript.append_felt(&FieldElement::from(num_factors as u64));
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
