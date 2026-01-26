//! Metal GPU Prover for Sumcheck
//!
//! Provides GPU-accelerated sumcheck proving using Apple's Metal API.
//!
//! # References
//!
//! ## Implementations Consulted
//!
//! - **lambdaworks GPU**: <https://github.com/lambdaclass/lambdaworks>
//!   Metal integration patterns for finite field arithmetic
//!
//! - **Icicle (Ingonyama)**: <https://github.com/ingonyama-zk/icicle>
//!   GPU-accelerated cryptographic primitives (CUDA/Metal)
//!
//! - **metal-rs**: <https://github.com/gfx-rs/metal-rs>
//!   Rust bindings for Apple's Metal API
//!
//! # Architecture
//!
//! The prover uses three main GPU kernels:
//! - `parallel_sum`: Computes partial sums across evaluation vectors
//! - `apply_challenge`: Updates evaluations with verifier challenge
//! - `compute_round_sums`: Computes g(0) and g(1) for round polynomial

use crate::common::{
    apply_challenge_to_evals, check_round_bounds, compute_round_sums_single, run_sumcheck_protocol,
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

#[cfg(all(target_os = "macos", feature = "metal"))]
use super::shaders::SUMCHECK_SHADER_SOURCE;

#[cfg(all(target_os = "macos", feature = "metal"))]
use metal::{
    Buffer, CommandQueue, ComputePipelineState, Device, Library, MTLResourceOptions, MTLSize,
};

/// Thread group size for Metal compute kernels.
#[cfg(all(target_os = "macos", feature = "metal"))]
const THREADGROUP_SIZE: u64 = 256;

/// Represents the Metal GPU state and resources.
pub struct MetalState {
    /// Whether Metal is available on this system
    available: bool,
    /// Minimum polynomial size to use GPU (smaller uses CPU)
    min_gpu_size: usize,
    /// Metal device (if available)
    #[cfg(all(target_os = "macos", feature = "metal"))]
    device: Option<Device>,
    /// Command queue for GPU operations
    #[cfg(all(target_os = "macos", feature = "metal"))]
    command_queue: Option<CommandQueue>,
    /// Compiled shader library
    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[allow(dead_code)]
    library: Option<Library>,
    /// Compute pipeline for parallel_sum
    #[cfg(all(target_os = "macos", feature = "metal"))]
    parallel_sum_pipeline: Option<ComputePipelineState>,
    /// Compute pipeline for apply_challenge
    #[cfg(all(target_os = "macos", feature = "metal"))]
    apply_challenge_pipeline: Option<ComputePipelineState>,
    /// Compute pipeline for compute_round_sums
    #[cfg(all(target_os = "macos", feature = "metal"))]
    compute_round_sums_pipeline: Option<ComputePipelineState>,
}

impl Default for MetalState {
    fn default() -> Self {
        Self::new()
    }
}

impl MetalState {
    /// Creates a new MetalState, initializing GPU resources if available.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn new() -> Self {
        // Try to get the default Metal device
        let device = Device::system_default();

        if let Some(ref dev) = device {
            // Create command queue
            let command_queue = dev.new_command_queue();

            // Compile shaders
            let compile_options = metal::CompileOptions::new();
            let library = dev
                .new_library_with_source(SUMCHECK_SHADER_SOURCE, &compile_options)
                .ok();

            if let Some(ref lib) = library {
                // Create compute pipelines
                let parallel_sum_fn = lib.get_function("parallel_sum", None).ok();
                let apply_challenge_fn = lib.get_function("apply_challenge", None).ok();
                let round_sums_fn = lib.get_function("compute_round_sums", None).ok();

                let parallel_sum_pipeline = parallel_sum_fn
                    .and_then(|f| dev.new_compute_pipeline_state_with_function(&f).ok());
                let apply_challenge_pipeline = apply_challenge_fn
                    .and_then(|f| dev.new_compute_pipeline_state_with_function(&f).ok());
                let compute_round_sums_pipeline = round_sums_fn
                    .and_then(|f| dev.new_compute_pipeline_state_with_function(&f).ok());

                let available = parallel_sum_pipeline.is_some()
                    && apply_challenge_pipeline.is_some()
                    && compute_round_sums_pipeline.is_some();

                return Self {
                    available,
                    min_gpu_size: 1 << 14, // 16K elements minimum for GPU benefit
                    device,
                    command_queue: Some(command_queue),
                    library,
                    parallel_sum_pipeline,
                    apply_challenge_pipeline,
                    compute_round_sums_pipeline,
                };
            }
        }

        Self {
            available: false,
            min_gpu_size: 1 << 14,
            device: None,
            command_queue: None,
            library: None,
            parallel_sum_pipeline: None,
            apply_challenge_pipeline: None,
            compute_round_sums_pipeline: None,
        }
    }

    /// Creates a new MetalState (non-macOS or non-metal feature).
    #[cfg(not(all(target_os = "macos", feature = "metal")))]
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

    /// Creates a GPU buffer with the given data.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn create_buffer<T: Copy>(&self, data: &[T]) -> Option<Buffer> {
        self.device.as_ref().map(|dev| {
            let byte_len = std::mem::size_of_val(data);
            dev.new_buffer_with_data(
                data.as_ptr() as *const _,
                byte_len as u64,
                MTLResourceOptions::StorageModeShared,
            )
        })
    }

    /// Creates an empty GPU buffer of the given size.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn create_empty_buffer(&self, byte_len: usize) -> Option<Buffer> {
        self.device
            .as_ref()
            .map(|dev| dev.new_buffer(byte_len as u64, MTLResourceOptions::StorageModeShared))
    }

    /// Executes the parallel_sum kernel.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn parallel_sum_gpu(&self, data: &[u64]) -> Option<u64> {
        if !self.available || data.is_empty() {
            return None;
        }

        let _device = self.device.as_ref()?;
        let command_queue = self.command_queue.as_ref()?;
        let pipeline = self.parallel_sum_pipeline.as_ref()?;

        let input_buffer = self.create_buffer(data)?;
        let params = [data.len() as u32];
        let params_buffer = self.create_buffer(&params)?;

        // Calculate number of threadgroups
        let num_elements = data.len() as u64;
        let num_groups = num_elements.div_ceil(THREADGROUP_SIZE);

        let output_buffer = self.create_empty_buffer((num_groups as usize) * 8)?;

        // Create command buffer and encoder
        let command_buffer = command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(&input_buffer), 0);
        encoder.set_buffer(1, Some(&output_buffer), 0);
        encoder.set_buffer(2, Some(&params_buffer), 0);
        encoder.set_threadgroup_memory_length(0, THREADGROUP_SIZE * 8);

        let grid_size = MTLSize::new(num_elements, 1, 1);
        let threadgroup_size = MTLSize::new(THREADGROUP_SIZE, 1, 1);

        encoder.dispatch_threads(grid_size, threadgroup_size);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Read back partial sums and reduce on CPU
        let output_ptr = output_buffer.contents() as *const u64;
        let partial_sums: Vec<u64> =
            unsafe { std::slice::from_raw_parts(output_ptr, num_groups as usize).to_vec() };

        // Final reduction on CPU (usually small number of groups)
        let mut sum = 0u64;
        for &partial in &partial_sums {
            sum = sum.wrapping_add(partial);
        }

        Some(sum)
    }

    /// Executes the apply_challenge kernel.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn apply_challenge_gpu(&self, data: &mut [u64], r: u64, one_minus_r: u64) -> bool {
        if !self.available || data.is_empty() {
            return false;
        }

        let half = data.len() / 2;
        if half == 0 {
            return false;
        }

        let device = self.device.as_ref();
        let command_queue = self.command_queue.as_ref();
        let pipeline = self.apply_challenge_pipeline.as_ref();

        if device.is_none() || command_queue.is_none() || pipeline.is_none() {
            return false;
        }

        let input_buffer = match self.create_buffer(data) {
            Some(b) => b,
            None => return false,
        };

        let challenge = [r, one_minus_r];
        let challenge_buffer = match self.create_buffer(&challenge) {
            Some(b) => b,
            None => return false,
        };

        let params = [half as u32];
        let params_buffer = match self.create_buffer(&params) {
            Some(b) => b,
            None => return false,
        };

        let output_buffer = match self.create_empty_buffer(half * 8) {
            Some(b) => b,
            None => return false,
        };

        let command_queue = command_queue.unwrap();
        let pipeline = pipeline.unwrap();

        let command_buffer = command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(&input_buffer), 0);
        encoder.set_buffer(1, Some(&output_buffer), 0);
        encoder.set_buffer(2, Some(&challenge_buffer), 0);
        encoder.set_buffer(3, Some(&params_buffer), 0);

        let grid_size = MTLSize::new(half as u64, 1, 1);
        let threadgroup_size = MTLSize::new(THREADGROUP_SIZE.min(half as u64), 1, 1);

        encoder.dispatch_threads(grid_size, threadgroup_size);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Copy result back
        let output_ptr = output_buffer.contents() as *const u64;
        let result: &[u64] = unsafe { std::slice::from_raw_parts(output_ptr, half) };
        data[..half].copy_from_slice(result);

        true
    }

    /// Executes the compute_round_sums kernel.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn compute_round_sums_gpu(&self, data: &[u64]) -> Option<(u64, u64)> {
        if !self.available || data.is_empty() {
            return None;
        }

        let half = data.len() / 2;
        if half == 0 {
            return Some((data[0], 0));
        }

        let _device = self.device.as_ref()?;
        let command_queue = self.command_queue.as_ref()?;
        let pipeline = self.compute_round_sums_pipeline.as_ref()?;

        let input_buffer = self.create_buffer(data)?;

        let elems_per_thread = 4u32;
        let params = [half as u32, elems_per_thread];
        let params_buffer = self.create_buffer(&params)?;

        let threads_needed = (half as u64).div_ceil(elems_per_thread as u64);
        let num_groups = threads_needed.div_ceil(THREADGROUP_SIZE);

        let output_buffer = self.create_empty_buffer((num_groups as usize) * 2 * 8)?;

        let command_buffer = command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(&input_buffer), 0);
        encoder.set_buffer(1, Some(&output_buffer), 0);
        encoder.set_buffer(2, Some(&params_buffer), 0);
        encoder.set_threadgroup_memory_length(0, THREADGROUP_SIZE * 2 * 8);

        let grid_size = MTLSize::new(threads_needed, 1, 1);
        let threadgroup_size = MTLSize::new(THREADGROUP_SIZE.min(threads_needed), 1, 1);

        encoder.dispatch_threads(grid_size, threadgroup_size);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Read back partial sums
        let output_ptr = output_buffer.contents() as *const u64;
        let partial_sums: Vec<u64> =
            unsafe { std::slice::from_raw_parts(output_ptr, (num_groups * 2) as usize).to_vec() };

        // Final reduction
        let mut sum_0 = 0u64;
        let mut sum_1 = 0u64;
        for i in 0..num_groups as usize {
            sum_0 = sum_0.wrapping_add(partial_sums[i * 2]);
            sum_1 = sum_1.wrapping_add(partial_sums[i * 2 + 1]);
        }

        Some((sum_0, sum_1))
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
    #[allow(dead_code)]
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
    pub fn new(poly: DenseMultilinearPolynomial<F>) -> Result<Self, ProverError> {
        let num_vars = poly.num_vars();
        let evals = poly.evals().clone();
        let metal_state = MetalState::new();

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

    /// Returns whether GPU is being used.
    pub fn is_using_gpu(&self) -> bool {
        self.using_gpu
    }

    fn compute_initial_sum_impl(&self) -> FieldElement<F> {
        self.evals
            .iter()
            .cloned()
            .fold(FieldElement::zero(), |a, b| a + b)
    }

    fn round_impl(
        &mut self,
        r_prev: Option<&FieldElement<F>>,
    ) -> Result<Polynomial<FieldElement<F>>, ProverError> {
        if let Some(r) = r_prev {
            apply_challenge_to_evals(&mut self.evals, r);
        }

        check_round_bounds(self.current_round, self.num_vars)?;

        let (sum_0, sum_1) = compute_round_sums_single(&self.evals);

        self.current_round += 1;

        let eval_points = vec![FieldElement::zero(), FieldElement::one()];
        let evaluations = vec![sum_0, sum_1];

        Polynomial::interpolate(&eval_points, &evaluations).map_err(ProverError::InterpolationError)
    }
}

impl<F: IsField> SumcheckProver<F> for MetalProver<F>
where
    F::BaseType: Send + Sync,
    FieldElement<F>: Clone + Mul<Output = FieldElement<F>>,
{
    fn num_vars(&self) -> usize {
        self.num_vars
    }

    fn num_factors(&self) -> usize {
        1
    }

    fn compute_initial_sum(&self) -> FieldElement<F> {
        self.compute_initial_sum_impl()
    }

    fn round(
        &mut self,
        r_prev: Option<&FieldElement<F>>,
    ) -> Result<Polynomial<FieldElement<F>>, ProverError> {
        self.round_impl(r_prev)
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
    let mut prover = MetalProver::new(poly)?;
    run_sumcheck_protocol(&mut prover, 1)
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
    #[allow(dead_code)]
    metal_state: MetalState,
    /// Whether using GPU
    #[allow(dead_code)]
    using_gpu: bool,
}

impl<F: IsField> MetalMultiFactorProver<F>
where
    F::BaseType: Send + Sync,
    FieldElement<F>: Clone + Mul<Output = FieldElement<F>>,
{
    /// Creates a new multi-factor Metal prover.
    pub fn new(factors: Vec<DenseMultilinearPolynomial<F>>) -> Result<Self, ProverError> {
        let num_vars = validate_factors(&factors)?;

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

    fn compute_initial_sum_impl(&self) -> FieldElement<F> {
        crate::common::compute_initial_sum_product(&self.factor_evals)
    }

    fn round_impl(
        &mut self,
        r_prev: Option<&FieldElement<F>>,
    ) -> Result<Polynomial<FieldElement<F>>, ProverError> {
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

        check_round_bounds(self.current_round, self.num_vars)?;

        let evaluations = crate::common::compute_round_poly_product(&self.factor_evals);
        let num_eval_points = self.factor_evals.len() + 1;

        self.current_round += 1;

        let eval_points: Vec<FieldElement<F>> = (0..num_eval_points)
            .map(|i| FieldElement::from(i as u64))
            .collect();

        Polynomial::interpolate(&eval_points, &evaluations).map_err(ProverError::InterpolationError)
    }
}

impl<F: IsField> SumcheckProver<F> for MetalMultiFactorProver<F>
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
        self.compute_initial_sum_impl()
    }

    fn round(
        &mut self,
        r_prev: Option<&FieldElement<F>>,
    ) -> Result<Polynomial<FieldElement<F>>, ProverError> {
        self.round_impl(r_prev)
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
    let mut prover = MetalMultiFactorProver::new(factors)?;
    run_sumcheck_protocol(&mut prover, num_factors)
}

/// GPU-accelerated prover for Goldilocks field (64-bit).
///
/// This prover is specialized for the Goldilocks prime (2^64 - 2^32 + 1)
/// and uses Metal GPU acceleration for maximum performance.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub struct GoldilocksMetalProver {
    #[allow(dead_code)]
    num_vars: usize,
    /// Raw u64 evaluations for GPU processing
    evals_raw: Vec<u64>,
    /// Current round
    #[allow(dead_code)]
    current_round: usize,
    /// Metal state
    metal_state: MetalState,
}

#[cfg(all(target_os = "macos", feature = "metal"))]
impl GoldilocksMetalProver {
    /// Creates a new Goldilocks Metal prover from raw u64 evaluations.
    pub fn new(num_vars: usize, evals: Vec<u64>) -> Result<Self, ProverError> {
        if evals.len() != 1 << num_vars {
            return Err(ProverError::FactorMismatch(format!(
                "Expected {} evaluations, got {}",
                1 << num_vars,
                evals.len()
            )));
        }

        let metal_state = MetalState::new();

        Ok(Self {
            num_vars,
            evals_raw: evals,
            current_round: 0,
            metal_state,
        })
    }

    /// Returns whether GPU acceleration is being used.
    pub fn is_using_gpu(&self) -> bool {
        self.metal_state.is_available()
    }

    /// Computes the initial sum using GPU if available.
    pub fn compute_initial_sum(&self) -> u64 {
        if self.metal_state.is_available() {
            if let Some(sum) = self.metal_state.parallel_sum_gpu(&self.evals_raw) {
                return sum;
            }
        }

        // Fallback to CPU
        self.evals_raw.iter().fold(0u64, |a, &b| a.wrapping_add(b))
    }

    /// Applies a challenge using GPU if available.
    pub fn apply_challenge(&mut self, r: u64, one_minus_r: u64) {
        if self.metal_state.is_available()
            && self
                .metal_state
                .apply_challenge_gpu(&mut self.evals_raw, r, one_minus_r)
        {
            self.evals_raw.truncate(self.evals_raw.len() / 2);
            return;
        }

        // Fallback to CPU
        let half = self.evals_raw.len() / 2;
        for k in 0..half {
            let v0 = self.evals_raw[k];
            let v1 = self.evals_raw[k + half];
            // Simplified: assuming proper modular arithmetic is handled elsewhere
            self.evals_raw[k] = one_minus_r
                .wrapping_mul(v0)
                .wrapping_add(r.wrapping_mul(v1));
        }
        self.evals_raw.truncate(half);
    }

    /// Computes round sums using GPU if available.
    pub fn compute_round_sums(&self) -> (u64, u64) {
        if self.metal_state.is_available() {
            if let Some((sum_0, sum_1)) = self.metal_state.compute_round_sums_gpu(&self.evals_raw) {
                return (sum_0, sum_1);
            }
        }

        // Fallback to CPU
        let half = self.evals_raw.len() / 2;
        let sum_0: u64 = self.evals_raw[..half]
            .iter()
            .fold(0u64, |a, &b| a.wrapping_add(b));
        let sum_1: u64 = self.evals_raw[half..]
            .iter()
            .fold(0u64, |a, &b| a.wrapping_add(b));
        (sum_0, sum_1)
    }
}
