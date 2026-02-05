//! Metal GPU-accelerated Multi-Scalar Multiplication (MSM).
//!
//! This module implements Pippenger's algorithm for MSM using Metal compute shaders.
//! The implementation uses:
//! - Signed digit recoding (NAF) to halve bucket count
//! - Montgomery arithmetic for efficient field operations
//! - Jacobian coordinates to avoid inversions during curve operations
//! - Parallel bucket accumulation on GPU
//!
//! # Algorithm Overview
//! 1. **Scalar recoding**: Convert scalars to signed digits (CPU)
//! 2. **Bucket accumulation**: Parallel point-to-bucket assignment (GPU)
//! 3. **Bucket reduction**: Sum buckets within each window (GPU)
//! 4. **Window combination**: Combine window results with Horner's method (CPU)
//!
//! # Performance
//! GPU acceleration provides significant speedups for large inputs (2^16+).
//! For smaller inputs, CPU implementation may be faster due to kernel dispatch overhead.

use metal::Buffer;

use super::abstractions::{
    errors::{MetalError, MetalResult},
    state::DynamicMetalState,
};

/// Metal MSM shader source code.
/// This is embedded at compile time for simplicity.
const MSM_SHADER_SOURCE: &str = include_str!("shaders/msm/msm.metal");

/// Configuration for MSM computation.
#[derive(Debug, Clone)]
pub struct MSMConfig {
    /// Window size in bits for Pippenger's algorithm.
    /// Larger windows = fewer windows but more buckets.
    pub window_size: usize,
    /// Number of limbs in the scalar representation.
    pub num_limbs: usize,
    /// Bits per limb (typically 64).
    pub bits_per_limb: usize,
}

impl MSMConfig {
    /// Creates a new MSM configuration for BLS12-381 (256-bit scalars).
    pub fn bls12_381() -> Self {
        Self {
            window_size: 16,
            num_limbs: 4,
            bits_per_limb: 64,
        }
    }

    /// Creates a new MSM configuration for BN254 (256-bit scalars).
    pub fn bn254() -> Self {
        Self {
            window_size: 16,
            num_limbs: 4,
            bits_per_limb: 64,
        }
    }

    /// Returns the optimal window size based on the number of points.
    pub fn optimal_window_size(num_points: usize) -> usize {
        match num_points {
            0..=4 => 2,
            5..=32 => 4,
            33..=128 => 6,
            129..=1024 => 8,
            1025..=4096 => 10,
            4097..=16384 => 12,
            16385..=65536 => 14,
            _ => 16,
        }
    }

    /// Returns the number of windows needed.
    pub fn num_windows(&self) -> usize {
        let total_bits = self.num_limbs * self.bits_per_limb;
        total_bits.div_ceil(self.window_size)
    }

    /// Returns the number of buckets per window (for signed representation).
    pub fn num_buckets(&self) -> usize {
        1 << (self.window_size - 1)
    }
}

impl Default for MSMConfig {
    fn default() -> Self {
        Self::bls12_381()
    }
}

/// Metal GPU MSM implementation.
pub struct MetalMSM {
    state: DynamicMetalState,
    config: MSMConfig,
    initialized: bool,
    max_threads_accumulation: u64,
    max_threads_reduction: u64,
}

impl MetalMSM {
    /// Creates a new MetalMSM instance.
    pub fn new(config: MSMConfig) -> MetalResult<Self> {
        let state = DynamicMetalState::new()?;
        Ok(Self {
            state,
            config,
            initialized: false,
            max_threads_accumulation: 0,
            max_threads_reduction: 0,
        })
    }

    /// Creates a new MetalMSM with default BLS12-381 configuration.
    pub fn new_bls12_381() -> MetalResult<Self> {
        Self::new(MSMConfig::bls12_381())
    }

    /// Initializes the Metal shaders. Must be called before compute().
    pub fn initialize(&mut self) -> MetalResult<()> {
        if self.initialized {
            return Ok(());
        }

        self.state.load_library(MSM_SHADER_SOURCE)?;
        self.max_threads_accumulation = self.state.prepare_pipeline("bucket_accumulation")?;
        self.max_threads_reduction = self.state.prepare_pipeline("bucket_reduction")?;
        self.initialized = true;
        Ok(())
    }

    /// Returns a reference to the underlying Metal state.
    pub fn state(&self) -> &DynamicMetalState {
        &self.state
    }

    /// Returns a mutable reference to the underlying Metal state.
    pub fn state_mut(&mut self) -> &mut DynamicMetalState {
        &mut self.state
    }

    /// Returns the current configuration.
    pub fn config(&self) -> &MSMConfig {
        &self.config
    }

    /// Updates the configuration.
    pub fn set_config(&mut self, config: MSMConfig) {
        self.config = config;
    }

    /// Computes MSM using Metal GPU acceleration.
    ///
    /// # Arguments
    /// * `scalars` - Flat array of scalar limbs [s0_l0, s0_l1, ..., s0_ln, s1_l0, ...]
    /// * `points` - Flat array of point coordinates [p0_x, p0_y, p0_z, p1_x, p1_y, p1_z, ...]
    ///   Each coordinate is represented as limbs in Montgomery form.
    ///
    /// # Returns
    /// The result point coordinates as a flat array [x, y, z] in the same format as input.
    ///
    /// # Note
    /// For small inputs (< 2^10), consider using CPU implementation instead.
    pub fn compute(&mut self, scalars: &[u64], points: &[u64]) -> MetalResult<Vec<u64>> {
        if !self.initialized {
            self.initialize()?;
        }

        let num_limbs = self.config.num_limbs;
        let num_scalars = scalars.len() / num_limbs;
        let coords_per_point = 3; // Jacobian: x, y, z
        let limbs_per_coord = num_limbs;
        let limbs_per_point = coords_per_point * limbs_per_coord;
        let num_points = points.len() / limbs_per_point;

        if num_scalars != num_points {
            return Err(MetalError::LengthMismatch(num_scalars, num_points));
        }

        if num_scalars == 0 {
            return Err(MetalError::EmptyInput);
        }

        // Step 1: Recode scalars to signed digits (CPU)
        let signed_digits = self.recode_scalars_signed(scalars, num_scalars);

        // Step 2: Create GPU buffers
        let scalars_buffer = self.state.alloc_buffer_with_data(&signed_digits)?;
        let points_buffer = self.state.alloc_buffer_with_data(points)?;

        // Create bucket buffers for each window
        let num_windows = self.config.num_windows();
        let num_buckets = self.config.num_buckets();
        let bucket_size = limbs_per_point; // Each bucket holds one Jacobian point

        // Output buffer for bucket sums
        let buckets_buffer = self.state.alloc_buffer(
            num_windows * num_buckets * bucket_size * std::mem::size_of::<u64>(),
        )?;

        // Step 3: Run bucket accumulation kernel
        self.run_bucket_accumulation(
            &scalars_buffer,
            &points_buffer,
            &buckets_buffer,
            num_scalars,
            num_windows,
            num_buckets,
        )?;

        // Step 4: Run bucket reduction kernel
        let window_sums_buffer = self
            .state
            .alloc_buffer(num_windows * bucket_size * std::mem::size_of::<u64>())?;

        self.run_bucket_reduction(
            &buckets_buffer,
            &window_sums_buffer,
            num_windows,
            num_buckets,
        )?;

        // Step 5: Read window sums back to CPU
        let window_sums: Vec<u64> = unsafe {
            self.state
                .read_buffer(&window_sums_buffer, num_windows * bucket_size)
        };

        // Step 6: Combine windows using Horner's method (CPU)
        // This is sequential and better done on CPU
        let result = self.combine_windows(&window_sums, num_windows, bucket_size);

        Ok(result)
    }

    /// Recodes scalars to signed digit representation.
    /// Uses NAF-like recoding to halve the bucket count.
    fn recode_scalars_signed(&self, scalars: &[u64], num_scalars: usize) -> Vec<i32> {
        let window_size = self.config.window_size;
        let num_windows = self.config.num_windows();
        let num_limbs = self.config.num_limbs;
        let half_bucket = 1i32 << (window_size - 1);
        let full_bucket = 1i32 << window_size;
        let mask = (1u64 << window_size) - 1;

        // Allocate flat storage for all digits
        let mut digits = vec![0i32; num_scalars * num_windows];

        for scalar_idx in 0..num_scalars {
            let scalar_base = scalar_idx * num_limbs;
            let mut carry = 0i32;

            for window_idx in 0..num_windows {
                // Extract window value from the scalar
                let bit_offset = window_idx * window_size;
                let limb_idx = bit_offset / 64;
                let bit_in_limb = bit_offset % 64;

                let raw_val = if limb_idx < num_limbs {
                    let limb = scalars[scalar_base + limb_idx];
                    let mut val = (limb >> bit_in_limb) & mask;

                    // Handle window spanning two limbs
                    if bit_in_limb + window_size > 64 && limb_idx + 1 < num_limbs {
                        let next_limb = scalars[scalar_base + limb_idx + 1];
                        let remaining_bits = bit_in_limb + window_size - 64;
                        val |= (next_limb & ((1u64 << remaining_bits) - 1)) << (64 - bit_in_limb);
                    }
                    val
                } else {
                    0
                };

                let window_val = raw_val as i32 + carry;

                // Convert to signed representation
                let digit = if window_val >= half_bucket {
                    carry = 1;
                    window_val - full_bucket
                } else {
                    carry = 0;
                    window_val
                };

                digits[scalar_idx * num_windows + window_idx] = digit;
            }
        }

        digits
    }

    /// Runs the bucket accumulation kernel on GPU.
    fn run_bucket_accumulation(
        &self,
        scalars_buffer: &Buffer,
        points_buffer: &Buffer,
        buckets_buffer: &Buffer,
        num_scalars: usize,
        num_windows: usize,
        num_buckets: usize,
    ) -> MetalResult<()> {
        // Create config buffer
        let config_data: [u32; 4] = [
            num_scalars as u32,
            num_windows as u32,
            num_buckets as u32,
            self.config.window_size as u32,
        ];
        let config_buffer = self.state.alloc_buffer_with_data(&config_data)?;

        // Dispatch threads: one thread per (scalar, window) pair
        let total_threads = (num_scalars * num_windows) as u64;

        self.state.execute_compute(
            "bucket_accumulation",
            &[scalars_buffer, points_buffer, buckets_buffer, &config_buffer],
            total_threads,
            self.max_threads_accumulation,
        )?;

        Ok(())
    }

    /// Runs the bucket reduction kernel on GPU.
    fn run_bucket_reduction(
        &self,
        buckets_buffer: &Buffer,
        window_sums_buffer: &Buffer,
        num_windows: usize,
        num_buckets: usize,
    ) -> MetalResult<()> {
        // Create config buffer
        let config_data: [u32; 2] = [num_windows as u32, num_buckets as u32];
        let config_buffer = self.state.alloc_buffer_with_data(&config_data)?;

        // One thread per window for reduction
        self.state.execute_compute(
            "bucket_reduction",
            &[buckets_buffer, window_sums_buffer, &config_buffer],
            num_windows as u64,
            self.max_threads_reduction,
        )?;

        Ok(())
    }

    /// Combines window sums using Horner's method.
    /// result = w[n-1] * 2^((n-1)*c) + w[n-2] * 2^((n-2)*c) + ... + w[0]
    ///        = ((w[n-1] * 2^c + w[n-2]) * 2^c + w[n-3]) * 2^c + ... + w[0]
    fn combine_windows(
        &self,
        window_sums: &[u64],
        num_windows: usize,
        point_size: usize,
    ) -> Vec<u64> {
        // For now, return the first window sum as a placeholder.
        // The actual implementation would perform point doubling and addition
        // using Jacobian coordinates on the CPU.
        //
        // TODO: Implement proper window combination with point arithmetic.
        // This requires implementing Jacobian point operations in Rust,
        // which should mirror the GPU implementation.

        if num_windows == 0 || window_sums.is_empty() {
            return vec![0u64; point_size];
        }

        // Return the first window for now (placeholder)
        window_sums[0..point_size].to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_msm_config_windows() {
        let config = MSMConfig {
            window_size: 8,
            num_limbs: 4,
            bits_per_limb: 64,
        };

        assert_eq!(config.num_windows(), 32); // 256 bits / 8 bits per window
        assert_eq!(config.num_buckets(), 128); // 2^(8-1) for signed
    }

    #[test]
    fn test_optimal_window_size() {
        assert_eq!(MSMConfig::optimal_window_size(1), 2);
        assert_eq!(MSMConfig::optimal_window_size(100), 6);
        assert_eq!(MSMConfig::optimal_window_size(10000), 12);
        assert_eq!(MSMConfig::optimal_window_size(100000), 16);
    }
}
