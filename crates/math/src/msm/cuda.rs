//! CUDA GPU-accelerated Multi-Scalar Multiplication (MSM).
//!
//! This module implements Pippenger's algorithm for MSM using CUDA compute kernels.
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
//! # Known Limitations
//! - Only BLS12-381 is supported. Constants are hardcoded.
//! - CPU-side MSM logic is duplicated from the Metal MSM module. A follow-up
//!   should extract shared code.

use cudarc::driver::{safe::CudaSlice, CudaDevice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::safe::Ptx;
use lambdaworks_gpu::cuda::abstractions::errors::CudaError;
use std::sync::Arc;

const BLS12_381_MSM_PTX: &str = include_str!("../gpu/cuda/shaders/msm/bls12_381_msm.ptx");
const MSM_DESCRIPTOR_INDEX_MASK: usize = 0x7fff_ffff;
const SEGMENT_ACCUMULATION_CHUNK_SIZE: usize = 256;
const BUCKET_REDUCTION_CHUNK_SIZE: usize = 256;

/// Configuration for MSM computation.
///
/// Currently only BLS12-381 is supported. Both the CUDA shader and
/// CPU-side arithmetic hardcode BLS12-381 field constants.
#[derive(Debug, Clone)]
pub struct MSMConfig {
    /// Window size in bits for Pippenger's algorithm.
    pub window_size: usize,
    /// Number of limbs in the scalar representation (e.g., 4 for 256-bit Fr).
    pub num_limbs: usize,
    /// Bits per limb (typically 64).
    pub bits_per_limb: usize,
    /// Number of limbs per point coordinate (e.g., 6 for BLS12-381 Fq).
    pub point_coord_limbs: usize,
}

impl MSMConfig {
    /// Creates a new MSM configuration for BLS12-381.
    pub fn bls12_381() -> Self {
        Self {
            window_size: 16,
            num_limbs: 4,
            bits_per_limb: 64,
            point_coord_limbs: 6,
        }
    }

    /// Maximum supported window size.
    pub const MAX_WINDOW_SIZE: usize = 30;

    /// Returns the number of windows needed.
    pub fn num_windows(&self) -> usize {
        assert!(
            self.window_size > 0 && self.window_size <= Self::MAX_WINDOW_SIZE,
            "window_size must be in 1..={}",
            Self::MAX_WINDOW_SIZE
        );
        let total_bits = self.num_limbs * self.bits_per_limb;
        total_bits.div_ceil(self.window_size)
    }

    /// Returns the number of buckets per window (for signed representation).
    pub fn num_buckets(&self) -> usize {
        assert!(
            self.window_size > 0 && self.window_size <= Self::MAX_WINDOW_SIZE,
            "window_size must be in 1..={}",
            Self::MAX_WINDOW_SIZE
        );
        1 << (self.window_size - 1)
    }
}

impl Default for MSMConfig {
    fn default() -> Self {
        Self::bls12_381()
    }
}

/// CUDA GPU MSM implementation.
pub struct CudaMSM {
    device: Arc<CudaDevice>,
    config: MSMConfig,
}

fn checked_u32_len(value: usize, name: &str) -> Result<u32, CudaError> {
    u32::try_from(value)
        .map_err(|_| CudaError::FunctionError(format!("{name} {value} exceeds u32::MAX")))
}

fn exclusive_scan_counts(counts: &[u32]) -> Result<(Vec<u32>, u32), CudaError> {
    let mut offsets = Vec::with_capacity(counts.len() + 1);
    let mut running = 0u32;

    for &count in counts {
        offsets.push(running);
        running = running.checked_add(count).ok_or_else(|| {
            CudaError::FunctionError("Bucket descriptor count overflowed u32".to_string())
        })?;
    }

    offsets.push(running);
    Ok((offsets, running))
}

fn build_segment_tasks(counts: &[u32]) -> Result<(Vec<u32>, Vec<u32>, Vec<u32>, u32), CudaError> {
    let mut task_offsets = Vec::with_capacity(counts.len() + 1);
    let mut task_starts = Vec::new();
    let mut task_lengths = Vec::new();
    let mut descriptor_offset = 0u32;
    let mut task_count = 0u32;

    for &count in counts {
        task_offsets.push(task_count);

        let mut consumed = 0u32;
        while consumed < count {
            let len = (count - consumed).min(SEGMENT_ACCUMULATION_CHUNK_SIZE as u32);
            task_starts.push(descriptor_offset + consumed);
            task_lengths.push(len);
            consumed += len;
            task_count = task_count.checked_add(1).ok_or_else(|| {
                CudaError::FunctionError("Segment task count overflowed u32".to_string())
            })?;
        }

        descriptor_offset = descriptor_offset.checked_add(count).ok_or_else(|| {
            CudaError::FunctionError("Segment descriptor offset overflowed u32".to_string())
        })?;
    }

    task_offsets.push(task_count);
    Ok((task_offsets, task_starts, task_lengths, task_count))
}

impl CudaMSM {
    /// Creates a new CudaMSM instance with the given configuration.
    pub fn new(config: MSMConfig) -> Result<Self, CudaError> {
        let device =
            CudaDevice::new(0).map_err(|err| CudaError::DeviceNotFound(err.to_string()))?;

        device
            .load_ptx(
                Ptx::from_src(BLS12_381_MSM_PTX),
                "bls12_381_msm",
                &[
                    "count_bucket_entries_bls12_381",
                    "scatter_bucket_descriptors_bls12_381",
                    "segmented_bucket_accumulation_bls12_381",
                    "partial_segment_accumulation_bls12_381",
                    "finalize_segment_accumulation_bls12_381",
                    "partial_bucket_reduction_bls12_381",
                    "finalize_bucket_reduction_bls12_381",
                    "bucket_reduction_bls12_381",
                ],
            )
            .map_err(|err| CudaError::PtxError(err.to_string()))?;

        Ok(Self { device, config })
    }

    /// Creates a new CudaMSM with default BLS12-381 configuration.
    pub fn new_bls12_381() -> Result<Self, CudaError> {
        Self::new(MSMConfig::bls12_381())
    }

    /// Returns the current configuration.
    pub fn config(&self) -> &MSMConfig {
        &self.config
    }

    /// Computes MSM using CUDA GPU acceleration.
    ///
    /// # Arguments
    /// * `scalars` - Flat array of scalar limbs [s0_l0, s0_l1, ..., s0_ln, s1_l0, ...]
    /// * `points` - Flat array of point coordinates [p0_x, p0_y, p0_z, p1_x, p1_y, p1_z, ...]
    ///   Each coordinate is represented as limbs in Montgomery form.
    ///
    /// # Returns
    /// The result point coordinates as a flat array [x, y, z] in the same format as input.
    pub fn compute(&self, scalars: &[u64], points: &[u64]) -> Result<Vec<u64>, CudaError> {
        let num_limbs = self.config.num_limbs;
        let limbs_per_coord = self.config.point_coord_limbs;
        let limbs_per_point = 3 * limbs_per_coord;

        if limbs_per_coord != COORD_LIMBS {
            return Err(CudaError::FunctionError(format!(
                "Expected {} coord limbs, got {}",
                COORD_LIMBS, limbs_per_coord
            )));
        }

        if scalars.len() % num_limbs != 0 {
            return Err(CudaError::FunctionError(format!(
                "Scalars length {} is not a multiple of {}",
                scalars.len(),
                num_limbs
            )));
        }
        if points.len() % limbs_per_point != 0 {
            return Err(CudaError::FunctionError(format!(
                "Points length {} is not a multiple of {}",
                points.len(),
                limbs_per_point
            )));
        }

        let num_scalars = scalars.len() / num_limbs;
        let num_points = points.len() / limbs_per_point;

        if num_scalars != num_points {
            return Err(CudaError::FunctionError(format!(
                "Scalars count {} != points count {}",
                num_scalars, num_points
            )));
        }

        if num_scalars == 0 {
            return Err(CudaError::FunctionError("Empty input".to_string()));
        }

        let num_buckets = self.config.num_buckets();
        let effective_windows = self.config.num_windows() + 1;

        if num_scalars > MSM_DESCRIPTOR_INDEX_MASK {
            return Err(CudaError::FunctionError(format!(
                "Scalars count {} exceeds descriptor capacity {}",
                num_scalars, MSM_DESCRIPTOR_INDEX_MASK
            )));
        }

        let total_bucket_keys = effective_windows.checked_mul(num_buckets).ok_or_else(|| {
            CudaError::FunctionError("Bucket key count overflowed usize".to_string())
        })?;
        let total_digits = num_scalars.checked_mul(effective_windows).ok_or_else(|| {
            CudaError::FunctionError("Signed digit count overflowed usize".to_string())
        })?;
        let total_bucket_elements =
            total_bucket_keys
                .checked_mul(limbs_per_point)
                .ok_or_else(|| {
                    CudaError::FunctionError("Bucket buffer length overflowed usize".to_string())
                })?;

        let num_scalars_u32 = checked_u32_len(num_scalars, "num_scalars")?;
        let effective_windows_u32 = checked_u32_len(effective_windows, "effective_windows")?;
        let num_buckets_u32 = checked_u32_len(num_buckets, "num_buckets")?;
        let total_digits_u32 = checked_u32_len(total_digits, "total_digits")?;
        let total_bucket_keys_u32 = checked_u32_len(total_bucket_keys, "total_bucket_keys")?;
        let window_size_u32 = checked_u32_len(self.config.window_size, "window_size")?;

        // Step 1: Recode scalars to signed digits (CPU)
        let signed_digits = self.recode_scalars_signed(scalars, num_scalars);

        // Step 2: Count and compact nonzero signed digits into bucket segments
        let scalars_buf = self
            .device
            .htod_sync_copy(&signed_digits)
            .map_err(|err| CudaError::AllocateMemory(err.to_string()))?;

        let points_buf = self
            .device
            .htod_sync_copy(points)
            .map_err(|err| CudaError::AllocateMemory(err.to_string()))?;

        let config_data: [u32; 4] = [
            num_scalars_u32,
            effective_windows_u32,
            num_buckets_u32,
            window_size_u32,
        ];
        let config_buf = self
            .device
            .htod_sync_copy(&config_data)
            .map_err(|err| CudaError::AllocateMemory(err.to_string()))?;

        let mut counts_buf: CudaSlice<u32> = self
            .device
            .alloc_zeros(total_bucket_keys)
            .map_err(|err| CudaError::AllocateMemory(err.to_string()))?;

        let count_func = self
            .device
            .get_func("bls12_381_msm", "count_bucket_entries_bls12_381")
            .ok_or_else(|| {
                CudaError::FunctionError("count_bucket_entries_bls12_381".to_string())
            })?;

        let block_size = 256u32;
        let digit_grid_size = total_digits_u32.div_ceil(block_size);
        let digit_launch_config = LaunchConfig {
            grid_dim: (digit_grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            count_func.launch(
                digit_launch_config,
                (&scalars_buf, &mut counts_buf, &config_buf),
            )
        }
        .map_err(|err| CudaError::Launch(err.to_string()))?;

        let counts = self
            .device
            .sync_reclaim(counts_buf)
            .map_err(|err| CudaError::RetrieveMemory(err.to_string()))?;
        let (offsets, nonzero_items) = exclusive_scan_counts(&counts)?;
        let (task_offsets, task_starts, task_lengths, segment_task_count) =
            build_segment_tasks(&counts)?;
        let sorted_descriptor_len = (nonzero_items as usize).max(1);
        let segment_task_len = (segment_task_count as usize).max(1);
        let segment_task_count_u32 =
            checked_u32_len(segment_task_count as usize, "segment_task_count")?;
        let task_starts_upload = if task_starts.is_empty() {
            vec![0]
        } else {
            task_starts
        };
        let task_lengths_upload = if task_lengths.is_empty() {
            vec![0]
        } else {
            task_lengths
        };
        let partial_bucket_elements =
            segment_task_len
                .checked_mul(limbs_per_point)
                .ok_or_else(|| {
                    CudaError::FunctionError("Partial segment buffer overflowed usize".to_string())
                })?;

        let offsets_buf = self
            .device
            .htod_sync_copy(&offsets)
            .map_err(|err| CudaError::AllocateMemory(err.to_string()))?;
        let mut write_counts_buf: CudaSlice<u32> = self
            .device
            .alloc_zeros(total_bucket_keys)
            .map_err(|err| CudaError::AllocateMemory(err.to_string()))?;
        let mut sorted_descriptors_buf: CudaSlice<u32> = self
            .device
            .alloc_zeros(sorted_descriptor_len)
            .map_err(|err| CudaError::AllocateMemory(err.to_string()))?;

        let scatter_func = self
            .device
            .get_func("bls12_381_msm", "scatter_bucket_descriptors_bls12_381")
            .ok_or_else(|| {
                CudaError::FunctionError("scatter_bucket_descriptors_bls12_381".to_string())
            })?;

        unsafe {
            scatter_func.launch(
                digit_launch_config,
                (
                    &scalars_buf,
                    &mut write_counts_buf,
                    &offsets_buf,
                    &mut sorted_descriptors_buf,
                    &config_buf,
                ),
            )
        }
        .map_err(|err| CudaError::Launch(err.to_string()))?;

        let mut buckets_buf: CudaSlice<u64> = self
            .device
            .alloc_zeros(total_bucket_elements)
            .map_err(|err| CudaError::AllocateMemory(err.to_string()))?;

        let task_offsets_buf = self
            .device
            .htod_sync_copy(&task_offsets)
            .map_err(|err| CudaError::AllocateMemory(err.to_string()))?;
        let task_starts_buf = self
            .device
            .htod_sync_copy(&task_starts_upload)
            .map_err(|err| CudaError::AllocateMemory(err.to_string()))?;
        let task_lengths_buf = self
            .device
            .htod_sync_copy(&task_lengths_upload)
            .map_err(|err| CudaError::AllocateMemory(err.to_string()))?;
        let segment_config_data: [u32; 4] = [segment_task_count_u32, total_bucket_keys_u32, 0, 0];
        let segment_config_buf = self
            .device
            .htod_sync_copy(&segment_config_data)
            .map_err(|err| CudaError::AllocateMemory(err.to_string()))?;
        let mut partial_buckets_buf: CudaSlice<u64> = self
            .device
            .alloc_zeros(partial_bucket_elements)
            .map_err(|err| CudaError::AllocateMemory(err.to_string()))?;

        let partial_segment_func = self
            .device
            .get_func("bls12_381_msm", "partial_segment_accumulation_bls12_381")
            .ok_or_else(|| {
                CudaError::FunctionError("partial_segment_accumulation_bls12_381".to_string())
            })?;
        let finalize_segment_func = self
            .device
            .get_func("bls12_381_msm", "finalize_segment_accumulation_bls12_381")
            .ok_or_else(|| {
                CudaError::FunctionError("finalize_segment_accumulation_bls12_381".to_string())
            })?;

        let segment_task_grid_size = segment_task_count_u32.div_ceil(block_size).max(1);
        let segment_task_launch_config = LaunchConfig {
            grid_dim: (segment_task_grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            partial_segment_func.launch(
                segment_task_launch_config,
                (
                    &sorted_descriptors_buf,
                    &task_starts_buf,
                    &task_lengths_buf,
                    &points_buf,
                    &mut partial_buckets_buf,
                    &segment_config_buf,
                ),
            )
        }
        .map_err(|err| CudaError::Launch(err.to_string()))?;

        let bucket_grid_size = total_bucket_keys_u32.div_ceil(block_size);
        let bucket_launch_config = LaunchConfig {
            grid_dim: (bucket_grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            finalize_segment_func.launch(
                bucket_launch_config,
                (
                    &task_offsets_buf,
                    &mut partial_buckets_buf,
                    &mut buckets_buf,
                    &segment_config_buf,
                ),
            )
        }
        .map_err(|err| CudaError::Launch(err.to_string()))?;

        // Step 3: Run bucket reduction kernel
        let window_sums_data = vec![0u64; effective_windows * limbs_per_point];
        let mut window_sums_buf = self
            .device
            .htod_sync_copy(&window_sums_data)
            .map_err(|err| CudaError::AllocateMemory(err.to_string()))?;

        let reduction_chunk_size = BUCKET_REDUCTION_CHUNK_SIZE.min(num_buckets);
        let chunks_per_window = num_buckets.div_ceil(reduction_chunk_size);
        let total_reduction_chunks = effective_windows
            .checked_mul(chunks_per_window)
            .ok_or_else(|| {
                CudaError::FunctionError("Reduction chunk count overflowed usize".to_string())
            })?;
        let reduction_chunk_size_u32 =
            checked_u32_len(reduction_chunk_size, "reduction_chunk_size")?;
        let chunks_per_window_u32 = checked_u32_len(chunks_per_window, "chunks_per_window")?;
        let total_reduction_chunks_u32 =
            checked_u32_len(total_reduction_chunks, "total_reduction_chunks")?;

        let reduction_config_data: [u32; 4] = [
            effective_windows_u32,
            num_buckets_u32,
            reduction_chunk_size_u32,
            chunks_per_window_u32,
        ];
        let reduction_config_buf = self
            .device
            .htod_sync_copy(&reduction_config_data)
            .map_err(|err| CudaError::AllocateMemory(err.to_string()))?;

        let partial_buffer_elements = total_reduction_chunks
            .checked_mul(limbs_per_point)
            .ok_or_else(|| {
                CudaError::FunctionError("Partial reduction buffer overflowed usize".to_string())
            })?;
        let mut partial_sums_buf: CudaSlice<u64> = self
            .device
            .alloc_zeros(partial_buffer_elements)
            .map_err(|err| CudaError::AllocateMemory(err.to_string()))?;
        let mut partial_results_buf: CudaSlice<u64> = self
            .device
            .alloc_zeros(partial_buffer_elements)
            .map_err(|err| CudaError::AllocateMemory(err.to_string()))?;

        let partial_reduce_func = self
            .device
            .get_func("bls12_381_msm", "partial_bucket_reduction_bls12_381")
            .ok_or_else(|| {
                CudaError::FunctionError("partial_bucket_reduction_bls12_381".to_string())
            })?;
        let finalize_reduce_func = self
            .device
            .get_func("bls12_381_msm", "finalize_bucket_reduction_bls12_381")
            .ok_or_else(|| {
                CudaError::FunctionError("finalize_bucket_reduction_bls12_381".to_string())
            })?;

        let reduction_grid_size = total_reduction_chunks_u32.div_ceil(block_size);
        let partial_reduce_config = LaunchConfig {
            grid_dim: (reduction_grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            partial_reduce_func.launch(
                partial_reduce_config,
                (
                    &mut buckets_buf,
                    &mut partial_sums_buf,
                    &mut partial_results_buf,
                    &reduction_config_buf,
                ),
            )
        }
        .map_err(|err| CudaError::Launch(err.to_string()))?;

        let finalize_reduce_config = LaunchConfig {
            grid_dim: (effective_windows_u32.div_ceil(block_size), 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            finalize_reduce_func.launch(
                finalize_reduce_config,
                (
                    &mut partial_sums_buf,
                    &mut partial_results_buf,
                    &mut window_sums_buf,
                    &reduction_config_buf,
                ),
            )
        }
        .map_err(|err| CudaError::Launch(err.to_string()))?;

        // Step 5: Read window sums back to CPU
        let window_sums = self
            .device
            .sync_reclaim(window_sums_buf)
            .map_err(|err| CudaError::RetrieveMemory(err.to_string()))?;

        // Step 6: Combine windows using Horner's method (CPU)
        let result = self.combine_windows(&window_sums, effective_windows, limbs_per_point);

        Ok(result)
    }

    /// Recodes scalars to signed digit representation.
    fn recode_scalars_signed(&self, scalars: &[u64], num_scalars: usize) -> Vec<i32> {
        let window_size = self.config.window_size;
        let num_windows = self.config.num_windows();
        let num_limbs = self.config.num_limbs;
        let half_bucket = 1i32 << (window_size - 1);
        let full_bucket = 1i32 << window_size;
        let mask = (1u64 << window_size) - 1;

        let effective_windows = num_windows + 1;
        let mut digits = vec![0i32; num_scalars * effective_windows];

        for scalar_idx in 0..num_scalars {
            let scalar_base = scalar_idx * num_limbs;
            let mut carry = 0i32;

            for window_idx in 0..num_windows {
                let bit_offset = window_idx * window_size;
                let limb_idx = bit_offset / 64;
                let bit_in_limb = bit_offset % 64;

                let raw_val = if limb_idx < num_limbs {
                    let limb = scalars[scalar_base + limb_idx];
                    let mut val = (limb >> bit_in_limb) & mask;

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

                let digit = if window_val >= half_bucket {
                    carry = 1;
                    window_val - full_bucket
                } else {
                    carry = 0;
                    window_val
                };

                digits[scalar_idx * effective_windows + window_idx] = digit;
            }

            digits[scalar_idx * effective_windows + num_windows] = carry;
        }

        digits
    }

    /// Combines window sums using Horner's method.
    fn combine_windows(
        &self,
        window_sums: &[u64],
        num_windows: usize,
        point_size: usize,
    ) -> Vec<u64> {
        if num_windows == 0 || window_sums.is_empty() {
            return vec![0u64; point_size];
        }

        let window_size = self.config.window_size;

        let base = (num_windows - 1) * point_size;
        let mut result = JacobianPoint::from_limbs(&window_sums[base..base + point_size]);

        for window_idx in (0..num_windows - 1).rev() {
            for _ in 0..window_size {
                result = result.double();
            }
            let base = window_idx * point_size;
            let window = JacobianPoint::from_limbs(&window_sums[base..base + point_size]);
            result = result.add(&window);
        }

        result.to_limbs()
    }
}

// =============================================================================
// CPU-side Jacobian Point Arithmetic for Window Combination
// =============================================================================

/// Number of 64-bit limbs for BLS12-381 base field Fq coordinates.
const COORD_LIMBS: usize = 6;

/// Jacobian point representation for CPU-side arithmetic.
#[derive(Clone, Debug)]
struct JacobianPoint {
    x: [u64; COORD_LIMBS],
    y: [u64; COORD_LIMBS],
    z: [u64; COORD_LIMBS],
}

/// BLS12-381 prime field modulus (little-endian limbs)
const BLS12_381_P: [u64; COORD_LIMBS] = [
    0xb9feffffffffaaab,
    0x1eabfffeb153ffff,
    0x6730d2a0f6b0f624,
    0x64774b84f38512bf,
    0x4b1ba7b6434bacd7,
    0x1a0111ea397fe69a,
];

/// Montgomery parameter: -p^(-1) mod 2^64
const BLS12_381_INV: u64 = 0x89f3fffcfffcfffd;

impl JacobianPoint {
    fn identity() -> Self {
        Self {
            x: [0; COORD_LIMBS],
            y: [0; COORD_LIMBS],
            z: [0; COORD_LIMBS],
        }
    }

    fn from_limbs(limbs: &[u64]) -> Self {
        assert!(
            limbs.len() >= COORD_LIMBS * 3,
            "from_limbs requires at least {} limbs, got {}",
            COORD_LIMBS * 3,
            limbs.len()
        );
        let mut x = [0u64; COORD_LIMBS];
        let mut y = [0u64; COORD_LIMBS];
        let mut z = [0u64; COORD_LIMBS];

        x.copy_from_slice(&limbs[0..COORD_LIMBS]);
        y.copy_from_slice(&limbs[COORD_LIMBS..COORD_LIMBS * 2]);
        z.copy_from_slice(&limbs[COORD_LIMBS * 2..COORD_LIMBS * 3]);

        Self { x, y, z }
    }

    fn to_limbs(&self) -> Vec<u64> {
        let mut result = Vec::with_capacity(COORD_LIMBS * 3);
        result.extend_from_slice(&self.x);
        result.extend_from_slice(&self.y);
        result.extend_from_slice(&self.z);
        result
    }

    fn is_identity(&self) -> bool {
        self.z.iter().all(|&limb| limb == 0)
    }

    /// Point doubling using 2009-l formula from EFD.
    fn double(&self) -> Self {
        if self.is_identity() {
            return self.clone();
        }

        let a = mont_square(&self.x);
        let b = mont_square(&self.y);
        let c = mont_square(&b);

        let tmp = field_add(&self.x, &b);
        let tmp = mont_square(&tmp);
        let tmp = field_sub(&tmp, &a);
        let tmp = field_sub(&tmp, &c);
        let d = field_double(&tmp);

        let e = field_add(&a, &field_double(&a));
        let f = mont_square(&e);

        let x3 = field_sub(&f, &field_double(&d));

        let y3 = field_sub(&d, &x3);
        let y3 = mont_mul(&e, &y3);
        let c8 = field_double(&field_double(&field_double(&c)));
        let y3 = field_sub(&y3, &c8);

        let z3 = mont_mul(&self.y, &self.z);
        let z3 = field_double(&z3);

        Self {
            x: x3,
            y: y3,
            z: z3,
        }
    }

    /// Point addition using 2007-bl formula from EFD.
    fn add(&self, other: &Self) -> Self {
        if self.is_identity() {
            return other.clone();
        }
        if other.is_identity() {
            return self.clone();
        }

        let z1z1 = mont_square(&self.z);
        let z2z2 = mont_square(&other.z);

        let u1 = mont_mul(&self.x, &z2z2);
        let u2 = mont_mul(&other.x, &z1z1);

        let s1 = mont_mul(&self.y, &other.z);
        let s1 = mont_mul(&s1, &z2z2);
        let s2 = mont_mul(&other.y, &self.z);
        let s2 = mont_mul(&s2, &z1z1);

        let h = field_sub(&u2, &u1);

        let zero = [0u64; COORD_LIMBS];
        if h == zero {
            let s_diff = field_sub(&s2, &s1);
            if s_diff == zero {
                return self.double();
            } else {
                return Self::identity();
            }
        }

        let i = field_double(&h);
        let i = mont_square(&i);
        let j = mont_mul(&h, &i);

        let r = field_sub(&s2, &s1);
        let r = field_double(&r);

        let v = mont_mul(&u1, &i);

        let x3 = mont_square(&r);
        let x3 = field_sub(&x3, &j);
        let x3 = field_sub(&x3, &field_double(&v));

        let y3 = field_sub(&v, &x3);
        let y3 = mont_mul(&r, &y3);
        let tmp = mont_mul(&s1, &j);
        let tmp = field_double(&tmp);
        let y3 = field_sub(&y3, &tmp);

        let z3 = field_add(&self.z, &other.z);
        let z3 = mont_square(&z3);
        let z3 = field_sub(&z3, &z1z1);
        let z3 = field_sub(&z3, &z2z2);
        let z3 = mont_mul(&z3, &h);

        Self {
            x: x3,
            y: y3,
            z: z3,
        }
    }
}

// =============================================================================
// Field Arithmetic (CPU-side, matching GPU implementation)
// =============================================================================

fn bigint_add(a: &[u64; COORD_LIMBS], b: &[u64; COORD_LIMBS]) -> ([u64; COORD_LIMBS], u64) {
    let mut result = [0u64; COORD_LIMBS];
    let mut carry = 0u64;

    for i in 0..COORD_LIMBS {
        let (sum1, c1) = a[i].overflowing_add(b[i]);
        let (sum2, c2) = sum1.overflowing_add(carry);
        result[i] = sum2;
        carry = (c1 as u64) + (c2 as u64);
    }

    (result, carry)
}

fn bigint_sub(a: &[u64; COORD_LIMBS], b: &[u64; COORD_LIMBS]) -> ([u64; COORD_LIMBS], u64) {
    let mut result = [0u64; COORD_LIMBS];
    let mut borrow = 0u64;

    for i in 0..COORD_LIMBS {
        let (diff1, b1) = a[i].overflowing_sub(b[i]);
        let (diff2, b2) = diff1.overflowing_sub(borrow);
        result[i] = diff2;
        borrow = (b1 as u64) + (b2 as u64);
    }

    (result, borrow)
}

fn field_add(a: &[u64; COORD_LIMBS], b: &[u64; COORD_LIMBS]) -> [u64; COORD_LIMBS] {
    let (sum, carry) = bigint_add(a, b);
    let (reduced, borrow) = bigint_sub(&sum, &BLS12_381_P);

    if carry != 0 || borrow == 0 {
        reduced
    } else {
        sum
    }
}

fn field_sub(a: &[u64; COORD_LIMBS], b: &[u64; COORD_LIMBS]) -> [u64; COORD_LIMBS] {
    let (diff, borrow) = bigint_sub(a, b);

    if borrow != 0 {
        let (result, _) = bigint_add(&diff, &BLS12_381_P);
        result
    } else {
        diff
    }
}

fn field_double(a: &[u64; COORD_LIMBS]) -> [u64; COORD_LIMBS] {
    field_add(a, a)
}

fn mul_wide(a: u64, b: u64) -> (u64, u64) {
    let full = (a as u128) * (b as u128);
    (full as u64, (full >> 64) as u64)
}

fn mont_reduce(t: &[u64; COORD_LIMBS * 2]) -> [u64; COORD_LIMBS] {
    let mut tmp = *t;

    for i in 0..COORD_LIMBS {
        let m = tmp[i].wrapping_mul(BLS12_381_INV);

        let mut carry = 0u64;
        for j in 0..COORD_LIMBS {
            let (lo, hi) = mul_wide(m, BLS12_381_P[j]);
            let (sum1, c1) = tmp[i + j].overflowing_add(lo);
            let (sum2, c2) = sum1.overflowing_add(carry);
            tmp[i + j] = sum2;
            carry = hi + (c1 as u64) + (c2 as u64);
        }

        for j in COORD_LIMBS..(COORD_LIMBS * 2 - i) {
            let (sum, c) = tmp[i + j].overflowing_add(carry);
            tmp[i + j] = sum;
            carry = c as u64;
            if carry == 0 {
                break;
            }
        }
    }

    let mut result = [0u64; COORD_LIMBS];
    result.copy_from_slice(&tmp[COORD_LIMBS..COORD_LIMBS * 2]);

    let (reduced, borrow) = bigint_sub(&result, &BLS12_381_P);
    if borrow == 0 {
        reduced
    } else {
        result
    }
}

fn mont_mul(a: &[u64; COORD_LIMBS], b: &[u64; COORD_LIMBS]) -> [u64; COORD_LIMBS] {
    let mut product = [0u64; COORD_LIMBS * 2];

    for i in 0..COORD_LIMBS {
        let mut carry = 0u64;
        for j in 0..COORD_LIMBS {
            let (lo, hi) = mul_wide(a[i], b[j]);
            let (sum1, c1) = product[i + j].overflowing_add(lo);
            let (sum2, c2) = sum1.overflowing_add(carry);
            product[i + j] = sum2;
            carry = hi + (c1 as u64) + (c2 as u64);
        }
        product[i + COORD_LIMBS] = carry;
    }

    mont_reduce(&product)
}

fn mont_square(a: &[u64; COORD_LIMBS]) -> [u64; COORD_LIMBS] {
    mont_mul(a, a)
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
            point_coord_limbs: 6,
        };

        assert_eq!(config.num_windows(), 32);
        assert_eq!(config.num_buckets(), 128);
    }

    #[test]
    fn test_exclusive_scan_counts_all_zero() {
        let (offsets, total) = exclusive_scan_counts(&[0, 0, 0]).expect("scan failed");
        assert_eq!(offsets, vec![0, 0, 0, 0]);
        assert_eq!(total, 0);
    }

    #[test]
    fn test_exclusive_scan_counts_single_nonzero() {
        let (offsets, total) = exclusive_scan_counts(&[0, 3, 0]).expect("scan failed");
        assert_eq!(offsets, vec![0, 0, 3, 3]);
        assert_eq!(total, 3);
    }

    #[test]
    fn test_exclusive_scan_counts_multiple_buckets() {
        let (offsets, total) = exclusive_scan_counts(&[2, 0, 5, 1]).expect("scan failed");
        assert_eq!(offsets, vec![0, 2, 2, 7, 8]);
        assert_eq!(total, 8);
    }

    #[test]
    fn test_exclusive_scan_counts_overflow() {
        assert!(exclusive_scan_counts(&[u32::MAX, 1]).is_err());
    }

    #[test]
    fn test_scalar_recoding_simple() {
        let config = MSMConfig {
            window_size: 4,
            num_limbs: 4,
            bits_per_limb: 64,
            point_coord_limbs: 6,
        };
        let msm = CudaMSM::new(config).expect("CUDA device required");

        // Scalar = 1
        let scalars = vec![1u64, 0, 0, 0];
        let digits = msm.recode_scalars_signed(&scalars, 1);
        assert_eq!(digits[0], 1);
        assert!(digits[1..].iter().all(|&d| d == 0));
    }

    #[test]
    fn test_scalar_recoding_zero() {
        let config = MSMConfig {
            window_size: 4,
            num_limbs: 4,
            bits_per_limb: 64,
            point_coord_limbs: 6,
        };
        let msm = CudaMSM::new(config).expect("CUDA device required");

        let scalars = vec![0u64; 4];
        let digits = msm.recode_scalars_signed(&scalars, 1);
        assert!(digits.iter().all(|&d| d == 0));
    }

    #[test]
    fn test_cpu_jacobian_arithmetic() {
        let id = JacobianPoint::identity();
        assert!(id.is_identity());

        let doubled = id.double();
        assert!(doubled.is_identity());

        let p = JacobianPoint {
            x: [0x1234, 0x5678, 0x9abc, 0xdef0, 0x1111, 0x2222],
            y: [0xfedc, 0xba98, 0x7654, 0x3210, 0x3333, 0x4444],
            z: [1, 0, 0, 0, 0, 0],
        };

        // P + identity = P
        let result = p.add(&id);
        assert_eq!(result.x, p.x);
        assert_eq!(result.y, p.y);
        assert_eq!(result.z, p.z);

        // identity + P = P
        let result = id.add(&p);
        assert_eq!(result.x, p.x);
        assert_eq!(result.y, p.y);
        assert_eq!(result.z, p.z);
    }

    #[test]
    fn test_field_add_basic() {
        let a = [1u64, 2, 3, 4, 5, 6];
        let zero = [0u64; COORD_LIMBS];
        assert_eq!(field_add(&a, &zero), a);
    }

    #[test]
    fn test_field_sub_basic() {
        let a = [1u64, 2, 3, 4, 5, 6];
        let result = field_sub(&a, &a);
        assert_eq!(result, [0u64; COORD_LIMBS]);
    }

    #[test]
    fn test_mont_mul_zero() {
        let a = [1u64, 2, 3, 4, 5, 6];
        let zero = [0u64; COORD_LIMBS];
        let result = mont_mul(&a, &zero);
        assert_eq!(result, zero);
    }

    #[test]
    fn test_mont_square_one() {
        // Montgomery form of 1 is R mod p
        let one_mont: [u64; COORD_LIMBS] = [
            0x760900000002fffd,
            0xebf4000bc40c0002,
            0x5f48985753c758ba,
            0x77ce585370525745,
            0x5c071a97a256ec6d,
            0x15f65ec3fa80e493,
        ];
        let result = mont_square(&one_mont);
        assert_eq!(result, one_mont);
    }

    // GPU integration tests require a CUDA device

    #[test]
    fn test_cuda_msm_identity() {
        let msm = CudaMSM::new_bls12_381().expect("CUDA device required");

        // Zero scalar -> identity point
        // Use scalar=0 but we need at least some nonzero structure
        // Actually zero scalar returns early in the kernel (digit==0 for all windows)
        // So all buckets stay at identity, reduction gives identity, combination gives identity
        let scalars = vec![0u64; 4];

        // Need a valid point - use BLS12-381 generator in Montgomery LE form
        // G.x in Montgomery LE
        let gx: [u64; 6] = [
            0x5cb38790fd530c16,
            0x7817fc679976fff5,
            0x154f95c7143ba1c1,
            0xf0ae6acdf3d0e747,
            0xedce6ecc21dbf440,
            0x120177419e0bfb75,
        ];
        // G.y in Montgomery LE
        let gy: [u64; 6] = [
            0xbaac93d50ce72271,
            0x8c22631a7918fd8e,
            0xdd595f13570725ce,
            0x51ac582950405194,
            0x0e1c8c3fad0059c0,
            0x0bbc3efc5008a26a,
        ];
        // G.z = R mod p (Montgomery form of 1)
        let gz: [u64; 6] = [
            0x760900000002fffd,
            0xebf4000bc40c0002,
            0x5f48985753c758ba,
            0x77ce585370525745,
            0x5c071a97a256ec6d,
            0x15f65ec3fa80e493,
        ];

        let mut points = Vec::with_capacity(18);
        points.extend_from_slice(&gx);
        points.extend_from_slice(&gy);
        points.extend_from_slice(&gz);

        let result = msm.compute(&scalars, &points).expect("compute failed");

        // Result z-coordinate should be all zeros (identity)
        let z = &result[12..18];
        assert!(z.iter().all(|&v| v == 0), "Expected identity point");
    }

    #[test]
    fn test_cuda_msm_single_point() {
        let g = BLS12381Curve::generator();
        let scalar = UnsignedInteger::<4>::from_u64(7);

        // CPU reference
        let cpu_result =
            pippenger::msm(&[scalar], std::slice::from_ref(&g)).expect("CPU MSM failed");
        let cpu_affine = cpu_result.to_affine();

        // GPU
        let scalars_flat = scalar_to_gpu_limbs(&scalar);
        let points_flat = point_to_gpu_flat(&g);

        let msm = CudaMSM::new_bls12_381().expect("CUDA device required");
        let gpu_result_flat = msm
            .compute(&scalars_flat, &points_flat)
            .expect("CUDA MSM compute failed");

        let gpu_point = gpu_flat_to_point(&gpu_result_flat);
        let gpu_affine = gpu_point.to_affine();

        assert_eq!(cpu_affine.x(), gpu_affine.x(), "x mismatch: 7 * G");
        assert_eq!(cpu_affine.y(), gpu_affine.y(), "y mismatch: 7 * G");
    }

    #[test]
    fn test_cuda_msm_small() {
        // Single-point MSMs to avoid the race condition
        let g = BLS12381Curve::generator();

        for k in [1u64, 2, 7, 42, 255, 1337, 65536] {
            let scalar = UnsignedInteger::<4>::from_u64(k);

            let cpu_result =
                pippenger::msm(&[scalar], std::slice::from_ref(&g)).expect("CPU MSM failed");
            let cpu_affine = cpu_result.to_affine();

            let scalars_flat = scalar_to_gpu_limbs(&scalar);
            let points_flat = point_to_gpu_flat(&g);

            let msm = CudaMSM::new_bls12_381().expect("CUDA device required");
            let gpu_result_flat = msm
                .compute(&scalars_flat, &points_flat)
                .expect("CUDA MSM compute failed");

            let gpu_point = gpu_flat_to_point(&gpu_result_flat);
            let gpu_affine = gpu_point.to_affine();

            assert_eq!(cpu_affine.x(), gpu_affine.x(), "x mismatch: {} * G", k);
            assert_eq!(cpu_affine.y(), gpu_affine.y(), "y mismatch: {} * G", k);
        }
    }

    #[test]
    fn test_cuda_msm_large_scalar() {
        let g = BLS12381Curve::generator();
        let scalar = UnsignedInteger::<4>::from_limbs([
            0x0000000000000001,
            0xFFFFFFFFFFFFFFFF,
            0x123456789ABCDEF0,
            0xFEDCBA9876543210,
        ]);

        let cpu_result =
            pippenger::msm(&[scalar], std::slice::from_ref(&g)).expect("CPU MSM failed");
        let cpu_affine = cpu_result.to_affine();

        let scalars_flat = scalar_to_gpu_limbs(&scalar);
        let points_flat = point_to_gpu_flat(&g);

        let msm = CudaMSM::new_bls12_381().expect("CUDA device required");
        let gpu_result_flat = msm
            .compute(&scalars_flat, &points_flat)
            .expect("CUDA MSM compute failed");

        let gpu_point = gpu_flat_to_point(&gpu_result_flat);
        let gpu_affine = gpu_point.to_affine();

        assert_eq!(cpu_affine.x(), gpu_affine.x(), "x mismatch");
        assert_eq!(cpu_affine.y(), gpu_affine.y(), "y mismatch");
    }

    #[test]
    fn test_cuda_msm_multi_point_matches_cpu() {
        let g = BLS12381Curve::generator();
        let scalars: Vec<_> = [1u64, 2, 7, 42, 255, 1337, 65536, 1 << 20]
            .into_iter()
            .map(UnsignedInteger::<4>::from_u64)
            .collect();
        let points: Vec<_> = (1u64..=scalars.len() as u64)
            .map(|power| g.operate_with_self(power))
            .collect();

        let cpu_result = pippenger::msm(&scalars, &points).expect("CPU MSM failed");
        let cpu_affine = cpu_result.to_affine();

        let scalars_flat: Vec<_> = scalars.iter().flat_map(scalar_to_gpu_limbs).collect();
        let points_flat: Vec<_> = points.iter().flat_map(point_to_gpu_flat).collect();

        let msm = CudaMSM::new_bls12_381().expect("CUDA device required");
        let gpu_result_flat = msm
            .compute(&scalars_flat, &points_flat)
            .expect("CUDA MSM compute failed");

        let gpu_point = gpu_flat_to_point(&gpu_result_flat);
        let gpu_affine = gpu_point.to_affine();

        assert_eq!(cpu_affine.x(), gpu_affine.x(), "x mismatch");
        assert_eq!(cpu_affine.y(), gpu_affine.y(), "y mismatch");
    }

    #[test]
    fn test_cuda_msm_bucket_collision_matches_cpu() {
        let g = BLS12381Curve::generator();
        let scalars: Vec<_> = [1u64; 8]
            .into_iter()
            .map(UnsignedInteger::<4>::from_u64)
            .collect();
        let points: Vec<_> = (1u64..=scalars.len() as u64)
            .map(|power| g.operate_with_self(power))
            .collect();

        let cpu_result = pippenger::msm(&scalars, &points).expect("CPU MSM failed");
        let cpu_affine = cpu_result.to_affine();

        let scalars_flat: Vec<_> = scalars.iter().flat_map(scalar_to_gpu_limbs).collect();
        let points_flat: Vec<_> = points.iter().flat_map(point_to_gpu_flat).collect();

        let msm = CudaMSM::new_bls12_381().expect("CUDA device required");
        let gpu_result_flat = msm
            .compute(&scalars_flat, &points_flat)
            .expect("CUDA MSM compute failed");

        let gpu_point = gpu_flat_to_point(&gpu_result_flat);
        let gpu_affine = gpu_point.to_affine();

        assert_eq!(cpu_affine.x(), gpu_affine.x(), "x mismatch");
        assert_eq!(cpu_affine.y(), gpu_affine.y(), "y mismatch");
    }

    #[test]
    fn test_cuda_msm_small_window_bucket_collision_matches_cpu() {
        assert_cuda_msm_matches_cpu_with_config(
            &[1u64; 8],
            MSMConfig {
                window_size: 4,
                num_limbs: 4,
                bits_per_limb: 64,
                point_coord_limbs: 6,
            },
        );
    }

    #[test]
    fn test_cuda_msm_negative_signed_digits_match_cpu() {
        assert_cuda_msm_matches_cpu_with_config(
            &[8u64, 9, 15, 24, 31, 40, 127, 255],
            MSMConfig {
                window_size: 4,
                num_limbs: 4,
                bits_per_limb: 64,
                point_coord_limbs: 6,
            },
        );
    }

    #[test]
    fn test_cuda_msm_zero_heavy_scalars_match_cpu() {
        assert_cuda_msm_matches_cpu_with_config(
            &[0u64, 0, 1, 0, 256, 0, 1 << 20, 0],
            MSMConfig::bls12_381(),
        );
    }

    #[test]
    fn test_cuda_msm_medium_deterministic_input_matches_cpu() {
        let scalars: Vec<_> = (0..257)
            .map(|i| {
                (i as u64 + 1)
                    .wrapping_mul(0x9e37_79b9_7f4a_7c15)
                    .rotate_left((i % 63) as u32)
            })
            .collect();
        assert_cuda_msm_matches_cpu_with_config(&scalars, MSMConfig::bls12_381());
    }

    // =========================================================================
    // Helpers for converting between lambdaworks types and GPU limb format
    // =========================================================================

    use crate::{
        cyclic_group::IsGroup,
        elliptic_curve::{
            short_weierstrass::{
                curves::bls12_381::{curve::BLS12381Curve, field_extension::BLS12381PrimeField},
                point::ShortWeierstrassJacobianPoint,
            },
            traits::IsEllipticCurve,
        },
        field::element::FieldElement,
        msm::pippenger,
        unsigned_integer::element::UnsignedInteger,
    };

    fn assert_cuda_msm_matches_cpu_with_config(scalar_values: &[u64], config: MSMConfig) {
        let g = BLS12381Curve::generator();
        let scalars: Vec<_> = scalar_values
            .iter()
            .copied()
            .map(UnsignedInteger::<4>::from_u64)
            .collect();
        let points: Vec<_> = (1u64..=scalars.len() as u64)
            .map(|power| g.operate_with_self(power))
            .collect();

        let cpu_result = pippenger::msm(&scalars, &points).expect("CPU MSM failed");
        let cpu_affine = cpu_result.to_affine();

        let scalars_flat: Vec<_> = scalars.iter().flat_map(scalar_to_gpu_limbs).collect();
        let points_flat: Vec<_> = points.iter().flat_map(point_to_gpu_flat).collect();

        let msm = CudaMSM::new(config).expect("CUDA device required");
        let gpu_result_flat = msm
            .compute(&scalars_flat, &points_flat)
            .expect("CUDA MSM compute failed");

        let gpu_point = gpu_flat_to_point(&gpu_result_flat);
        let gpu_affine = gpu_point.to_affine();

        assert_eq!(cpu_affine.x(), gpu_affine.x(), "x mismatch");
        assert_eq!(cpu_affine.y(), gpu_affine.y(), "y mismatch");
    }

    fn fe_to_gpu_limbs(fe: &FieldElement<BLS12381PrimeField>) -> [u64; 6] {
        let be_limbs = fe.value().limbs;
        let mut le_limbs = [0u64; 6];
        for i in 0..6 {
            le_limbs[i] = be_limbs[5 - i];
        }
        le_limbs
    }

    fn gpu_limbs_to_fe(le_limbs: &[u64]) -> FieldElement<BLS12381PrimeField> {
        let mut be_limbs = [0u64; 6];
        for i in 0..6 {
            be_limbs[i] = le_limbs[5 - i];
        }
        FieldElement::from_raw(UnsignedInteger::from_limbs(be_limbs))
    }

    fn point_to_gpu_flat(point: &ShortWeierstrassJacobianPoint<BLS12381Curve>) -> Vec<u64> {
        let [x, y, z] = point.coordinates();
        let mut flat = Vec::with_capacity(18);
        flat.extend_from_slice(&fe_to_gpu_limbs(x));
        flat.extend_from_slice(&fe_to_gpu_limbs(y));
        flat.extend_from_slice(&fe_to_gpu_limbs(z));
        flat
    }

    fn gpu_flat_to_point(limbs: &[u64]) -> ShortWeierstrassJacobianPoint<BLS12381Curve> {
        let x = gpu_limbs_to_fe(&limbs[0..6]);
        let y = gpu_limbs_to_fe(&limbs[6..12]);
        let z = gpu_limbs_to_fe(&limbs[12..18]);
        ShortWeierstrassJacobianPoint::new_unchecked([x, y, z])
    }

    fn scalar_to_gpu_limbs(scalar: &UnsignedInteger<4>) -> Vec<u64> {
        let be_limbs = scalar.limbs;
        let mut le_limbs = vec![0u64; 4];
        for i in 0..4 {
            le_limbs[i] = be_limbs[3 - i];
        }
        le_limbs
    }
}
