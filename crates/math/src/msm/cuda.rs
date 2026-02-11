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
//! - The bucket accumulation kernel has a race condition when multiple threads
//!   write to the same bucket. This is the same limitation as the Metal MSM.
//!   A sorting-based fix is planned as follow-up.
//! - Only BLS12-381 is supported. Constants are hardcoded.
//! - CPU-side MSM logic is duplicated from the Metal MSM module. A follow-up
//!   should extract shared code.

use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::safe::Ptx;
use lambdaworks_gpu::cuda::abstractions::errors::CudaError;
use std::sync::Arc;

const BLS12_381_MSM_PTX: &str = include_str!("../gpu/cuda/shaders/msm/bls12_381_msm.ptx");

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
                    "bucket_accumulation_bls12_381",
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

        // TODO: Remove this guard once bucket accumulation race condition is fixed
        // (sorting-based approach or per-thread local buckets with tree reduction)
        if num_scalars > 1 {
            return Err(CudaError::FunctionError(
                "Multi-point MSM not yet supported: bucket accumulation has a known \
                 race condition. Use single-point MSM or CPU pippenger for multiple points."
                    .to_string(),
            ));
        }

        // Step 1: Recode scalars to signed digits (CPU)
        let signed_digits = self.recode_scalars_signed(scalars, num_scalars);

        // Step 2: Create GPU buffers
        let num_buckets = self.config.num_buckets();
        let effective_windows = self.config.num_windows() + 1;

        let scalars_buf = self
            .device
            .htod_sync_copy(&signed_digits)
            .map_err(|err| CudaError::AllocateMemory(err.to_string()))?;

        let points_buf = self
            .device
            .htod_sync_copy(points)
            .map_err(|err| CudaError::AllocateMemory(err.to_string()))?;

        // Zero-initialized buckets (identity points have z=0)
        let total_bucket_elements = effective_windows * num_buckets * limbs_per_point;
        let bucket_data = vec![0u64; total_bucket_elements];
        let mut buckets_buf = self
            .device
            .htod_sync_copy(&bucket_data)
            .map_err(|err| CudaError::AllocateMemory(err.to_string()))?;

        // Config for accumulation: [num_scalars, num_windows, num_buckets, window_size]
        let config_data: [u32; 4] = [
            num_scalars as u32,
            effective_windows as u32,
            num_buckets as u32,
            self.config.window_size as u32,
        ];
        let config_buf = self
            .device
            .htod_sync_copy(&config_data)
            .map_err(|err| CudaError::AllocateMemory(err.to_string()))?;

        // Step 3: Run bucket accumulation kernel
        let accum_func = self
            .device
            .get_func("bls12_381_msm", "bucket_accumulation_bls12_381")
            .ok_or_else(|| CudaError::FunctionError("bucket_accumulation_bls12_381".to_string()))?;

        let total_threads = num_scalars * effective_windows;
        let block_size = 256u32;
        let grid_size = ((total_threads as u32) + block_size - 1) / block_size;

        let accum_config = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            accum_func.launch(
                accum_config,
                (&scalars_buf, &points_buf, &mut buckets_buf, &config_buf),
            )
        }
        .map_err(|err| CudaError::Launch(err.to_string()))?;

        // Step 4: Run bucket reduction kernel
        let window_sums_data = vec![0u64; effective_windows * limbs_per_point];
        let mut window_sums_buf = self
            .device
            .htod_sync_copy(&window_sums_data)
            .map_err(|err| CudaError::AllocateMemory(err.to_string()))?;

        // Reduction config: [num_windows, num_buckets]
        let reduction_config_data: [u32; 2] = [effective_windows as u32, num_buckets as u32];
        let reduction_config_buf = self
            .device
            .htod_sync_copy(&reduction_config_data)
            .map_err(|err| CudaError::AllocateMemory(err.to_string()))?;

        let reduce_func = self
            .device
            .get_func("bls12_381_msm", "bucket_reduction_bls12_381")
            .ok_or_else(|| CudaError::FunctionError("bucket_reduction_bls12_381".to_string()))?;

        let reduce_config = LaunchConfig {
            grid_dim: (effective_windows as u32, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            reduce_func.launch(
                reduce_config,
                (
                    &mut buckets_buf,
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
        use crate::{
            cyclic_group::IsGroup,
            elliptic_curve::{
                short_weierstrass::{
                    curves::bls12_381::{
                        curve::BLS12381Curve, field_extension::BLS12381PrimeField,
                    },
                    point::ShortWeierstrassJacobianPoint,
                },
                traits::IsEllipticCurve,
            },
            field::element::FieldElement,
            msm::pippenger,
            unsigned_integer::element::UnsignedInteger,
        };

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
        use crate::{
            cyclic_group::IsGroup,
            elliptic_curve::{
                short_weierstrass::{
                    curves::bls12_381::{
                        curve::BLS12381Curve, field_extension::BLS12381PrimeField,
                    },
                    point::ShortWeierstrassJacobianPoint,
                },
                traits::IsEllipticCurve,
            },
            field::element::FieldElement,
            msm::pippenger,
            unsigned_integer::element::UnsignedInteger,
        };

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
        use crate::{
            cyclic_group::IsGroup,
            elliptic_curve::{
                short_weierstrass::{
                    curves::bls12_381::{
                        curve::BLS12381Curve, field_extension::BLS12381PrimeField,
                    },
                    point::ShortWeierstrassJacobianPoint,
                },
                traits::IsEllipticCurve,
            },
            field::element::FieldElement,
            msm::pippenger,
            unsigned_integer::element::UnsignedInteger,
        };

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

    // =========================================================================
    // Helpers for converting between lambdaworks types and GPU limb format
    // =========================================================================

    use crate::{
        elliptic_curve::short_weierstrass::{
            curves::bls12_381::{curve::BLS12381Curve, field_extension::BLS12381PrimeField},
            point::ShortWeierstrassJacobianPoint,
        },
        field::element::FieldElement,
        unsigned_integer::element::UnsignedInteger,
    };

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
