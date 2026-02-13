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
///
/// Currently only BLS12-381 is supported. Both the Metal shader and
/// CPU-side arithmetic hardcode BLS12-381 field constants (modulus, R²,
/// Montgomery inverse).
#[derive(Debug, Clone)]
pub struct MSMConfig {
    /// Window size in bits for Pippenger's algorithm.
    /// Larger windows = fewer windows but more buckets.
    pub window_size: usize,
    /// Number of limbs in the scalar representation (e.g., 4 for 256-bit Fr).
    pub num_limbs: usize,
    /// Bits per limb (typically 64).
    pub bits_per_limb: usize,
    /// Number of limbs per point coordinate (e.g., 6 for BLS12-381 Fq).
    pub point_coord_limbs: usize,
}

impl MSMConfig {
    /// Creates a new MSM configuration for BLS12-381 (256-bit scalars, 381-bit base field).
    pub fn bls12_381() -> Self {
        Self {
            window_size: 16,
            num_limbs: 4,
            bits_per_limb: 64,
            point_coord_limbs: 6,
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

    /// Maximum supported window size (limited by i32 signed digit representation).
    pub const MAX_WINDOW_SIZE: usize = 30;

    /// Validates that `window_size` is in the allowed range.
    fn validate_window_size(&self) {
        assert!(
            self.window_size > 0 && self.window_size <= Self::MAX_WINDOW_SIZE,
            "window_size must be in 1..={}",
            Self::MAX_WINDOW_SIZE
        );
    }

    /// Returns the number of windows needed.
    ///
    /// # Panics
    /// Panics if `window_size` is 0 or exceeds `MAX_WINDOW_SIZE`.
    pub fn num_windows(&self) -> usize {
        self.validate_window_size();
        let total_bits = self.num_limbs * self.bits_per_limb;
        total_bits.div_ceil(self.window_size)
    }

    /// Returns the number of buckets per window (for signed representation).
    ///
    /// # Panics
    /// Panics if `window_size` is 0 or exceeds `MAX_WINDOW_SIZE`.
    pub fn num_buckets(&self) -> usize {
        self.validate_window_size();
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
        let coords_per_point = 3; // Jacobian: x, y, z
        let limbs_per_coord = self.config.point_coord_limbs;
        let limbs_per_point = coords_per_point * limbs_per_coord;

        // CPU-side JacobianPoint arithmetic hardcodes COORD_LIMBS for BLS12-381.
        if limbs_per_coord != COORD_LIMBS {
            return Err(MetalError::InvalidInputSize {
                expected: COORD_LIMBS,
                actual: limbs_per_coord,
            });
        }

        if !scalars.len().is_multiple_of(num_limbs) {
            return Err(MetalError::InvalidInputSize {
                expected: num_limbs,
                actual: scalars.len(),
            });
        }
        if !points.len().is_multiple_of(limbs_per_point) {
            return Err(MetalError::InvalidInputSize {
                expected: limbs_per_point,
                actual: points.len(),
            });
        }

        let num_scalars = scalars.len() / num_limbs;
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

        // Create bucket buffers for each window.
        // Use effective_windows (num_windows + 1) to match the signed digit array,
        // which includes an extra carry window for correctness with all scalar values.
        let num_buckets = self.config.num_buckets();
        let effective_windows = self.config.num_windows() + 1;
        let bucket_size = limbs_per_point; // Each bucket holds one Jacobian point

        // Initialize buckets to identity points (all zeros, since identity has z=0)
        // This is critical for correct accumulation.
        let total_bucket_elements = effective_windows * num_buckets * bucket_size;
        let bucket_data = vec![0u64; total_bucket_elements];
        let buckets_buffer = self.state.alloc_buffer_with_data(&bucket_data)?;

        // WARNING: The current bucket_accumulation kernel has a race condition.
        // Multiple threads may write to the same bucket without synchronization.
        // For production use, implement either:
        // 1. Sorting-based approach (cuZK paper) - sort (bucket_idx, point) pairs
        // 2. Atomic operations (complex for 256-bit point addition)
        // 3. Per-thread local buckets with reduction
        // For now, this implementation may produce incorrect results when
        // multiple points map to the same bucket.

        // Step 3: Run bucket accumulation kernel
        self.run_bucket_accumulation(
            &scalars_buffer,
            &points_buffer,
            &buckets_buffer,
            num_scalars,
            effective_windows,
            num_buckets,
        )?;

        // Step 4: Run bucket reduction kernel
        let window_sums_buffer = self
            .state
            .alloc_buffer(effective_windows * bucket_size * std::mem::size_of::<u64>())?;

        self.run_bucket_reduction(
            &buckets_buffer,
            &window_sums_buffer,
            effective_windows,
            num_buckets,
        )?;

        // Step 5: Read window sums back to CPU
        let window_sums: Vec<u64> = unsafe {
            self.state
                .read_buffer(&window_sums_buffer, effective_windows * bucket_size)
        };

        // Step 6: Combine windows using Horner's method (CPU)
        // This is sequential and better done on CPU
        let result = self.combine_windows(&window_sums, effective_windows, bucket_size);

        Ok(result)
    }

    /// Recodes scalars to signed digit representation.
    /// Uses NAF-like recoding to halve the bucket count.
    ///
    /// Returns `num_scalars * effective_windows()` digits, where
    /// `effective_windows() = num_windows() + 1` to capture carry overflow.
    fn recode_scalars_signed(&self, scalars: &[u64], num_scalars: usize) -> Vec<i32> {
        let window_size = self.config.window_size;
        let num_windows = self.config.num_windows();
        let num_limbs = self.config.num_limbs;
        let half_bucket = 1i32 << (window_size - 1);
        let full_bucket = 1i32 << window_size;
        let mask = (1u64 << window_size) - 1;

        // +1 extra window to capture carry overflow from signed digit recoding.
        // Without this, scalars with large top bits produce incorrect encodings.
        let effective_windows = num_windows + 1;
        let mut digits = vec![0i32; num_scalars * effective_windows];

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

                digits[scalar_idx * effective_windows + window_idx] = digit;
            }

            // Store final carry in the extra window to handle scalars with large top bits.
            digits[scalar_idx * effective_windows + num_windows] = carry;
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
            &[
                scalars_buffer,
                points_buffer,
                buckets_buffer,
                &config_buffer,
            ],
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
        if num_windows == 0 || window_sums.is_empty() {
            return vec![0u64; point_size];
        }

        // Horner's method: start from highest window and work down
        // result = ((w[n-1] * 2^c + w[n-2]) * 2^c + w[n-3]) * 2^c + ... + w[0]
        let window_size = self.config.window_size;

        // Start with the highest window (index num_windows - 1)
        let base = (num_windows - 1) * point_size;
        let mut result = JacobianPoint::from_limbs(&window_sums[base..base + point_size]);

        // Process remaining windows in reverse order (from num_windows-2 down to 0)
        for window_idx in (0..num_windows - 1).rev() {
            // Double result `window_size` times
            for _ in 0..window_size {
                result = result.double();
            }
            // Add the next window sum
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
/// Uses the same Montgomery representation as the GPU.
#[derive(Clone, Debug)]
struct JacobianPoint {
    x: [u64; COORD_LIMBS],
    y: [u64; COORD_LIMBS],
    z: [u64; COORD_LIMBS],
}

/// BLS12-381 prime field modulus (little-endian limbs)
const BLS12_381_P: [u64; COORD_LIMBS] = [
    0xb9feffffffffaaab, // limb 0 (LSB)
    0x1eabfffeb153ffff,
    0x6730d2a0f6b0f624,
    0x64774b84f38512bf,
    0x4b1ba7b6434bacd7,
    0x1a0111ea397fe69a, // limb 5 (MSB)
];

/// Montgomery parameter: -p^(-1) mod 2^64
const BLS12_381_INV: u64 = 0x89f3fffcfffcfffd;

impl JacobianPoint {
    /// Creates the identity point (point at infinity).
    #[allow(dead_code)] // Used in tests
    fn identity() -> Self {
        Self {
            x: [0; COORD_LIMBS],
            y: [0; COORD_LIMBS],
            z: [0; COORD_LIMBS],
        }
    }

    /// Creates a point from a flat array of u64 limbs.
    ///
    /// # Panics
    /// Panics if `limbs.len() < COORD_LIMBS * 3`.
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

    /// Converts the point to a flat array of u64 limbs.
    fn to_limbs(&self) -> Vec<u64> {
        let mut result = Vec::with_capacity(COORD_LIMBS * 3);
        result.extend_from_slice(&self.x);
        result.extend_from_slice(&self.y);
        result.extend_from_slice(&self.z);
        result
    }

    /// Checks if this is the identity point (z == 0).
    fn is_identity(&self) -> bool {
        self.z.iter().all(|&limb| limb == 0)
    }

    /// Point doubling using 2009-l formula from EFD.
    fn double(&self) -> Self {
        if self.is_identity() {
            return self.clone();
        }

        // A = X1^2
        let a = mont_square(&self.x);
        // B = Y1^2
        let b = mont_square(&self.y);
        // C = B^2
        let c = mont_square(&b);

        // D = 2*((X1+B)^2-A-C)
        let tmp = field_add(&self.x, &b);
        let tmp = mont_square(&tmp);
        let tmp = field_sub(&tmp, &a);
        let tmp = field_sub(&tmp, &c);
        let d = field_double(&tmp);

        // E = 3*A
        let e = field_add(&a, &field_double(&a));

        // F = E^2
        let f = mont_square(&e);

        // X3 = F-2*D
        let x3 = field_sub(&f, &field_double(&d));

        // Y3 = E*(D-X3)-8*C
        let y3 = field_sub(&d, &x3);
        let y3 = mont_mul(&e, &y3);
        let c8 = field_double(&field_double(&field_double(&c)));
        let y3 = field_sub(&y3, &c8);

        // Z3 = 2*Y1*Z1
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

        // Z1Z1 = Z1^2
        let z1z1 = mont_square(&self.z);
        // Z2Z2 = Z2^2
        let z2z2 = mont_square(&other.z);

        // U1 = X1*Z2Z2
        let u1 = mont_mul(&self.x, &z2z2);
        // U2 = X2*Z1Z1
        let u2 = mont_mul(&other.x, &z1z1);

        // S1 = Y1*Z2*Z2Z2
        let s1 = mont_mul(&self.y, &other.z);
        let s1 = mont_mul(&s1, &z2z2);
        // S2 = Y2*Z1*Z1Z1
        let s2 = mont_mul(&other.y, &self.z);
        let s2 = mont_mul(&s2, &z1z1);

        // H = U2-U1
        let h = field_sub(&u2, &u1);

        // Handle P == Q case: when H == 0, the addition formula degenerates.
        let zero = [0u64; COORD_LIMBS];
        if h == zero {
            let s_diff = field_sub(&s2, &s1);
            if s_diff == zero {
                return self.double();
            } else {
                return Self::identity();
            }
        }

        // I = (2*H)^2
        let i = field_double(&h);
        let i = mont_square(&i);
        // J = H*I
        let j = mont_mul(&h, &i);

        // r = 2*(S2-S1)
        let r = field_sub(&s2, &s1);
        let r = field_double(&r);

        // V = U1*I
        let v = mont_mul(&u1, &i);

        // X3 = r^2-J-2*V
        let x3 = mont_square(&r);
        let x3 = field_sub(&x3, &j);
        let x3 = field_sub(&x3, &field_double(&v));

        // Y3 = r*(V-X3)-2*S1*J
        let y3 = field_sub(&v, &x3);
        let y3 = mont_mul(&r, &y3);
        let tmp = mont_mul(&s1, &j);
        let tmp = field_double(&tmp);
        let y3 = field_sub(&y3, &tmp);

        // Z3 = ((Z1+Z2)^2-Z1Z1-Z2Z2)*H
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

/// Add two big integers, returns (result, carry).
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

/// Subtract two big integers, returns (result, borrow).
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

/// Field addition: (a + b) mod p
fn field_add(a: &[u64; COORD_LIMBS], b: &[u64; COORD_LIMBS]) -> [u64; COORD_LIMBS] {
    let (sum, carry) = bigint_add(a, b);
    let (reduced, borrow) = bigint_sub(&sum, &BLS12_381_P);

    if carry != 0 || borrow == 0 {
        reduced
    } else {
        sum
    }
}

/// Field subtraction: (a - b) mod p
fn field_sub(a: &[u64; COORD_LIMBS], b: &[u64; COORD_LIMBS]) -> [u64; COORD_LIMBS] {
    let (diff, borrow) = bigint_sub(a, b);

    if borrow != 0 {
        let (result, _) = bigint_add(&diff, &BLS12_381_P);
        result
    } else {
        diff
    }
}

/// Field doubling: 2a mod p
fn field_double(a: &[u64; COORD_LIMBS]) -> [u64; COORD_LIMBS] {
    field_add(a, a)
}

/// Wide multiplication of two u64 values, returns (lo, hi).
fn mul_wide(a: u64, b: u64) -> (u64, u64) {
    let full = (a as u128) * (b as u128);
    (full as u64, (full >> 64) as u64)
}

/// Montgomery reduction: T * R^(-1) mod p
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

        // Propagate carry
        for j in COORD_LIMBS..(COORD_LIMBS * 2 - i) {
            let (sum, c) = tmp[i + j].overflowing_add(carry);
            tmp[i + j] = sum;
            carry = c as u64;
            if carry == 0 {
                break;
            }
        }
    }

    // Result is in upper half
    let mut result = [0u64; COORD_LIMBS];
    result.copy_from_slice(&tmp[COORD_LIMBS..COORD_LIMBS * 2]);

    // Final reduction if result >= p
    let (reduced, borrow) = bigint_sub(&result, &BLS12_381_P);
    if borrow == 0 {
        reduced
    } else {
        result
    }
}

/// Montgomery multiplication: a * b * R^(-1) mod p
fn mont_mul(a: &[u64; COORD_LIMBS], b: &[u64; COORD_LIMBS]) -> [u64; COORD_LIMBS] {
    let mut product = [0u64; COORD_LIMBS * 2];

    // Multiply a * b
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

/// Montgomery squaring
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

    // =============================================================================
    // CPU-side Field Arithmetic Tests
    // =============================================================================

    #[test]
    fn test_field_add_basic() {
        // Test a + 0 = a
        let a = [1u64, 2, 3, 4, 5, 6];
        let zero = [0u64; COORD_LIMBS];
        assert_eq!(field_add(&a, &zero), a);
    }

    #[test]
    fn test_field_sub_basic() {
        // Test a - a = 0
        let a = [1u64, 2, 3, 4, 5, 6];
        let result = field_sub(&a, &a);
        assert_eq!(result, [0u64; COORD_LIMBS]);
    }

    #[test]
    fn test_field_double_basic() {
        // Test 2 * 0 = 0
        let zero = [0u64; COORD_LIMBS];
        assert_eq!(field_double(&zero), zero);
    }

    #[test]
    fn test_jacobian_identity() {
        let id = JacobianPoint::identity();
        assert!(id.is_identity());
    }

    #[test]
    fn test_jacobian_double_identity() {
        let id = JacobianPoint::identity();
        let doubled = id.double();
        assert!(doubled.is_identity());
    }

    #[test]
    fn test_jacobian_add_identity() {
        let id = JacobianPoint::identity();
        let p = JacobianPoint {
            x: [1, 2, 3, 4, 5, 6],
            y: [7, 8, 9, 10, 11, 12],
            z: [1, 0, 0, 0, 0, 0], // Non-identity (z != 0)
        };

        // id + p = p
        let result = id.add(&p);
        assert_eq!(result.x, p.x);
        assert_eq!(result.y, p.y);
        assert_eq!(result.z, p.z);

        // p + id = p
        let result = p.add(&id);
        assert_eq!(result.x, p.x);
        assert_eq!(result.y, p.y);
        assert_eq!(result.z, p.z);
    }

    #[test]
    fn test_bigint_add_no_overflow() {
        let a = [1u64, 2, 3, 4, 5, 6];
        let b = [5u64, 6, 7, 8, 9, 10];
        let (result, carry) = bigint_add(&a, &b);
        assert_eq!(result, [6, 8, 10, 12, 14, 16]);
        assert_eq!(carry, 0);
    }

    #[test]
    fn test_bigint_sub_no_borrow() {
        let a = [5u64, 6, 7, 8, 9, 10];
        let b = [1u64, 2, 3, 4, 5, 6];
        let (result, borrow) = bigint_sub(&a, &b);
        assert_eq!(result, [4, 4, 4, 4, 4, 4]);
        assert_eq!(borrow, 0);
    }

    #[test]
    fn test_mont_mul_zero() {
        // a * 0 = 0 (in Montgomery form)
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
        // mont_square(R mod p) should equal R mod p (since 1*1 = 1 in the field)
        let result = mont_square(&one_mont);
        assert_eq!(result, one_mont, "mont_square(1_mont) should be 1_mont");
    }

    #[test]
    fn test_mont_square_gx() {
        // G.x in Montgomery LE form
        let gx_mont: [u64; COORD_LIMBS] = [
            0x5cb38790fd530c16,
            0x7817fc679976fff5,
            0x154f95c7143ba1c1,
            0xf0ae6acdf3d0e747,
            0xedce6ecc21dbf440,
            0x120177419e0bfb75,
        ];
        // Expected result from Python: mont_square(gx) = gx^2 * R^(-1) mod p
        let expected: [u64; COORD_LIMBS] = [
            0x9e5c25e1f840429e,
            0x0bb5e06755c1bb91,
            0x34b02a9a934e43b1,
            0x13b6742f7c29eca5,
            0x53e41a48a899ccd5,
            0x0a55331f9bb57ced,
        ];
        let result = mont_square(&gx_mont);
        assert_eq!(
            result, expected,
            "mont_square(gx) mismatch\ngot:      {:x?}\nexpected: {:x?}",
            result, expected
        );
    }

    #[test]
    fn test_mul_wide() {
        let (lo, hi) = mul_wide(u64::MAX, u64::MAX);
        // u64::MAX * u64::MAX = 2^128 - 2*2^64 + 1
        // = (2^64 - 2) * 2^64 + 1
        // hi = 2^64 - 2, lo = 1
        assert_eq!(lo, 1);
        assert_eq!(hi, u64::MAX - 1);
    }

    // =============================================================================
    // Scalar Recoding Tests
    // =============================================================================

    #[test]
    fn test_scalar_recoding_zero() {
        let config = MSMConfig {
            window_size: 4,
            num_limbs: 4,
            bits_per_limb: 64,
            point_coord_limbs: 6,
        };
        let msm = MetalMSM::new(config).expect("Metal device required");

        // Scalar = 0
        let scalars = vec![0u64; 4];
        let digits = msm.recode_scalars_signed(&scalars, 1);

        // All digits should be 0
        assert!(digits.iter().all(|&d| d == 0));
    }

    #[test]
    fn test_scalar_recoding_one() {
        let config = MSMConfig {
            window_size: 4,
            num_limbs: 4,
            bits_per_limb: 64,
            point_coord_limbs: 6,
        };
        let msm = MetalMSM::new(config).expect("Metal device required");

        // Scalar = 1 (in least significant limb)
        let scalars = vec![1u64, 0, 0, 0];
        let digits = msm.recode_scalars_signed(&scalars, 1);

        // First digit should be 1, rest should be 0
        assert_eq!(digits[0], 1);
        assert!(digits[1..].iter().all(|&d| d == 0));
    }

    // =============================================================================
    // Integration Tests (require Metal device)
    // =============================================================================

    #[test]
    fn test_metal_msm_initialization() {
        let mut msm = MetalMSM::new_bls12_381().expect("Metal device required");
        msm.initialize().expect("Shader compilation failed");
        assert!(msm.initialized);

        // Re-initialization should be a no-op
        msm.initialize().expect("Re-initialization failed");
    }
}

// =============================================================================
// Differential Fuzzing Tests (compare Metal vs CPU implementation)
// =============================================================================

#[cfg(test)]
mod fuzz_tests {
    use super::*;
    use proptest::prelude::*;

    use lambdaworks_math::{
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

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 10,
            ..ProptestConfig::default()
        })]

        /// Property test: MSM configuration is self-consistent
        #[test]
        fn prop_config_consistency(window_size in 2usize..=16) {
            let config = MSMConfig {
                window_size,
                num_limbs: 4,
                bits_per_limb: 64,
                point_coord_limbs: 6,
            };

            let total_bits = config.num_limbs * config.bits_per_limb;
            let expected_windows = (total_bits + window_size - 1) / window_size;
            let expected_buckets = 1 << (window_size - 1);

            prop_assert_eq!(config.num_windows(), expected_windows);
            prop_assert_eq!(config.num_buckets(), expected_buckets);
        }

        /// Property test: Scalar recoding produces valid signed digits
        #[test]
        fn prop_scalar_recoding_valid_digits(
            limbs in prop::array::uniform4(any::<u64>()),
            window_size in 2usize..=8
        ) {
            let config = MSMConfig {
                window_size,
                num_limbs: 4,
                bits_per_limb: 64,
                point_coord_limbs: 6,
            };
            let msm = MetalMSM::new(config).expect("Metal device required");

            let scalars: Vec<u64> = limbs.to_vec();
            let digits = msm.recode_scalars_signed(&scalars, 1);

            let half_bucket = 1i32 << (window_size - 1);

            // All digits should be in range [-half_bucket, half_bucket)
            for digit in digits {
                prop_assert!(digit >= -half_bucket && digit < half_bucket,
                    "Digit {} out of range [-{}, {})", digit, half_bucket, half_bucket);
            }
        }

        /// Property test: Scalar recoding reconstructs original scalar value.
        /// Verifies that sum(digit_i * 2^(i*window_size)) equals the original scalar (mod 2^256).
        #[test]
        fn prop_scalar_recoding_reconstruction(
            limbs in prop::array::uniform4(any::<u64>()),
            window_size in 2usize..=8
        ) {
            let config = MSMConfig {
                window_size,
                num_limbs: 4,
                bits_per_limb: 64,
                point_coord_limbs: 6,
            };
            let num_windows = config.num_windows();
            let msm = MetalMSM::new(config).expect("Metal device required");

            let scalars: Vec<u64> = limbs.to_vec();
            let digits = msm.recode_scalars_signed(&scalars, 1);

            // Reconstruct scalar from signed digits using wrapping 256-bit arithmetic.
            // The carry window (index num_windows) contributes 2^(num_windows*c) which
            // is >= 2^256, so it vanishes mod 2^256 and we only need the first num_windows digits.
            let mut reconstructed = [0u64; 4];

            for (i, &digit) in digits.iter().enumerate().take(num_windows) {
                if digit == 0 { continue; }

                let bit_offset = i * window_size;
                let limb_idx = bit_offset / 64;
                let bit_in_limb = bit_offset % 64;

                if limb_idx >= 4 { continue; }

                let positive = digit > 0;
                let abs_val = digit.unsigned_abs() as u64;

                let lo = abs_val << bit_in_limb;
                let hi = if bit_in_limb > 0 {
                    abs_val >> (64 - bit_in_limb)
                } else {
                    0
                };

                if positive {
                    let (s, c) = reconstructed[limb_idx].overflowing_add(lo);
                    reconstructed[limb_idx] = s;
                    let mut carry = c as u64;

                    if limb_idx + 1 < 4 {
                        let (s, c1) = reconstructed[limb_idx + 1].overflowing_add(hi);
                        let (s, c2) = s.overflowing_add(carry);
                        reconstructed[limb_idx + 1] = s;
                        carry = (c1 as u64) + (c2 as u64);
                    }

                    for k in (limb_idx + 2)..4 {
                        if carry == 0 { break; }
                        let (s, c) = reconstructed[k].overflowing_add(carry);
                        reconstructed[k] = s;
                        carry = c as u64;
                    }
                } else {
                    let (s, b) = reconstructed[limb_idx].overflowing_sub(lo);
                    reconstructed[limb_idx] = s;
                    let mut borrow = b as u64;

                    if limb_idx + 1 < 4 {
                        let (s, b1) = reconstructed[limb_idx + 1].overflowing_sub(hi);
                        let (s, b2) = s.overflowing_sub(borrow);
                        reconstructed[limb_idx + 1] = s;
                        borrow = (b1 as u64) + (b2 as u64);
                    }

                    for k in (limb_idx + 2)..4 {
                        if borrow == 0 { break; }
                        let (s, b) = reconstructed[k].overflowing_sub(borrow);
                        reconstructed[k] = s;
                        borrow = b as u64;
                    }
                }
            }

            // Compare mod 2^256
            for i in 0..4 {
                prop_assert_eq!(
                    reconstructed[i], limbs[i],
                    "Reconstruction mismatch at limb {} for window_size {}",
                    i, window_size
                );
            }
        }
    }

    /// Test that the CPU-side Jacobian point arithmetic is self-consistent.
    #[test]
    fn test_jacobian_arithmetic_consistency() {
        let p = JacobianPoint {
            x: [0x1234, 0x5678, 0x9abc, 0xdef0, 0x1111, 0x2222],
            y: [0xfedc, 0xba98, 0x7654, 0x3210, 0x3333, 0x4444],
            z: [1, 0, 0, 0, 0, 0],
        };

        // P + identity = P
        let id = JacobianPoint::identity();
        let result = p.add(&id);
        assert_eq!(result.x, p.x);
        assert_eq!(result.y, p.y);
        assert_eq!(result.z, p.z);

        // identity + P = P
        let result = id.add(&p);
        assert_eq!(result.x, p.x);
        assert_eq!(result.y, p.y);
        assert_eq!(result.z, p.z);

        // identity doubled = identity
        let result = id.double();
        assert!(result.is_identity());
    }

    /// Test window combination with simple cases.
    #[test]
    fn test_combine_windows_single() {
        let config = MSMConfig {
            window_size: 4,
            num_limbs: 4,
            bits_per_limb: 64,
            point_coord_limbs: 6,
        };
        let msm = MetalMSM::new(config).expect("Metal device required");

        // Single window: result should be that window (18 limbs = 3 coords * 6 limbs)
        let point_size = COORD_LIMBS * 3;
        let window_sum: Vec<u64> = (1..=point_size as u64).collect();
        let result = msm.combine_windows(&window_sum, 1, point_size);
        assert_eq!(result, window_sum);
    }

    /// Test combine_windows with all-zero windows (identity points).
    #[test]
    fn test_combine_windows_identity() {
        let config = MSMConfig {
            window_size: 4,
            num_limbs: 4,
            bits_per_limb: 64,
            point_coord_limbs: 6,
        };
        let msm = MetalMSM::new(config).expect("Metal device required");

        let point_size = COORD_LIMBS * 3;
        let window_sums = vec![0u64; 2 * point_size];
        let result = msm.combine_windows(&window_sums, 2, point_size);

        // Result should be identity (z = 0)
        assert!(
            result[COORD_LIMBS * 2..COORD_LIMBS * 3]
                .iter()
                .all(|&x| x == 0),
            "Expected identity point"
        );
    }

    // =========================================================================
    // Differential Tests: GPU MSM vs CPU pippenger
    // =========================================================================

    /// Convert a FieldElement (big-endian Montgomery) to GPU little-endian limbs.
    fn fe_to_gpu_limbs(fe: &FieldElement<BLS12381PrimeField>) -> [u64; 6] {
        let be_limbs = fe.value().limbs; // big-endian: limbs[0] = MSB
        let mut le_limbs = [0u64; 6];
        for i in 0..6 {
            le_limbs[i] = be_limbs[5 - i];
        }
        le_limbs
    }

    /// Convert GPU little-endian limbs back to a FieldElement (big-endian Montgomery).
    fn gpu_limbs_to_fe(le_limbs: &[u64]) -> FieldElement<BLS12381PrimeField> {
        let mut be_limbs = [0u64; 6];
        for i in 0..6 {
            be_limbs[i] = le_limbs[5 - i];
        }
        FieldElement::from_raw(UnsignedInteger::from_limbs(be_limbs))
    }

    /// Convert a Jacobian point to flat GPU buffer (little-endian Montgomery limbs).
    fn point_to_gpu_flat(point: &ShortWeierstrassJacobianPoint<BLS12381Curve>) -> Vec<u64> {
        let [x, y, z] = point.coordinates();
        let mut flat = Vec::with_capacity(18);
        flat.extend_from_slice(&fe_to_gpu_limbs(x));
        flat.extend_from_slice(&fe_to_gpu_limbs(y));
        flat.extend_from_slice(&fe_to_gpu_limbs(z));
        flat
    }

    /// Convert GPU result limbs back to a Jacobian point.
    fn gpu_flat_to_point(limbs: &[u64]) -> ShortWeierstrassJacobianPoint<BLS12381Curve> {
        let x = gpu_limbs_to_fe(&limbs[0..6]);
        let y = gpu_limbs_to_fe(&limbs[6..12]);
        let z = gpu_limbs_to_fe(&limbs[12..18]);
        ShortWeierstrassJacobianPoint::new_unchecked([x, y, z])
    }

    /// Convert a scalar UnsignedInteger<4> to flat GPU buffer (little-endian limbs).
    fn scalar_to_gpu_limbs(scalar: &UnsignedInteger<4>) -> Vec<u64> {
        let be_limbs = scalar.limbs; // big-endian
        let mut le_limbs = vec![0u64; 4];
        for i in 0..4 {
            le_limbs[i] = be_limbs[3 - i];
        }
        le_limbs
    }

    /// Helper: run single-point Metal MSM and compare with CPU pippenger.
    fn assert_gpu_cpu_match(
        scalar: &UnsignedInteger<4>,
        point: &ShortWeierstrassJacobianPoint<BLS12381Curve>,
        label: &str,
    ) {
        // CPU reference
        let cpu_result =
            pippenger::msm(&[*scalar], std::slice::from_ref(point)).expect("CPU MSM failed");
        let cpu_affine = cpu_result.to_affine();

        // GPU
        let scalars_flat = scalar_to_gpu_limbs(scalar);
        let points_flat = point_to_gpu_flat(point);

        let mut msm = MetalMSM::new_bls12_381().expect("Metal device required");
        let gpu_result_flat = msm
            .compute(&scalars_flat, &points_flat)
            .expect("Metal MSM compute failed");

        let gpu_point = gpu_flat_to_point(&gpu_result_flat);
        let gpu_affine = gpu_point.to_affine();

        assert_eq!(cpu_affine.x(), gpu_affine.x(), "x mismatch: {}", label);
        assert_eq!(cpu_affine.y(), gpu_affine.y(), "y mismatch: {}", label);
    }

    /// Differential test: compute scalar * G via Metal MSM and compare with
    /// CPU pippenger. Uses a single point to avoid the race condition in
    /// bucket_accumulation.
    #[test]
    fn test_metal_msm_single_point() {
        let g = BLS12381Curve::generator();
        let scalar = UnsignedInteger::<4>::from_u64(7);
        assert_gpu_cpu_match(&scalar, &g, "7 * G");
    }

    /// Differential test: 1 * G should equal G.
    #[test]
    fn test_metal_msm_identity_scalar() {
        let g = BLS12381Curve::generator();
        let scalar = UnsignedInteger::<4>::from_u64(1);
        assert_gpu_cpu_match(&scalar, &g, "1 * G");
    }

    /// Differential test: 2 * G should equal G + G.
    #[test]
    fn test_metal_msm_generator_double() {
        let g = BLS12381Curve::generator();
        let scalar = UnsignedInteger::<4>::from_u64(2);
        assert_gpu_cpu_match(&scalar, &g, "2 * G");
    }

    /// Differential test: large scalar (> 64 bits) to exercise multi-limb recoding.
    #[test]
    fn test_metal_msm_large_scalar() {
        let g = BLS12381Curve::generator();
        // Scalar that spans multiple 64-bit limbs
        let scalar = UnsignedInteger::<4>::from_limbs([
            0x0000000000000001, // MSB (big-endian)
            0xFFFFFFFFFFFFFFFF,
            0x123456789ABCDEF0,
            0xFEDCBA9876543210, // LSB
        ]);
        assert_gpu_cpu_match(&scalar, &g, "large multi-limb scalar * G");
    }

    /// Differential test: scalar with all bits set (maximum scalar).
    #[test]
    fn test_metal_msm_max_scalar() {
        let g = BLS12381Curve::generator();
        let scalar = UnsignedInteger::<4>::from_limbs([u64::MAX, u64::MAX, u64::MAX, u64::MAX]);
        assert_gpu_cpu_match(&scalar, &g, "max scalar * G");
    }

    /// Differential test: power of two scalars to exercise window boundaries.
    #[test]
    fn test_metal_msm_power_of_two_scalars() {
        let g = BLS12381Curve::generator();
        let points_flat = point_to_gpu_flat(&g);
        let mut msm = MetalMSM::new_bls12_381().expect("Metal device required");
        msm.initialize().expect("Shader compilation failed");

        for shift in [1, 15, 16, 17, 63, 64, 65, 127, 128, 255] {
            let scalar = UnsignedInteger::<4>::from_u64(1) << shift;

            let cpu_result =
                pippenger::msm(&[scalar], std::slice::from_ref(&g)).expect("CPU MSM failed");
            let cpu_affine = cpu_result.to_affine();

            let scalars_flat = scalar_to_gpu_limbs(&scalar);
            let gpu_result_flat = msm
                .compute(&scalars_flat, &points_flat)
                .expect("Metal MSM compute failed");

            let gpu_point = gpu_flat_to_point(&gpu_result_flat);
            let gpu_affine = gpu_point.to_affine();

            assert_eq!(
                cpu_affine.x(),
                gpu_affine.x(),
                "x mismatch: 2^{} * G",
                shift
            );
            assert_eq!(
                cpu_affine.y(),
                gpu_affine.y(),
                "y mismatch: 2^{} * G",
                shift
            );
        }
    }

    /// Differential test: scalar * P where P = k*G (non-generator base point).
    #[test]
    fn test_metal_msm_non_generator_point() {
        let g = BLS12381Curve::generator();
        let p = g.operate_with_self(42u64); // P = 42*G
        let scalar = UnsignedInteger::<4>::from_u64(1337);
        assert_gpu_cpu_match(&scalar, &p, "1337 * (42*G)");
    }

    /// Differential test: CPU-side field arithmetic matches lambdaworks.
    /// Verifies that our CPU JacobianPoint.double() produces the same result
    /// as lambdaworks for the BLS12-381 generator.
    #[test]
    fn test_cpu_point_double_matches_lambdaworks() {
        let g = BLS12381Curve::generator();
        let lw_2g = g.operate_with(&g).to_affine();

        // Convert generator to our CPU-side representation
        let g_flat = point_to_gpu_flat(&g);
        let cpu_g = super::JacobianPoint::from_limbs(&g_flat);

        // Double using our CPU arithmetic
        let cpu_2g = cpu_g.double();
        let cpu_2g_flat = cpu_2g.to_limbs();
        let cpu_2g_point = gpu_flat_to_point(&cpu_2g_flat);
        let cpu_2g_affine = cpu_2g_point.to_affine();

        assert_eq!(lw_2g.x(), cpu_2g_affine.x(), "double x mismatch");
        assert_eq!(lw_2g.y(), cpu_2g_affine.y(), "double y mismatch");
    }

    /// Differential test: CPU-side field arithmetic add matches lambdaworks.
    #[test]
    fn test_cpu_point_add_matches_lambdaworks() {
        let g = BLS12381Curve::generator();
        let p2 = g.operate_with_self(5u64);
        let lw_sum = g.operate_with(&p2).to_affine();

        let g_flat = point_to_gpu_flat(&g);
        let p2_flat = point_to_gpu_flat(&p2);
        let cpu_g = super::JacobianPoint::from_limbs(&g_flat);
        let cpu_p2 = super::JacobianPoint::from_limbs(&p2_flat);

        let cpu_sum = cpu_g.add(&cpu_p2);
        let cpu_sum_flat = cpu_sum.to_limbs();
        let cpu_sum_point = gpu_flat_to_point(&cpu_sum_flat);
        let cpu_sum_affine = cpu_sum_point.to_affine();

        assert_eq!(lw_sum.x(), cpu_sum_affine.x(), "add x mismatch");
        assert_eq!(lw_sum.y(), cpu_sum_affine.y(), "add y mismatch");
    }

    // =========================================================================
    // Proptest Differential Fuzzing: GPU MSM vs CPU pippenger
    // =========================================================================

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 5,
            ..ProptestConfig::default()
        })]

        /// Differential fuzz: random scalar * G, GPU vs CPU.
        /// Uses single-point MSM to avoid the bucket_accumulation race condition.
        #[test]
        fn prop_metal_msm_random_scalar_vs_cpu(
            limbs in prop::array::uniform4(any::<u64>())
        ) {
            let g = BLS12381Curve::generator();
            let scalar = UnsignedInteger::<4>::from_limbs(limbs);

            // CPU reference
            let cpu_result =
                pippenger::msm(&[scalar], std::slice::from_ref(&g)).expect("CPU MSM failed");
            let cpu_affine = cpu_result.to_affine();

            // GPU
            let scalars_flat = scalar_to_gpu_limbs(&scalar);
            let points_flat = point_to_gpu_flat(&g);

            let mut msm = MetalMSM::new_bls12_381().expect("Metal device required");
            let gpu_result_flat = msm
                .compute(&scalars_flat, &points_flat)
                .expect("Metal MSM compute failed");

            let gpu_point = gpu_flat_to_point(&gpu_result_flat);
            let gpu_affine = gpu_point.to_affine();

            prop_assert_eq!(
                cpu_affine.x(), gpu_affine.x(),
                "x mismatch for random scalar"
            );
            prop_assert_eq!(
                cpu_affine.y(), gpu_affine.y(),
                "y mismatch for random scalar"
            );
        }

        /// Differential fuzz: random scalar * random_point, GPU vs CPU.
        /// Tests non-generator base points.
        #[test]
        fn prop_metal_msm_random_scalar_random_point_vs_cpu(
            scalar_limbs in prop::array::uniform4(any::<u64>()),
            point_power in 1u64..1000
        ) {
            let base_point = BLS12381Curve::generator().operate_with_self(point_power);
            let scalar = UnsignedInteger::<4>::from_limbs(scalar_limbs);

            // CPU reference
            let cpu_result =
                pippenger::msm(&[scalar], std::slice::from_ref(&base_point)).expect("CPU MSM failed");
            let cpu_affine = cpu_result.to_affine();

            // GPU
            let scalars_flat = scalar_to_gpu_limbs(&scalar);
            let points_flat = point_to_gpu_flat(&base_point);

            let mut msm = MetalMSM::new_bls12_381().expect("Metal device required");
            let gpu_result_flat = msm
                .compute(&scalars_flat, &points_flat)
                .expect("Metal MSM compute failed");

            let gpu_point = gpu_flat_to_point(&gpu_result_flat);
            let gpu_affine = gpu_point.to_affine();

            prop_assert_eq!(
                cpu_affine.x(), gpu_affine.x(),
                "x mismatch for random scalar * random point"
            );
            prop_assert_eq!(
                cpu_affine.y(), gpu_affine.y(),
                "y mismatch for random scalar * random point"
            );
        }

        /// Differential fuzz: CPU-side point doubling matches lambdaworks.
        #[test]
        fn prop_cpu_double_matches_lambdaworks(
            k in 1u64..100000
        ) {
            let p = BLS12381Curve::generator().operate_with_self(k);
            let lw_double = p.operate_with(&p).to_affine();

            let p_flat = point_to_gpu_flat(&p);
            let cpu_p = super::JacobianPoint::from_limbs(&p_flat);
            let cpu_double = cpu_p.double();
            let cpu_double_flat = cpu_double.to_limbs();
            let cpu_double_point = gpu_flat_to_point(&cpu_double_flat);
            let cpu_double_affine = cpu_double_point.to_affine();

            prop_assert_eq!(lw_double.x(), cpu_double_affine.x(), "double x mismatch for {}*G", k);
            prop_assert_eq!(lw_double.y(), cpu_double_affine.y(), "double y mismatch for {}*G", k);
        }

        /// Differential fuzz: CPU-side point addition matches lambdaworks.
        #[test]
        fn prop_cpu_add_matches_lambdaworks(
            k1 in 1u64..100000,
            k2 in 1u64..100000
        ) {
            let g = BLS12381Curve::generator();
            let p1 = g.operate_with_self(k1);
            let p2 = g.operate_with_self(k2);
            let lw_sum = p1.operate_with(&p2).to_affine();

            let p1_flat = point_to_gpu_flat(&p1);
            let p2_flat = point_to_gpu_flat(&p2);
            let cpu_p1 = super::JacobianPoint::from_limbs(&p1_flat);
            let cpu_p2 = super::JacobianPoint::from_limbs(&p2_flat);
            let cpu_sum = cpu_p1.add(&cpu_p2);
            let cpu_sum_flat = cpu_sum.to_limbs();
            let cpu_sum_point = gpu_flat_to_point(&cpu_sum_flat);
            let cpu_sum_affine = cpu_sum_point.to_affine();

            prop_assert_eq!(lw_sum.x(), cpu_sum_affine.x(), "add x mismatch for {}*G + {}*G", k1, k2);
            prop_assert_eq!(lw_sum.y(), cpu_sum_affine.y(), "add y mismatch for {}*G + {}*G", k1, k2);
        }
    }

    /// Differential test: edge case scalars that exercise window boundaries.
    #[test]
    fn test_metal_msm_window_boundary_scalars() {
        let g = BLS12381Curve::generator();

        // Scalars at window boundaries for window_size=16 (default)
        let edge_scalars: Vec<u64> = vec![
            0,                // zero (handled specially)
            1,                // minimum non-zero
            (1u64 << 16) - 1, // max single window
            1u64 << 16,       // exactly one window boundary
            (1u64 << 16) + 1, // just past boundary
            (1u64 << 32) - 1, // two full windows
            u64::MAX,         // max single limb
        ];

        for &s in &edge_scalars {
            if s == 0 {
                continue; // zero scalar gives identity, skip
            }
            let scalar = UnsignedInteger::<4>::from_u64(s);
            assert_gpu_cpu_match(&scalar, &g, &format!("edge scalar {} * G", s));
        }
    }
}
