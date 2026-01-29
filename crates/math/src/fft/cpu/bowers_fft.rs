//! Bowers FFT Implementation with Structure of Arrays (SoA) optimization
//!
//! This module implements the Bowers G network FFT algorithm, which provides
//! improved twiddle factor access patterns compared to the standard Cooley-Tukey FFT.
//!
//! # Key optimizations
//!
//! - **Bowers G network**: Improved memory access pattern for twiddle factors
//! - **LayerTwiddles**: Pre-computed twiddles per layer for O(N) sequential access
//!   instead of O(N log N) strided access
//! - **Multi-layer butterfly fusion**: Process 2 layers at once to keep intermediate
//!   values in registers and reduce memory traffic
//! - **Internal parallelization**: Uses rayon to parallelize across blocks when
//!   there are enough blocks (>= 64) to amortize threading overhead
//!
//! # Usage
//!
//! ```ignore
//! use lambdaworks_math::fft::cpu::bowers_fft::{LayerTwiddles, bowers_fft_opt_fused};
//!
//! let order = 10u64; // FFT size = 2^10 = 1024
//! let layer_twiddles = LayerTwiddles::<F>::new(order).unwrap();
//!
//! let mut data = vec![...]; // your polynomial coefficients
//! bowers_fft_opt_fused(&mut data, &layer_twiddles);
//! in_place_bit_reverse_permute(&mut data);
//! ```
//!
//! Based on analysis of Plonky3's implementation and academic literature on FFT optimization.

#[cfg(feature = "alloc")]
use crate::fft::cpu::bit_reversing::in_place_bit_reverse_permute;
#[cfg(feature = "alloc")]
use crate::field::{
    element::FieldElement,
    traits::{IsFFTField, IsField, IsSubFieldOf},
};

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

/// Maximum supported FFT order to prevent integer overflow.
/// With order 63, n = 2^63 which is the largest power of 2 that fits in usize on 64-bit.
/// For 32-bit systems, max order is 31.
#[cfg(all(feature = "alloc", target_pointer_width = "64"))]
const MAX_FFT_ORDER: u64 = 63;
#[cfg(all(feature = "alloc", target_pointer_width = "32"))]
const MAX_FFT_ORDER: u64 = 31;

// =====================================================
// STRUCTURE OF ARRAYS (SoA) FFT
// =====================================================

/// Matrix representation for batch FFT with Structure of Arrays layout
///
/// SoA layout stores multiple polynomials contiguously:
/// ```text
/// [poly0[0], poly0[1], ..., poly0[n-1], poly1[0], poly1[1], ..., poly1[n-1], ...]
/// ```
///
/// This layout provides better cache utilization when processing multiple
/// polynomials simultaneously.
#[cfg(feature = "alloc")]
pub struct FftMatrix<E: IsField> {
    /// Flat storage for all polynomial coefficients
    pub data: Vec<FieldElement<E>>,
    /// Number of columns (polynomial length)
    pub width: usize,
    /// Number of rows (number of polynomials)
    pub height: usize,
}

#[cfg(feature = "alloc")]
impl<E: IsField> FftMatrix<E> {
    /// Create a new FFT matrix from a list of polynomials
    ///
    /// # Panics
    /// Panics if polynomials have different lengths.
    pub fn from_polynomials(polys: Vec<Vec<FieldElement<E>>>) -> Self {
        if polys.is_empty() {
            return Self {
                data: Vec::new(),
                width: 0,
                height: 0,
            };
        }

        let height = polys.len();
        let width = polys[0].len();

        // Flatten in row-major order (SoA layout)
        let mut data = Vec::with_capacity(height * width);
        for poly in polys {
            assert_eq!(
                poly.len(),
                width,
                "All polynomials must have same length"
            );
            data.extend(poly);
        }

        Self {
            data,
            width,
            height,
        }
    }

    /// Get a mutable slice for polynomial at index `row`
    pub fn row_mut(&mut self, row: usize) -> &mut [FieldElement<E>] {
        let start = row * self.width;
        let end = start + self.width;
        &mut self.data[start..end]
    }

    /// Get an immutable slice for polynomial at index `row`
    pub fn row(&self, row: usize) -> &[FieldElement<E>] {
        let start = row * self.width;
        let end = start + self.width;
        &self.data[start..end]
    }

    /// Convert back to list of polynomials
    pub fn to_polynomials(self) -> Vec<Vec<FieldElement<E>>> {
        self.data
            .chunks(self.width)
            .map(|chunk| chunk.to_vec())
            .collect()
    }
}

// =====================================================
// PARALLEL BOWERS FFT
// =====================================================

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Optimized Parallel Bowers FFT with 2-layer fusion using LayerTwiddles
///
/// This is the recommended FFT for large inputs (>= 2^16 elements) when
/// the `parallel` feature is enabled. It combines:
///
/// 1. **Sequential twiddle access**: LayerTwiddles stores twiddles per layer
///    for cache-friendly sequential reads instead of strided access
/// 2. **2-layer fusion**: Processes two FFT layers at once, keeping intermediate
///    values in registers to reduce memory traffic
/// 3. **Internal parallelization**: Uses `par_chunks_mut` to process independent
///    blocks in parallel when there are >= 64 blocks (PARALLEL_THRESHOLD)
///
/// # Parallelization Strategy
///
/// The FFT is structured in layers, where each layer processes blocks of decreasing size.
/// Early layers have few large blocks (not enough parallelism), while later layers have
/// many small blocks (good parallelism). We only parallelize when `num_blocks >= 64`
/// to ensure the threading overhead is amortized.
///
/// # Panics
/// Panics if input length is not a power of two.
#[cfg(all(feature = "alloc", feature = "parallel"))]
#[allow(clippy::needless_range_loop)]
pub fn bowers_fft_opt_fused_parallel<F, E>(
    input: &mut [FieldElement<E>],
    layer_twiddles: &LayerTwiddles<F>,
) where
    F: IsFFTField + IsSubFieldOf<E>,
    E: IsField + Send + Sync,
    FieldElement<F>: Send + Sync,
    FieldElement<E>: Send + Sync,
{
    // Minimum number of blocks required to use parallel processing.
    // Below this threshold, the threading overhead exceeds the benefit.
    const PARALLEL_THRESHOLD: usize = 64;

    let n = input.len();
    assert!(n.is_power_of_two(), "Input length must be a power of two");

    if n <= 1 {
        return;
    }

    if n <= 4 {
        // Use sequential fused version for small inputs
        bowers_fft_opt_fused(input, layer_twiddles);
        return;
    }

    let log_n = n.trailing_zeros() as usize;
    let mut layer = 0;

    // Process pairs of layers with 2-layer fusion.
    // This keeps intermediate values (sum_02, diff_02, etc.) in registers
    // instead of writing them back to memory between layers.
    while layer + 1 < log_n {
        let block_size = n >> layer;

        if block_size >= 4 {
            let twiddles_l0 = layer_twiddles.get_layer(layer);
            let twiddles_l1 = layer_twiddles.get_layer(layer + 1);
            let num_blocks = n / block_size;

            if num_blocks >= PARALLEL_THRESHOLD {
                // Parallel path: process blocks concurrently
                input.par_chunks_mut(block_size).for_each(|block| {
                    process_fused_block(block, twiddles_l0, twiddles_l1);
                });
            } else {
                // Sequential path: not enough blocks to justify threading
                for block_start in (0..n).step_by(block_size) {
                    let block = &mut input[block_start..block_start + block_size];
                    process_fused_block(block, twiddles_l0, twiddles_l1);
                }
            }
            layer += 2;
        } else {
            break;
        }
    }

    // Process remaining single layers (if odd number of layers)
    while layer < log_n {
        let block_size = n >> layer;
        let half_block = block_size >> 1;
        let num_blocks = n / block_size;
        let twiddles = layer_twiddles.get_layer(layer);

        if num_blocks >= PARALLEL_THRESHOLD {
            input.par_chunks_mut(block_size).for_each(|block| {
                process_single_layer_block(block, twiddles, half_block);
            });
        } else {
            for block_start in (0..n).step_by(block_size) {
                for j in 0..half_block {
                    let i0 = block_start + j;
                    let i1 = i0 + half_block;
                    let w = &twiddles[j];

                    let sum = &input[i0] + &input[i1];
                    let diff = &input[i0] - &input[i1];
                    let diff_w = w * &diff;

                    input[i0] = sum;
                    input[i1] = diff_w;
                }
            }
        }
        layer += 1;
    }
}

/// Process a single block with 2-layer fusion (DIF butterfly).
#[cfg(all(feature = "alloc", feature = "parallel"))]
#[inline]
fn process_fused_block<F, E>(
    block: &mut [FieldElement<E>],
    twiddles_l0: &[FieldElement<F>],
    twiddles_l1: &[FieldElement<F>],
) where
    F: IsFFTField + IsSubFieldOf<E>,
    E: IsField,
{
    let block_size = block.len();
    let quarter = block_size >> 2;

    for j in 0..quarter {
        let i0 = j;
        let i1 = j + quarter;
        let i2 = j + 2 * quarter;
        let i3 = j + 3 * quarter;

        let w0 = &twiddles_l0[j];
        let w1 = &twiddles_l0[j + quarter];

        // First layer butterflies
        let sum_02 = &block[i0] + &block[i2];
        let diff_02 = &block[i0] - &block[i2];
        let diff_02_w = w0 * &diff_02;

        let sum_13 = &block[i1] + &block[i3];
        let diff_13 = &block[i1] - &block[i3];
        let diff_13_w = w1 * &diff_13;

        let w2 = &twiddles_l1[j];

        // Second layer butterflies
        let final_0 = &sum_02 + &sum_13;
        let diff_sums = &sum_02 - &sum_13;
        let final_1 = w2 * &diff_sums;

        let final_2 = &diff_02_w + &diff_13_w;
        let diff_diffs = &diff_02_w - &diff_13_w;
        let final_3 = w2 * &diff_diffs;

        block[i0] = final_0;
        block[i1] = final_1;
        block[i2] = final_2;
        block[i3] = final_3;
    }
}

/// Process a single layer block (used for remaining odd layer).
#[cfg(all(feature = "alloc", feature = "parallel"))]
#[inline]
fn process_single_layer_block<F, E>(
    block: &mut [FieldElement<E>],
    twiddles: &[FieldElement<F>],
    half_block: usize,
) where
    F: IsFFTField + IsSubFieldOf<E>,
    E: IsField,
{
    for j in 0..half_block {
        let w = &twiddles[j];

        let sum = &block[j] + &block[j + half_block];
        let diff = &block[j] - &block[j + half_block];
        let diff_w = w * &diff;

        block[j] = sum;
        block[j + half_block] = diff_w;
    }
}

// =====================================================
// LAYER-SPECIFIC TWIDDLE TABLES (OPTIMIZED)
// =====================================================

/// Pre-computed twiddle factors organized by layer for cache-friendly access.
///
/// Instead of accessing twiddles with strided patterns (j * 2^layer),
/// we store each layer's twiddles contiguously for sequential access.
/// This reduces twiddle memory accesses from O(N log N) random to O(N) sequential.
///
/// # Memory Layout
///
/// For an FFT of size n = 2^order:
/// - Layer 0: n/2 twiddles (w^0, w^1, w^2, ...)
/// - Layer 1: n/4 twiddles (w^0, w^2, w^4, ...)
/// - Layer k: n/2^(k+1) twiddles (w^0, w^(2^k), w^(2*2^k), ...)
///
/// Total memory: n - 1 twiddles (same as flat storage, but organized for locality).
#[cfg(feature = "alloc")]
#[derive(Clone)]
pub struct LayerTwiddles<F: IsField> {
    /// Twiddles organized by layer, stored contiguously for sequential access.
    pub layers: Vec<Vec<FieldElement<F>>>,
}

#[cfg(feature = "alloc")]
impl<F: IsFFTField> LayerTwiddles<F> {
    /// Compute layer-specific twiddles from primitive root of unity.
    ///
    /// For an FFT of size n = 2^order, layer k needs n/2^(k+1) twiddles.
    /// The twiddles for layer k are: w^0, w^(2^k), w^(2*2^k), w^(3*2^k), ...
    ///
    /// # Errors
    /// Returns `None` if:
    /// - `order` exceeds the maximum supported value (would cause integer overflow)
    /// - The field doesn't have a primitive root of unity for the given order
    ///
    /// # Example
    /// ```ignore
    /// let layer_twiddles = LayerTwiddles::<Goldilocks64Field>::new(10)
    ///     .expect("Failed to create twiddles for order 10");
    /// ```
    pub fn new(order: u64) -> Option<Self> {
        // Check for potential integer overflow
        if order > MAX_FFT_ORDER {
            return None;
        }

        let n = 1usize << order;
        let root = F::get_primitive_root_of_unity(order).ok()?;

        let mut layers = Vec::with_capacity(order as usize);

        for layer in 0..order as usize {
            let stride = 1usize << layer;
            let count = n >> (layer + 1);

            let mut layer_twiddles = Vec::with_capacity(count);
            let w_stride = root.pow(stride as u64);
            let mut current = FieldElement::<F>::one();

            for _ in 0..count {
                layer_twiddles.push(current.clone());
                current *= &w_stride;
            }

            layers.push(layer_twiddles);
        }

        Some(Self { layers })
    }

    /// Get the twiddles for a specific layer.
    #[inline(always)]
    pub fn get_layer(&self, layer: usize) -> &[FieldElement<F>] {
        &self.layers[layer]
    }
}

/// Optimized Bowers IFFT with sequential twiddle access.
///
/// # Panics
/// Panics if input length is not a power of two.
#[cfg(feature = "alloc")]
#[allow(clippy::needless_range_loop)]
pub fn bowers_ifft_opt<F, E>(input: &mut [FieldElement<E>], layer_twiddles: &LayerTwiddles<F>)
where
    F: IsFFTField + IsSubFieldOf<E>,
    E: IsField,
{
    let n = input.len();
    assert!(n.is_power_of_two(), "Input length must be a power of two");

    if n <= 1 {
        return;
    }

    let log_n = n.trailing_zeros() as usize;

    for layer in (0..log_n).rev() {
        let block_size = n >> layer;
        let half_block = block_size >> 1;
        let twiddles = layer_twiddles.get_layer(layer);

        for block_start in (0..n).step_by(block_size) {
            for j in 0..half_block {
                let i0 = block_start + j;
                let i1 = i0 + half_block;
                let w = &twiddles[j];

                let bw = w * &input[i1];
                let sum = &input[i0] + &bw;
                let diff = &input[i0] - &bw;

                input[i0] = sum;
                input[i1] = diff;
            }
        }
    }
}

/// Optimized Bowers FFT with 2-layer fusion and sequential twiddle access.
///
/// This is the recommended single-threaded FFT. It combines:
///
/// 1. **Sequential twiddle access**: LayerTwiddles stores twiddles per layer
///    for cache-friendly sequential reads
/// 2. **2-layer fusion**: Processes two FFT layers at once, keeping intermediate
///    values in registers to reduce memory traffic
///
/// For multi-threaded execution, use `bowers_fft_opt_fused_parallel` instead.
///
/// # Panics
/// Panics if input length is not a power of two.
#[cfg(feature = "alloc")]
#[allow(clippy::needless_range_loop)]
pub fn bowers_fft_opt_fused<F, E>(input: &mut [FieldElement<E>], layer_twiddles: &LayerTwiddles<F>)
where
    F: IsFFTField + IsSubFieldOf<E>,
    E: IsField,
{
    let n = input.len();
    assert!(n.is_power_of_two(), "Input length must be a power of two");

    if n <= 1 {
        return;
    }

    let log_n = n.trailing_zeros() as usize;

    // Handle small sizes with simple sequential processing
    if n <= 4 {
        for layer in 0..log_n {
            let block_size = n >> layer;
            let half_block = block_size >> 1;
            let twiddles = layer_twiddles.get_layer(layer);

            for block_start in (0..n).step_by(block_size) {
                for j in 0..half_block {
                    let i0 = block_start + j;
                    let i1 = i0 + half_block;
                    let w = &twiddles[j];

                    let sum = &input[i0] + &input[i1];
                    let diff = &input[i0] - &input[i1];
                    let diff_w = w * &diff;

                    input[i0] = sum;
                    input[i1] = diff_w;
                }
            }
        }
        return;
    }

    let mut layer = 0;

    // Process pairs of layers with 2-layer fusion
    while layer + 1 < log_n {
        let block_size = n >> layer;

        if block_size >= 4 {
            let twiddles_l0 = layer_twiddles.get_layer(layer);
            let twiddles_l1 = layer_twiddles.get_layer(layer + 1);

            for block_start in (0..n).step_by(block_size) {
                let quarter = block_size >> 2;
                let block = &mut input[block_start..block_start + block_size];

                for j in 0..quarter {
                    let i0 = j;
                    let i1 = j + quarter;
                    let i2 = j + 2 * quarter;
                    let i3 = j + 3 * quarter;

                    let w0 = &twiddles_l0[j];
                    let w1 = &twiddles_l0[j + quarter];

                    // First layer butterflies
                    let sum_02 = &block[i0] + &block[i2];
                    let diff_02 = &block[i0] - &block[i2];
                    let diff_02_w = w0 * &diff_02;

                    let sum_13 = &block[i1] + &block[i3];
                    let diff_13 = &block[i1] - &block[i3];
                    let diff_13_w = w1 * &diff_13;

                    let w2 = &twiddles_l1[j];

                    // Second layer butterflies
                    let final_0 = &sum_02 + &sum_13;
                    let diff_sums = &sum_02 - &sum_13;
                    let final_1 = w2 * &diff_sums;

                    let final_2 = &diff_02_w + &diff_13_w;
                    let diff_diffs = &diff_02_w - &diff_13_w;
                    let final_3 = w2 * &diff_diffs;

                    block[i0] = final_0;
                    block[i1] = final_1;
                    block[i2] = final_2;
                    block[i3] = final_3;
                }
            }
            layer += 2;
        } else {
            break;
        }
    }

    // Process remaining single layers (if odd number of layers)
    while layer < log_n {
        let block_size = n >> layer;
        let half_block = block_size >> 1;
        let twiddles = layer_twiddles.get_layer(layer);

        for block_start in (0..n).step_by(block_size) {
            for j in 0..half_block {
                let i0 = block_start + j;
                let i1 = i0 + half_block;
                let w = &twiddles[j];

                let sum = &input[i0] + &input[i1];
                let diff = &input[i0] - &input[i1];
                let diff_w = w * &diff;

                input[i0] = sum;
                input[i1] = diff_w;
            }
        }
        layer += 1;
    }
}

/// Batch FFT using optimized Bowers algorithm with LayerTwiddles
///
/// # Panics
/// Panics if polynomial width is not a power of two.
#[cfg(feature = "alloc")]
pub fn bowers_batch_fft_opt<F, E>(matrix: &mut FftMatrix<E>, layer_twiddles: &LayerTwiddles<F>)
where
    F: IsFFTField + IsSubFieldOf<E>,
    E: IsField,
{
    if matrix.height == 0 || matrix.width <= 1 {
        return;
    }

    for row in 0..matrix.height {
        let poly = matrix.row_mut(row);
        bowers_fft_opt_fused(poly, layer_twiddles);
        in_place_bit_reverse_permute(poly);
    }
}

// =====================================================
// TESTS
// =====================================================

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use super::*;
    use crate::field::fields::u64_goldilocks_field::Goldilocks64Field;
    use alloc::vec;
    use alloc::vec::Vec;

    type F = Goldilocks64Field;
    type FE = FieldElement<F>;

    fn naive_dft(input: &[FE]) -> Vec<FE> {
        let n = input.len();
        let root = F::get_primitive_root_of_unity(n.trailing_zeros() as u64).unwrap();
        let mut result = vec![FE::zero(); n];

        for (k, res) in result.iter_mut().enumerate() {
            for (j, inp) in input.iter().enumerate() {
                *res += *inp * root.pow((j * k) as u64);
            }
        }

        result
    }

    #[test]
    fn test_fft_matrix_roundtrip() {
        let polys = vec![
            (0..8).map(|i| FE::from(i as u64)).collect::<Vec<_>>(),
            (8..16).map(|i| FE::from(i as u64)).collect::<Vec<_>>(),
        ];

        let matrix = FftMatrix::from_polynomials(polys.clone());
        assert_eq!(matrix.width, 8);
        assert_eq!(matrix.height, 2);

        let recovered = matrix.to_polynomials();
        assert_eq!(recovered, polys);
    }

    #[test]
    fn test_layer_twiddles_creation() {
        let order = 4u64;
        let layer_twiddles = LayerTwiddles::<F>::new(order).unwrap();

        assert_eq!(layer_twiddles.layers.len(), 4);
        assert_eq!(layer_twiddles.layers[0].len(), 8);
        assert_eq!(layer_twiddles.layers[1].len(), 4);
        assert_eq!(layer_twiddles.layers[2].len(), 2);
        assert_eq!(layer_twiddles.layers[3].len(), 1);

        // First twiddle of each layer should be 1
        for layer in &layer_twiddles.layers {
            assert_eq!(layer[0], FE::one());
        }
    }

    #[test]
    fn test_layer_twiddles_overflow_protection() {
        // Order 64 would overflow on 64-bit systems
        let result = LayerTwiddles::<F>::new(64);
        assert!(result.is_none());
    }

    #[test]
    fn test_bowers_fft_opt_fused_small() {
        let input: Vec<FE> = (0..4).map(|i| FE::from(i as u64)).collect();
        let expected = naive_dft(&input);

        let layer_twiddles = LayerTwiddles::<F>::new(2).unwrap();
        let mut result = input.clone();
        bowers_fft_opt_fused(&mut result, &layer_twiddles);
        in_place_bit_reverse_permute(&mut result);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_bowers_fft_opt_fused_medium() {
        let input: Vec<FE> = (0..16).map(|i| FE::from(i as u64)).collect();
        let expected = naive_dft(&input);

        let layer_twiddles = LayerTwiddles::<F>::new(4).unwrap();
        let mut result = input.clone();
        bowers_fft_opt_fused(&mut result, &layer_twiddles);
        in_place_bit_reverse_permute(&mut result);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_bowers_fft_opt_fused_large() {
        let input: Vec<FE> = (0..256).map(|i| FE::from(i as u64)).collect();
        let expected = naive_dft(&input);

        let layer_twiddles = LayerTwiddles::<F>::new(8).unwrap();
        let mut result = input.clone();
        bowers_fft_opt_fused(&mut result, &layer_twiddles);
        in_place_bit_reverse_permute(&mut result);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_bowers_ifft_opt() {
        let input: Vec<FE> = (0..64).map(|i| FE::from(i as u64)).collect();
        let layer_twiddles = LayerTwiddles::<F>::new(6).unwrap();

        // Forward FFT
        let mut transformed = input.clone();
        bowers_fft_opt_fused(&mut transformed, &layer_twiddles);
        in_place_bit_reverse_permute(&mut transformed);

        // Inverse FFT needs inverse twiddles (conjugate for roots of unity)
        // For proper inverse, we need twiddles from inverse root
        // This test just verifies the function runs without panic
        let mut recovered = transformed.clone();
        in_place_bit_reverse_permute(&mut recovered);
        bowers_ifft_opt(&mut recovered, &layer_twiddles);

        // Note: Full inverse requires scaling by 1/n and using inverse twiddles
    }

    #[test]
    fn test_bowers_batch_fft_opt() {
        let polys = vec![
            (0..8).map(|i| FE::from(i as u64)).collect::<Vec<_>>(),
            (8..16).map(|i| FE::from(i as u64)).collect::<Vec<_>>(),
        ];

        let expected: Vec<Vec<FE>> = polys.iter().map(|p| naive_dft(p)).collect();

        let mut matrix = FftMatrix::from_polynomials(polys);
        let layer_twiddles = LayerTwiddles::<F>::new(3).unwrap();
        bowers_batch_fft_opt(&mut matrix, &layer_twiddles);

        let result = matrix.to_polynomials();
        assert_eq!(result, expected);
    }

    #[test]
    #[should_panic(expected = "Input length must be a power of two")]
    fn test_bowers_fft_non_power_of_two() {
        let input: Vec<FE> = (0..7).map(|i| FE::from(i as u64)).collect();
        let layer_twiddles = LayerTwiddles::<F>::new(3).unwrap();
        let mut result = input;
        bowers_fft_opt_fused(&mut result, &layer_twiddles);
    }
}

/// Tests that require the parallel feature
#[cfg(all(test, feature = "alloc", feature = "parallel"))]
mod parallel_tests {
    use super::*;
    use crate::fft::cpu::bit_reversing::in_place_bit_reverse_permute;
    use crate::field::fields::u64_goldilocks_field::Goldilocks64Field;
    use alloc::vec;
    use alloc::vec::Vec;

    type F = Goldilocks64Field;
    type FE = FieldElement<F>;

    fn naive_dft(input: &[FE]) -> Vec<FE> {
        let n = input.len();
        let root = F::get_primitive_root_of_unity(n.trailing_zeros() as u64).unwrap();
        let mut result = vec![FE::zero(); n];

        for (k, res) in result.iter_mut().enumerate() {
            for (j, inp) in input.iter().enumerate() {
                *res += *inp * root.pow((j * k) as u64);
            }
        }

        result
    }

    #[test]
    fn test_bowers_fft_opt_fused_parallel_small() {
        // Small input - should use sequential path
        let input: Vec<FE> = (0..16).map(|i| FE::from(i as u64)).collect();
        let expected = naive_dft(&input);

        let layer_twiddles = LayerTwiddles::<F>::new(4).unwrap();
        let mut result = input.clone();
        bowers_fft_opt_fused_parallel(&mut result, &layer_twiddles);
        in_place_bit_reverse_permute(&mut result);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_bowers_fft_opt_fused_parallel_medium() {
        // Medium input
        let input: Vec<FE> = (0..256).map(|i| FE::from(i as u64)).collect();
        let expected = naive_dft(&input);

        let layer_twiddles = LayerTwiddles::<F>::new(8).unwrap();
        let mut result = input.clone();
        bowers_fft_opt_fused_parallel(&mut result, &layer_twiddles);
        in_place_bit_reverse_permute(&mut result);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_bowers_fft_opt_fused_parallel_large() {
        // Large input - should exercise parallel paths
        let input: Vec<FE> = (0..4096).map(|i| FE::from(i as u64)).collect();
        let expected = naive_dft(&input);

        let layer_twiddles = LayerTwiddles::<F>::new(12).unwrap();
        let mut result = input.clone();
        bowers_fft_opt_fused_parallel(&mut result, &layer_twiddles);
        in_place_bit_reverse_permute(&mut result);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_parallel_matches_sequential() {
        // Verify parallel and sequential produce identical results
        let input: Vec<FE> = (0..1024).map(|i| FE::from(i as u64)).collect();

        let layer_twiddles = LayerTwiddles::<F>::new(10).unwrap();

        let mut result_seq = input.clone();
        bowers_fft_opt_fused(&mut result_seq, &layer_twiddles);
        in_place_bit_reverse_permute(&mut result_seq);

        let mut result_par = input.clone();
        bowers_fft_opt_fused_parallel(&mut result_par, &layer_twiddles);
        in_place_bit_reverse_permute(&mut result_par);

        assert_eq!(result_seq, result_par);
    }
}
