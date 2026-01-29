//! Bowers FFT Implementation with optimized twiddle factor access
//!
//! This module implements the Bowers G network FFT algorithm with layer-specific
//! twiddle tables for cache-friendly sequential access.
//!
//! Key optimizations:
//! - Layer-specific twiddle tables: O(N) sequential access instead of O(N log N) strided
//! - Bowers G network: DIF structure with improved memory patterns
//! - Multi-layer butterfly fusion: Process 2 layers at once to reduce memory traffic
//! - Twiddle caching: RwLock-based cache for reusing twiddles across FFT calls
//!
//! Based on analysis of Plonky3's implementation and academic literature on FFT optimization.

use crate::fft::cpu::bit_reversing::in_place_bit_reverse_permute;
use crate::field::{
    element::FieldElement,
    traits::{IsFFTField, IsField, IsSubFieldOf},
};

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

#[cfg(feature = "std")]
use std::collections::BTreeMap;
#[cfg(feature = "std")]
use std::sync::{Arc, RwLock};

// =====================================================
// LAYER-SPECIFIC TWIDDLE TABLES
// =====================================================

/// Pre-computed twiddle factors organized by layer for cache-friendly access.
///
/// Instead of accessing twiddles with strided patterns (j * 2^layer),
/// we store each layer's twiddles contiguously for sequential access.
/// This reduces twiddle memory accesses from O(N log N) random to O(N) sequential.
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
    pub fn new(order: u64) -> Self {
        let n = 1usize << order;
        let root = F::get_primitive_root_of_unity(order).unwrap();

        let mut layers = Vec::with_capacity(order as usize);

        for layer in 0..order as usize {
            let stride = 1usize << layer;
            let count = n >> (layer + 1);

            let mut layer_twiddles = Vec::with_capacity(count);
            let w_stride = root.pow(stride as u64);
            let mut current = FieldElement::<F>::one();

            for _ in 0..count {
                layer_twiddles.push(current.clone());
                current = current * &w_stride;
            }

            layers.push(layer_twiddles);
        }

        Self { layers }
    }

    #[inline(always)]
    pub fn get_layer(&self, layer: usize) -> &[FieldElement<F>] {
        &self.layers[layer]
    }
}

// =====================================================
// CACHED BOWERS FFT (like Plonky3's Radix2Dit)
// =====================================================

/// Bowers FFT with twiddle caching for optimal performance.
///
/// This struct caches computed `LayerTwiddles` by order (log2 of FFT size),
/// allowing fast reuse across multiple FFT operations. Uses double-checked
/// locking with `RwLock` for thread-safe concurrent access.
///
/// # Example
///
/// ```ignore
/// use lambdaworks_math::fft::cpu::bowers_fft::BowersFft;
/// use lambdaworks_math::field::fields::u64_goldilocks_field::Goldilocks64Field;
///
/// // Create once, reuse many times
/// let fft = BowersFft::<Goldilocks64Field>::new();
///
/// // First call computes and caches twiddles
/// fft.fft(&mut poly1);
///
/// // Subsequent calls reuse cached twiddles (fast path with read lock)
/// fft.fft(&mut poly2);
/// fft.fft(&mut poly3);
/// ```
#[cfg(feature = "std")]
#[derive(Clone, Default)]
pub struct BowersFft<F: IsFFTField> {
    /// Cached twiddles indexed by order (log2 of FFT size).
    ///
    /// Uses `RwLock` for interior mutability with thread safety.
    /// Multiple readers can access cached twiddles concurrently.
    twiddles: Arc<RwLock<BTreeMap<u64, Arc<LayerTwiddles<F>>>>>,
}

#[cfg(feature = "std")]
impl<F: IsFFTField> BowersFft<F> {
    /// Create a new `BowersFft` instance with an empty cache.
    pub fn new() -> Self {
        Self {
            twiddles: Arc::new(RwLock::new(BTreeMap::new())),
        }
    }

    /// Get or compute twiddles for the given order.
    ///
    /// Uses double-checked locking pattern:
    /// 1. Fast path: Try to get with read lock (no contention for cached values)
    /// 2. Slow path: Acquire write lock and compute if needed
    fn get_or_compute_twiddles(&self, order: u64) -> Arc<LayerTwiddles<F>> {
        // Fast path: check if twiddles exist with read lock
        if let Some(twiddles) = self.twiddles.read().unwrap().get(&order) {
            return Arc::clone(twiddles);
        }

        // Slow path: compute and insert with write lock
        let mut cache = self.twiddles.write().unwrap();

        // Double-check: another thread might have computed while we waited
        cache
            .entry(order)
            .or_insert_with(|| Arc::new(LayerTwiddles::new(order)))
            .clone()
    }

    /// Perform FFT on input with automatic twiddle caching.
    ///
    /// The result is in natural order (bit-reversal is applied internally).
    pub fn fft<E>(&self, input: &mut [FieldElement<E>])
    where
        F: IsSubFieldOf<E>,
        E: IsField,
    {
        if input.len() <= 1 {
            return;
        }

        let order = input.len().trailing_zeros() as u64;
        let twiddles = self.get_or_compute_twiddles(order);
        bowers_fft_opt(input, &twiddles);
        in_place_bit_reverse_permute(input);
    }

    /// Perform FFT with 2-layer fusion and automatic twiddle caching.
    ///
    /// The result is in natural order (bit-reversal is applied internally).
    pub fn fft_fused<E>(&self, input: &mut [FieldElement<E>])
    where
        F: IsSubFieldOf<E>,
        E: IsField,
    {
        if input.len() <= 1 {
            return;
        }

        let order = input.len().trailing_zeros() as u64;
        let twiddles = self.get_or_compute_twiddles(order);
        bowers_fft_opt_fused(input, &twiddles);
        in_place_bit_reverse_permute(input);
    }

    /// Perform IFFT on input with automatic twiddle caching.
    ///
    /// Note: Does not include the 1/n scaling factor.
    pub fn ifft<E>(&self, input: &mut [FieldElement<E>])
    where
        F: IsSubFieldOf<E>,
        E: IsField,
    {
        if input.len() <= 1 {
            return;
        }

        let order = input.len().trailing_zeros() as u64;
        let twiddles = self.get_or_compute_twiddles(order);
        in_place_bit_reverse_permute(input);
        bowers_ifft_opt(input, &twiddles);
    }

    /// Get a reference to cached twiddles for a given order, if they exist.
    pub fn get_cached_twiddles(&self, order: u64) -> Option<Arc<LayerTwiddles<F>>> {
        self.twiddles.read().unwrap().get(&order).cloned()
    }

    /// Pre-compute and cache twiddles for a given order.
    ///
    /// Useful for warming up the cache before performance-critical sections.
    pub fn precompute(&self, order: u64) {
        let _ = self.get_or_compute_twiddles(order);
    }
}

// =====================================================
// OPTIMIZED BOWERS FFT WITH LAYER TWIDDLES
// =====================================================

/// Optimized Bowers FFT with sequential twiddle access.
///
/// Uses pre-computed layer-specific twiddles for O(N) sequential memory access.
#[cfg(feature = "alloc")]
pub fn bowers_fft_opt<F, E>(input: &mut [FieldElement<E>], layer_twiddles: &LayerTwiddles<F>)
where
    F: IsFFTField + IsSubFieldOf<E>,
    E: IsField,
{
    let n = input.len();
    debug_assert!(n.is_power_of_two());

    if n <= 1 {
        return;
    }

    let log_n = n.trailing_zeros() as usize;

    for layer in 0..log_n {
        let block_size = n >> layer;
        let half_block = block_size >> 1;
        let twiddles = layer_twiddles.get_layer(layer);

        for block_start in (0..n).step_by(block_size) {
            for j in 0..half_block {
                let i0 = block_start + j;
                let i1 = i0 + half_block;
                let w = &twiddles[j]; // Sequential access!

                let sum = &input[i0] + &input[i1];
                let diff = &input[i0] - &input[i1];
                let diff_w = w * &diff;

                input[i0] = sum;
                input[i1] = diff_w;
            }
        }
    }
}

/// Optimized Bowers IFFT with sequential twiddle access.
#[cfg(feature = "alloc")]
pub fn bowers_ifft_opt<F, E>(input: &mut [FieldElement<E>], layer_twiddles: &LayerTwiddles<F>)
where
    F: IsFFTField + IsSubFieldOf<E>,
    E: IsField,
{
    let n = input.len();
    debug_assert!(n.is_power_of_two());

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

/// Optimized Bowers FFT with 2-layer fusion.
#[cfg(feature = "alloc")]
pub fn bowers_fft_opt_fused<F, E>(input: &mut [FieldElement<E>], layer_twiddles: &LayerTwiddles<F>)
where
    F: IsFFTField + IsSubFieldOf<E>,
    E: IsField,
{
    let n = input.len();
    debug_assert!(n.is_power_of_two());

    if n <= 1 {
        return;
    }

    if n <= 4 {
        bowers_fft_opt(input, layer_twiddles);
        return;
    }

    let log_n = n.trailing_zeros() as usize;
    let mut layer = 0;

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

                    let sum_02 = &block[i0] + &block[i2];
                    let diff_02 = &block[i0] - &block[i2];
                    let diff_02_w = w0 * &diff_02;

                    let sum_13 = &block[i1] + &block[i3];
                    let diff_13 = &block[i1] - &block[i3];
                    let diff_13_w = w1 * &diff_13;

                    let w2 = &twiddles_l1[j];

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

// =====================================================
// LEGACY BOWERS FFT (backward compatibility)
// =====================================================

/// Legacy Bowers FFT with strided twiddle access.
/// For better performance, use `bowers_fft_opt` with `LayerTwiddles`.
pub fn bowers_fft<F, E>(input: &mut [FieldElement<E>], twiddles: &[FieldElement<F>])
where
    F: IsFFTField + IsSubFieldOf<E>,
    E: IsField,
{
    let n = input.len();
    debug_assert!(n.is_power_of_two());

    if n <= 1 {
        return;
    }

    let log_n = n.trailing_zeros() as usize;

    for layer in 0..log_n {
        let block_size = n >> layer;
        let half_block = block_size >> 1;
        let twiddle_stride = 1 << layer;

        for block_start in (0..n).step_by(block_size) {
            for j in 0..half_block {
                let i0 = block_start + j;
                let i1 = i0 + half_block;

                let tw_idx = j * twiddle_stride;
                let w = if tw_idx < twiddles.len() {
                    &twiddles[tw_idx]
                } else {
                    &twiddles[0]
                };

                let sum = &input[i0] + &input[i1];
                let diff = &input[i0] - &input[i1];
                let diff_w = w * &diff;

                input[i0] = sum;
                input[i1] = diff_w;
            }
        }
    }
}

/// Legacy Bowers IFFT.
pub fn bowers_ifft<F, E>(input: &mut [FieldElement<E>], twiddles: &[FieldElement<F>])
where
    F: IsFFTField + IsSubFieldOf<E>,
    E: IsField,
{
    let n = input.len();
    debug_assert!(n.is_power_of_two());

    if n <= 1 {
        return;
    }

    let log_n = n.trailing_zeros() as usize;

    for layer in (0..log_n).rev() {
        let block_size = n >> layer;
        let half_block = block_size >> 1;
        let twiddle_stride = 1 << layer;

        for block_start in (0..n).step_by(block_size) {
            for j in 0..half_block {
                let i0 = block_start + j;
                let i1 = i0 + half_block;

                let tw_idx = j * twiddle_stride;
                let w = if tw_idx < twiddles.len() {
                    &twiddles[tw_idx]
                } else {
                    &twiddles[0]
                };

                let bw = w * &input[i1];
                let sum = &input[i0] + &bw;
                let diff = &input[i0] - &bw;

                input[i0] = sum;
                input[i1] = diff;
            }
        }
    }
}

/// Legacy fused Bowers FFT.
pub fn bowers_fft_fused<F, E>(input: &mut [FieldElement<E>], twiddles: &[FieldElement<F>])
where
    F: IsFFTField + IsSubFieldOf<E>,
    E: IsField,
{
    let n = input.len();
    debug_assert!(n.is_power_of_two());

    if n <= 1 {
        return;
    }

    if n <= 4 {
        bowers_fft(input, twiddles);
        return;
    }

    let log_n = n.trailing_zeros() as usize;
    let mut layer = 0;

    while layer + 1 < log_n {
        let block_size = n >> layer;

        if block_size >= 4 {
            let tw_stride_l0 = 1 << layer;
            let tw_stride_l1 = 1 << (layer + 1);

            for block_start in (0..n).step_by(block_size) {
                let quarter = block_size >> 2;
                let block = &mut input[block_start..block_start + block_size];

                for j in 0..quarter {
                    let i0 = j;
                    let i1 = j + quarter;
                    let i2 = j + 2 * quarter;
                    let i3 = j + 3 * quarter;

                    let w0 = &twiddles[(j * tw_stride_l0) % twiddles.len()];
                    let w1 = &twiddles[((j + quarter) * tw_stride_l0) % twiddles.len()];

                    let sum_01 = &block[i0] + &block[i2];
                    let diff_01 = &block[i0] - &block[i2];
                    let diff_01_w = w0 * &diff_01;

                    let sum_23 = &block[i1] + &block[i3];
                    let diff_23 = &block[i1] - &block[i3];
                    let diff_23_w = w1 * &diff_23;

                    let w2 = &twiddles[(j * tw_stride_l1) % twiddles.len()];

                    let final_0 = &sum_01 + &sum_23;
                    let final_1 = &sum_01 - &sum_23;
                    let final_1_w = w2 * &final_1;

                    let final_2 = &diff_01_w + &diff_23_w;
                    let final_3 = &diff_01_w - &diff_23_w;
                    let final_3_w = w2 * &final_3;

                    block[i0] = final_0;
                    block[i1] = final_1_w;
                    block[i2] = final_2;
                    block[i3] = final_3_w;
                }
            }
            layer += 2;
        } else {
            break;
        }
    }

    while layer < log_n {
        let block_size = n >> layer;
        let half_block = block_size >> 1;
        let twiddle_stride = 1 << layer;

        for block_start in (0..n).step_by(block_size) {
            for j in 0..half_block {
                let i0 = block_start + j;
                let i1 = i0 + half_block;

                let tw_idx = j * twiddle_stride;
                let w = &twiddles[tw_idx % twiddles.len()];

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

// =====================================================
// STRUCTURE OF ARRAYS (SoA) FFT
// =====================================================

#[cfg(feature = "alloc")]
pub struct FftMatrix<E: IsField> {
    pub data: Vec<FieldElement<E>>,
    pub width: usize,
    pub height: usize,
}

#[cfg(feature = "alloc")]
impl<E: IsField> FftMatrix<E> {
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

        let mut data = Vec::with_capacity(height * width);
        for poly in polys {
            debug_assert_eq!(poly.len(), width);
            data.extend(poly);
        }

        Self {
            data,
            width,
            height,
        }
    }

    pub fn row_mut(&mut self, row: usize) -> &mut [FieldElement<E>] {
        let start = row * self.width;
        &mut self.data[start..start + self.width]
    }

    pub fn row(&self, row: usize) -> &[FieldElement<E>] {
        let start = row * self.width;
        &self.data[start..start + self.width]
    }

    pub fn to_polynomials(self) -> Vec<Vec<FieldElement<E>>> {
        self.data
            .chunks(self.width)
            .map(|chunk| chunk.to_vec())
            .collect()
    }
}

#[cfg(feature = "alloc")]
pub fn bowers_batch_fft<F, E>(matrix: &mut FftMatrix<E>, twiddles: &[FieldElement<F>])
where
    F: IsFFTField + IsSubFieldOf<E>,
    E: IsField,
{
    if matrix.height == 0 || matrix.width <= 1 {
        return;
    }

    for row in 0..matrix.height {
        let poly = matrix.row_mut(row);
        bowers_fft(poly, twiddles);
        in_place_bit_reverse_permute(poly);
    }
}

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
        bowers_fft_opt(poly, layer_twiddles);
        in_place_bit_reverse_permute(poly);
    }
}

// =====================================================
// TESTS
// =====================================================

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use super::*;
    use crate::fft::cpu::roots_of_unity::get_powers_of_primitive_root;
    use crate::field::fields::u64_goldilocks_field::Goldilocks64Field;
    use crate::field::traits::RootsConfig;
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
    fn test_layer_twiddles_creation() {
        let order = 4u64;
        let layer_twiddles = LayerTwiddles::<F>::new(order);

        assert_eq!(layer_twiddles.layers.len(), 4);
        assert_eq!(layer_twiddles.layers[0].len(), 8);
        assert_eq!(layer_twiddles.layers[1].len(), 4);
        assert_eq!(layer_twiddles.layers[2].len(), 2);
        assert_eq!(layer_twiddles.layers[3].len(), 1);

        for layer in &layer_twiddles.layers {
            assert_eq!(layer[0], FE::one());
        }
    }

    #[test]
    fn test_bowers_fft_opt_small() {
        let input: Vec<FE> = (0..4).map(|i| FE::from(i as u64)).collect();
        let expected = naive_dft(&input);

        let layer_twiddles = LayerTwiddles::<F>::new(2);
        let mut result = input.clone();
        bowers_fft_opt(&mut result, &layer_twiddles);
        in_place_bit_reverse_permute(&mut result);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_bowers_fft_opt_medium() {
        let input: Vec<FE> = (0..16).map(|i| FE::from(i as u64)).collect();
        let expected = naive_dft(&input);

        let layer_twiddles = LayerTwiddles::<F>::new(4);
        let mut result = input.clone();
        bowers_fft_opt(&mut result, &layer_twiddles);
        in_place_bit_reverse_permute(&mut result);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_bowers_fft_opt_large() {
        let input: Vec<FE> = (0..256).map(|i| FE::from(i as u64)).collect();
        let expected = naive_dft(&input);

        let layer_twiddles = LayerTwiddles::<F>::new(8);
        let mut result = input.clone();
        bowers_fft_opt(&mut result, &layer_twiddles);
        in_place_bit_reverse_permute(&mut result);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_bowers_fft_opt_fused() {
        let input: Vec<FE> = (0..256).map(|i| FE::from(i as u64)).collect();
        let expected = naive_dft(&input);

        let layer_twiddles = LayerTwiddles::<F>::new(8);
        let mut result = input.clone();
        bowers_fft_opt_fused(&mut result, &layer_twiddles);
        in_place_bit_reverse_permute(&mut result);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_bowers_fft_opt_matches_legacy() {
        let input: Vec<FE> = (0..64).map(|i| FE::from(i as u64)).collect();

        let twiddles =
            get_powers_of_primitive_root::<F>(6, 32, RootsConfig::Natural).unwrap();
        let layer_twiddles = LayerTwiddles::<F>::new(6);

        let mut result_legacy = input.clone();
        bowers_fft(&mut result_legacy, &twiddles);
        in_place_bit_reverse_permute(&mut result_legacy);

        let mut result_opt = input.clone();
        bowers_fft_opt(&mut result_opt, &layer_twiddles);
        in_place_bit_reverse_permute(&mut result_opt);

        assert_eq!(result_legacy, result_opt);
    }

    #[test]
    fn test_bowers_fft_small() {
        let input: Vec<FE> = (0..4).map(|i| FE::from(i as u64)).collect();
        let expected = naive_dft(&input);

        let twiddles = get_powers_of_primitive_root::<F>(2, 2, RootsConfig::Natural).unwrap();
        let mut result = input.clone();
        bowers_fft(&mut result, &twiddles);
        in_place_bit_reverse_permute(&mut result);

        assert_eq!(result, expected);
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

    #[cfg(feature = "std")]
    mod cached_fft_tests {
        use super::*;
        use crate::fft::cpu::bowers_fft::BowersFft;

        #[test]
        fn test_bowers_fft_cached_basic() {
            let fft = BowersFft::<F>::new();

            let input: Vec<FE> = (0..16).map(|i| FE::from(i as u64)).collect();
            let expected = naive_dft(&input);

            let mut result = input.clone();
            fft.fft(&mut result);

            assert_eq!(result, expected);
        }

        #[test]
        fn test_bowers_fft_cached_reuses_twiddles() {
            let fft = BowersFft::<F>::new();

            // First call - should compute twiddles
            let mut poly1: Vec<FE> = (0..16).map(|i| FE::from(i as u64)).collect();
            fft.fft(&mut poly1);

            // Twiddles should be cached now
            assert!(fft.get_cached_twiddles(4).is_some());

            // Second call - should reuse cached twiddles
            let mut poly2: Vec<FE> = (0..16).map(|i| FE::from(i as u64 + 100)).collect();
            let expected = naive_dft(&poly2);
            fft.fft(&mut poly2);

            assert_eq!(poly2, expected);
        }

        #[test]
        fn test_bowers_fft_cached_different_sizes() {
            let fft = BowersFft::<F>::new();

            // FFT of size 16
            let input16: Vec<FE> = (0..16).map(|i| FE::from(i as u64)).collect();
            let expected16 = naive_dft(&input16);
            let mut result16 = input16.clone();
            fft.fft(&mut result16);
            assert_eq!(result16, expected16);

            // FFT of size 64 - different order, should compute new twiddles
            let input64: Vec<FE> = (0..64).map(|i| FE::from(i as u64)).collect();
            let expected64 = naive_dft(&input64);
            let mut result64 = input64.clone();
            fft.fft(&mut result64);
            assert_eq!(result64, expected64);

            // Both should be cached
            assert!(fft.get_cached_twiddles(4).is_some()); // log2(16) = 4
            assert!(fft.get_cached_twiddles(6).is_some()); // log2(64) = 6
        }

        #[test]
        fn test_bowers_fft_cached_fused() {
            let fft = BowersFft::<F>::new();

            let input: Vec<FE> = (0..256).map(|i| FE::from(i as u64)).collect();
            let expected = naive_dft(&input);

            let mut result = input.clone();
            fft.fft_fused(&mut result);

            assert_eq!(result, expected);
        }

        #[test]
        fn test_bowers_fft_precompute() {
            let fft = BowersFft::<F>::new();

            // Nothing cached initially
            assert!(fft.get_cached_twiddles(4).is_none());

            // Precompute
            fft.precompute(4);

            // Now cached
            assert!(fft.get_cached_twiddles(4).is_some());
        }

        #[test]
        fn test_bowers_fft_clone_shares_cache() {
            let fft1 = BowersFft::<F>::new();

            // Compute twiddles with fft1
            let mut poly: Vec<FE> = (0..16).map(|i| FE::from(i as u64)).collect();
            fft1.fft(&mut poly);

            // Clone shares the same cache (Arc)
            let fft2 = fft1.clone();

            // fft2 should see the cached twiddles
            assert!(fft2.get_cached_twiddles(4).is_some());
        }
    }
}
