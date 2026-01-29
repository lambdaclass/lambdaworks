//! Bowers FFT Implementation with Structure of Arrays (SoA) optimization
//!
//! This module implements the Bowers G network FFT algorithm, which provides
//! improved twiddle factor access patterns compared to the standard Cooley-Tukey FFT.
//!
//! Key optimizations:
//! - Bowers G network: Improved memory access pattern for twiddle factors
//! - Structure of Arrays (SoA): Better cache utilization for batch processing
//! - Multi-layer butterfly fusion: Process 2-3 layers at once to reduce memory traffic
//!
//! Based on analysis of Plonky3's implementation and academic literature on FFT optimization.

use crate::fft::cpu::bit_reversing::in_place_bit_reverse_permute;
use crate::field::{
    element::FieldElement,
    traits::{IsFFTField, IsField, IsSubFieldOf},
};

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

// =====================================================
// BOWERS G NETWORK FFT
// =====================================================

/// In-Place Bowers G Network Radix-2 FFT
///
/// The Bowers G network is a decimation-in-frequency (DIF) FFT that processes
/// butterflies in a specific order to improve twiddle factor locality.
///
/// Unlike standard Cooley-Tukey which accesses twiddles at varying strides,
/// the Bowers network accesses twiddles sequentially within each stage.
///
/// The output is in bit-reversed order and needs to be permuted for natural order.
///
/// # Arguments
/// * `input` - Mutable slice of field elements (length must be power of 2)
/// * `twiddles` - Twiddle factors in natural order (w^0, w^1, w^2, ...)
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

    // Bowers G: DIF style - start with large blocks, end with small
    // This is the "forward" direction of Bowers G network
    for layer in 0..log_n {
        let block_size = n >> layer;
        let half_block = block_size >> 1;
        let twiddle_stride = 1 << layer;

        for block_start in (0..n).step_by(block_size) {
            for j in 0..half_block {
                let i0 = block_start + j;
                let i1 = i0 + half_block;

                // Twiddle index for Bowers G: sequential access within stage
                let tw_idx = j * twiddle_stride;
                let w = if tw_idx < twiddles.len() {
                    &twiddles[tw_idx]
                } else {
                    &twiddles[0] // Safety fallback
                };

                // DIF butterfly: (a, b) -> (a + b, (a - b) * w)
                let sum = &input[i0] + &input[i1];
                let diff = &input[i0] - &input[i1];
                let diff_w = w * &diff;

                input[i0] = sum;
                input[i1] = diff_w;
            }
        }
    }
}

/// In-Place Bowers G Inverse Network Radix-2 IFFT
///
/// The inverse of Bowers G network, which is a decimation-in-time (DIT) FFT
/// that starts with small blocks and ends with large ones.
///
/// The input should be in bit-reversed order, and output will be in natural order.
///
/// # Arguments
/// * `input` - Mutable slice of field elements (length must be power of 2)
/// * `twiddles` - Twiddle factors (inverse roots) in natural order
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

    // Bowers G^T: DIT style - start with small blocks, end with large
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

                // DIT butterfly: (a, b) -> (a + b*w, a - b*w)
                let bw = w * &input[i1];
                let sum = &input[i0] + &bw;
                let diff = &input[i0] - &bw;

                input[i0] = sum;
                input[i1] = diff;
            }
        }
    }
}

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
            debug_assert_eq!(poly.len(), width, "All polynomials must have same length");
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

/// Batch FFT using Bowers algorithm with SoA layout
///
/// Performs FFT on multiple polynomials stored in a matrix.
/// This is more efficient than processing polynomials one at a time
/// because it allows for better cache utilization and potential parallelization.
///
/// # Arguments
/// * `matrix` - FFT matrix containing polynomials to transform
/// * `twiddles` - Twiddle factors in natural order
#[cfg(feature = "alloc")]
pub fn bowers_batch_fft<F, E>(matrix: &mut FftMatrix<E>, twiddles: &[FieldElement<F>])
where
    F: IsFFTField + IsSubFieldOf<E>,
    E: IsField,
{
    if matrix.height == 0 || matrix.width <= 1 {
        return;
    }

    // Process each polynomial using Bowers FFT
    for row in 0..matrix.height {
        let poly = matrix.row_mut(row);
        bowers_fft(poly, twiddles);
        in_place_bit_reverse_permute(poly);
    }
}

// =====================================================
// MULTI-LAYER BUTTERFLY FUSION
// =====================================================

/// Fused 2-layer butterfly operation
///
/// Processes two consecutive FFT layers in a single pass, reducing
/// memory traffic by keeping intermediate values in registers.
///
/// This is particularly effective for small block sizes where the
/// data fits in L1 cache.
#[inline(always)]
fn butterfly_2_layers<F, E>(
    data: &mut [FieldElement<E>],
    twiddles: &[FieldElement<F>],
    block_size: usize,
    tw_stride_layer0: usize,
    tw_stride_layer1: usize,
) where
    F: IsFFTField + IsSubFieldOf<E>,
    E: IsField,
{
    let quarter = block_size >> 2;

    for j in 0..quarter {
        let i0 = j;
        let i1 = j + quarter;
        let i2 = j + 2 * quarter;
        let i3 = j + 3 * quarter;

        // Layer 0 twiddle indices
        let tw0_idx = j * tw_stride_layer0;
        let tw1_idx = (j + quarter) * tw_stride_layer0;

        let w0 = &twiddles[tw0_idx % twiddles.len()];
        let w1 = &twiddles[tw1_idx % twiddles.len()];

        // Layer 0 butterflies
        let sum_01 = &data[i0] + &data[i2];
        let diff_01 = &data[i0] - &data[i2];
        let diff_01_w = w0 * &diff_01;

        let sum_23 = &data[i1] + &data[i3];
        let diff_23 = &data[i1] - &data[i3];
        let diff_23_w = w1 * &diff_23;

        // Layer 1 twiddle index
        let tw2_idx = j * tw_stride_layer1;
        let w2 = &twiddles[tw2_idx % twiddles.len()];

        // Layer 1 butterflies
        let final_0 = &sum_01 + &sum_23;
        let final_1 = &sum_01 - &sum_23;
        let final_1_w = w2 * &final_1;

        let final_2 = &diff_01_w + &diff_23_w;
        let final_3 = &diff_01_w - &diff_23_w;
        let final_3_w = w2 * &final_3;

        data[i0] = final_0;
        data[i1] = final_1_w;
        data[i2] = final_2;
        data[i3] = final_3_w;
    }
}

/// Optimized Bowers FFT with 2-layer fusion
///
/// Uses fused butterflies for small block sizes to improve performance.
/// Falls back to single-layer processing for the final stages.
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
        // For very small sizes, use standard butterfly
        bowers_fft(input, twiddles);
        return;
    }

    let log_n = n.trailing_zeros() as usize;
    let mut layer = 0;

    // Process pairs of layers when possible
    while layer + 1 < log_n {
        let block_size = n >> layer;

        if block_size >= 4 {
            let tw_stride_layer0 = 1 << layer;
            let tw_stride_layer1 = 1 << (layer + 1);

            for block_start in (0..n).step_by(block_size) {
                butterfly_2_layers(
                    &mut input[block_start..block_start + block_size],
                    twiddles,
                    block_size,
                    tw_stride_layer0,
                    tw_stride_layer1,
                );
            }
            layer += 2;
        } else {
            break;
        }
    }

    // Process remaining single layers
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
// PARALLEL BOWERS FFT
// =====================================================

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Parallel Bowers FFT for large inputs
///
/// Parallelizes the outer loops of the FFT when processing large inputs.
/// Uses a threshold to avoid overhead for small inputs.
#[cfg(all(feature = "alloc", feature = "parallel"))]
pub fn bowers_fft_parallel<F, E>(input: &mut [FieldElement<E>], twiddles: &[FieldElement<F>])
where
    F: IsFFTField + IsSubFieldOf<E>,
    E: IsField + Send + Sync,
    FieldElement<F>: Send + Sync,
    FieldElement<E>: Send + Sync,
{
    const PARALLEL_THRESHOLD: usize = 64; // Minimum blocks for parallelization

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
        let num_blocks = n / block_size;

        if num_blocks >= PARALLEL_THRESHOLD {
            // Parallel processing for many blocks
            let chunks: Vec<&mut [FieldElement<E>]> = input.chunks_mut(block_size).collect();

            chunks.into_par_iter().for_each(|block| {
                for j in 0..half_block {
                    let tw_idx = j * twiddle_stride;
                    let w = &twiddles[tw_idx % twiddles.len()];

                    let sum = &block[j] + &block[j + half_block];
                    let diff = &block[j] - &block[j + half_block];
                    let diff_w = w * &diff;

                    block[j] = sum;
                    block[j + half_block] = diff_w;
                }
            });
        } else {
            // Sequential processing for few blocks
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
        }
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

    type F = Goldilocks64Field;
    type FE = FieldElement<F>;

    fn naive_dft(input: &[FE]) -> Vec<FE> {
        let n = input.len();
        let root = F::get_primitive_root_of_unity(n.trailing_zeros() as u64).unwrap();
        let mut result = vec![FE::zero(); n];

        for k in 0..n {
            for j in 0..n {
                result[k] = result[k] + input[j] * root.pow((j * k) as u64);
            }
        }

        result
    }

    #[test]
    fn test_bowers_fft_small() {
        let input: Vec<FE> = (0..4).map(|i| FE::from(i as u64)).collect();
        let expected = naive_dft(&input);

        let order = 2u64;
        let twiddles = get_powers_of_primitive_root::<F>(order, 2, RootsConfig::Natural).unwrap();

        let mut result = input.clone();
        bowers_fft(&mut result, &twiddles);
        in_place_bit_reverse_permute(&mut result);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_bowers_fft_medium() {
        let input: Vec<FE> = (0..16).map(|i| FE::from(i as u64)).collect();
        let expected = naive_dft(&input);

        let order = 4u64;
        let twiddles = get_powers_of_primitive_root::<F>(order, 8, RootsConfig::Natural).unwrap();

        let mut result = input.clone();
        bowers_fft(&mut result, &twiddles);
        in_place_bit_reverse_permute(&mut result);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_bowers_fft_fused_small() {
        let input: Vec<FE> = (0..16).map(|i| FE::from(i as u64)).collect();
        let expected = naive_dft(&input);

        let order = 4u64;
        let twiddles = get_powers_of_primitive_root::<F>(order, 8, RootsConfig::Natural).unwrap();

        let mut result = input.clone();
        bowers_fft_fused(&mut result, &twiddles);
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
}
