// Circle FFT (CFFT) butterfly kernels for Metal.
//
// These kernels implement the butterfly operations for Circle FFT evaluation
// and interpolation. Unlike standard radix-2 FFT, CFFT uses layered twiddles
// (layer i has 2^i elements for evaluation) rather than a flat bit-reversed array.
//
// The CFFT butterfly is dispatched once per layer from the host (Rust) side,
// which passes the layer index and twiddle offset as parameters.

#pragma once

#include <metal_stdlib>

/// CFFT evaluation butterfly operation.
///
/// For layer i with half_chunk = 2^i:
///   temp = low * twiddle
///   low  = hi - temp
///   hi   = hi + temp
///
/// Matches circle/cfft.rs cfft() lines 64-84.
///
/// Parameters:
/// - input: Array of field elements (modified in-place)
/// - twiddles: Flattened twiddle factors (all layers concatenated)
/// - half_chunk_shift: log2(half_chunk_size) for this layer (= layer index i)
/// - tw_offset: Offset into the flattened twiddles array for this layer
template<typename Fp>
[[kernel]] void cfft_butterfly(
    device Fp* input               [[ buffer(0) ]],
    constant Fp* twiddles          [[ buffer(1) ]],
    constant uint32_t& half_chunk_shift [[ buffer(2) ]],
    constant uint32_t& tw_offset   [[ buffer(3) ]],
    uint32_t thread_pos            [[ thread_position_in_grid ]]
)
{
    uint32_t half_chunk = 1u << half_chunk_shift;

    // Map thread to butterfly indices
    uint32_t chunk_idx = thread_pos >> half_chunk_shift;
    uint32_t j = thread_pos & (half_chunk - 1u);

    uint32_t hi_idx = chunk_idx * (half_chunk << 1u) + j;
    uint32_t low_idx = hi_idx + half_chunk;

    Fp tw = twiddles[tw_offset + j];
    Fp hi = input[hi_idx];
    Fp low = input[low_idx];

    Fp temp = low * tw;
    input[low_idx] = hi - temp;
    input[hi_idx] = hi + temp;
}

/// ICFFT interpolation butterfly operation.
///
/// For each layer with the given half_chunk size:
///   temp = hi + low
///   low  = (hi - low) * twiddle
///   hi   = temp
///
/// Matches circle/cfft.rs icfft() lines 122-142.
///
/// Parameters:
/// - input: Array of field elements (modified in-place)
/// - twiddles: Flattened twiddle factors (all layers concatenated)
/// - half_chunk_shift: log2(half_chunk_size) for this layer
/// - tw_offset: Offset into the flattened twiddles array for this layer
template<typename Fp>
[[kernel]] void icfft_butterfly(
    device Fp* input               [[ buffer(0) ]],
    constant Fp* twiddles          [[ buffer(1) ]],
    constant uint32_t& half_chunk_shift [[ buffer(2) ]],
    constant uint32_t& tw_offset   [[ buffer(3) ]],
    uint32_t thread_pos            [[ thread_position_in_grid ]]
)
{
    uint32_t half_chunk = 1u << half_chunk_shift;

    // Map thread to butterfly indices
    uint32_t chunk_idx = thread_pos >> half_chunk_shift;
    uint32_t j = thread_pos & (half_chunk - 1u);

    uint32_t hi_idx = chunk_idx * (half_chunk << 1u) + j;
    uint32_t low_idx = hi_idx + half_chunk;

    Fp tw = twiddles[tw_offset + j];
    Fp hi = input[hi_idx];
    Fp low = input[low_idx];

    Fp temp = hi + low;
    input[low_idx] = (hi - low) * tw;
    input[hi_idx] = temp;
}
