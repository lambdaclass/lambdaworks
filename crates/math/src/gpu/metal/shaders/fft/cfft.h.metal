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

// ===========================================================================
// Threadgroup-cached variants: twiddles are cooperatively loaded into shared
// memory before the butterfly, reducing global memory traffic.
// ===========================================================================

/// CFFT evaluation butterfly with threadgroup-cached twiddles.
///
/// Same computation as cfft_butterfly but twiddles are first loaded into
/// threadgroup (shared) memory cooperatively by all threads in the group,
/// then read from there. This is beneficial when many threads in the same
/// threadgroup access the same twiddle values.
template<typename Fp>
[[kernel]] void cfft_butterfly_tg(
    device Fp* input               [[ buffer(0) ]],
    constant Fp* twiddles          [[ buffer(1) ]],
    constant uint32_t& half_chunk_shift [[ buffer(2) ]],
    constant uint32_t& tw_offset   [[ buffer(3) ]],
    constant uint32_t& tw_count    [[ buffer(4) ]],
    uint32_t thread_pos            [[ thread_position_in_grid ]],
    uint32_t tg_pos                [[ thread_position_in_threadgroup ]],
    uint32_t tg_size               [[ threads_per_threadgroup ]],
    threadgroup Fp* shared_tw      [[ threadgroup(0) ]]
)
{
    // Cooperatively load twiddles into threadgroup memory
    for (uint32_t i = tg_pos; i < tw_count; i += tg_size) {
        shared_tw[i] = twiddles[tw_offset + i];
    }
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    uint32_t half_chunk = 1u << half_chunk_shift;

    uint32_t chunk_idx = thread_pos >> half_chunk_shift;
    uint32_t j = thread_pos & (half_chunk - 1u);

    uint32_t hi_idx = chunk_idx * (half_chunk << 1u) + j;
    uint32_t low_idx = hi_idx + half_chunk;

    Fp tw = shared_tw[j];
    Fp hi = input[hi_idx];
    Fp low = input[low_idx];

    Fp temp = low * tw;
    input[low_idx] = hi - temp;
    input[hi_idx] = hi + temp;
}

/// ICFFT interpolation butterfly with threadgroup-cached twiddles.
template<typename Fp>
[[kernel]] void icfft_butterfly_tg(
    device Fp* input               [[ buffer(0) ]],
    constant Fp* twiddles          [[ buffer(1) ]],
    constant uint32_t& half_chunk_shift [[ buffer(2) ]],
    constant uint32_t& tw_offset   [[ buffer(3) ]],
    constant uint32_t& tw_count    [[ buffer(4) ]],
    uint32_t thread_pos            [[ thread_position_in_grid ]],
    uint32_t tg_pos                [[ thread_position_in_threadgroup ]],
    uint32_t tg_size               [[ threads_per_threadgroup ]],
    threadgroup Fp* shared_tw      [[ threadgroup(0) ]]
)
{
    // Cooperatively load twiddles into threadgroup memory
    for (uint32_t i = tg_pos; i < tw_count; i += tg_size) {
        shared_tw[i] = twiddles[tw_offset + i];
    }
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    uint32_t half_chunk = 1u << half_chunk_shift;

    uint32_t chunk_idx = thread_pos >> half_chunk_shift;
    uint32_t j = thread_pos & (half_chunk - 1u);

    uint32_t hi_idx = chunk_idx * (half_chunk << 1u) + j;
    uint32_t low_idx = hi_idx + half_chunk;

    Fp tw = shared_tw[j];
    Fp hi = input[hi_idx];
    Fp low = input[low_idx];

    Fp temp = hi + low;
    input[low_idx] = (hi - low) * tw;
    input[hi_idx] = temp;
}

// ===========================================================================
// Fused multi-stage kernel: processes FUSED_STAGES consecutive butterfly
// stages in a single dispatch, using threadgroup memory for the data block.
// This eliminates (FUSED_STAGES - 1) global memory round-trips.
// ===========================================================================

/// Number of butterfly stages fused into a single kernel dispatch.
/// Each threadgroup processes a block of (1 << CFFT_FUSED_STAGES) elements.
/// Block size = 16 elements = 64 bytes for Mersenne31, well within 32KB limit.
constant constexpr uint32_t CFFT_FUSED_STAGES = 4;

/// Fused CFFT evaluation: performs CFFT_FUSED_STAGES butterfly stages in shared memory.
///
/// Each threadgroup handles a contiguous block of (1 << CFFT_FUSED_STAGES) elements.
/// All twiddles for the fused stages are also loaded into threadgroup memory.
///
/// Parameters:
/// - input: Array of field elements (modified in-place)
/// - twiddles: Flattened twiddle factors (all layers concatenated)
/// - start_layer: The first layer index to process (layers start_layer .. start_layer + FUSED_STAGES)
/// - tw_offsets: Array of CFFT_FUSED_STAGES offsets into the flattened twiddles array
template<typename Fp>
[[kernel]] void cfft_butterfly_fused(
    device Fp* input               [[ buffer(0) ]],
    constant Fp* twiddles          [[ buffer(1) ]],
    constant uint32_t& start_layer [[ buffer(2) ]],
    constant uint32_t* tw_offsets  [[ buffer(3) ]],
    uint32_t tg_id                 [[ threadgroup_position_in_grid ]],
    uint32_t tg_pos                [[ thread_position_in_threadgroup ]],
    uint32_t tg_size               [[ threads_per_threadgroup ]],
    threadgroup Fp* shared_data    [[ threadgroup(0) ]]
)
{
    const uint32_t BLOCK_SIZE = 1u << CFFT_FUSED_STAGES; // 16
    const uint32_t HALF_BLOCK = BLOCK_SIZE >> 1;          // 8

    // Each threadgroup processes one block of BLOCK_SIZE elements
    uint32_t block_start = tg_id * BLOCK_SIZE;

    // Load block from global to threadgroup memory
    for (uint32_t i = tg_pos; i < BLOCK_SIZE; i += tg_size) {
        shared_data[i] = input[block_start + i];
    }
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    // Process CFFT_FUSED_STAGES butterfly layers locally
    for (uint32_t s = 0; s < CFFT_FUSED_STAGES; s++) {
        uint32_t layer = start_layer + s;
        uint32_t half_chunk = 1u << s; // within the block: layer 0 => half_chunk=1, layer 1 => 2, etc.

        // Each thread processes one butterfly
        for (uint32_t t = tg_pos; t < HALF_BLOCK; t += tg_size) {
            uint32_t chunk_idx = t >> s;
            uint32_t j = t & (half_chunk - 1u);

            uint32_t hi_local = chunk_idx * (half_chunk << 1u) + j;
            uint32_t low_local = hi_local + half_chunk;

            // Global twiddle index: for evaluation, layer i has 2^i twiddles
            // Within the block, the twiddle index wraps: j maps to the correct one.
            // The block is positioned at block_start in the global array.
            // For layer i, the global chunk_idx = (block_start / (2*half_chunk_global)) + local_chunk_idx
            // and the twiddle index = j (local position within half_chunk).
            // BUT half_chunk for this layer globally is 2^layer, and j < 2^layer.
            // Within our block of 2^FUSED_STAGES elements, j < 2^s < 2^layer always holds.
            //
            // The twiddle to use: twiddles[tw_offsets[s] + (block_start_in_global_chunk * 0) + j]
            // Actually: for CFFT eval, the twiddle for position j within ANY chunk of layer i
            // is just twiddles[tw_offset_i + j]. So we only need j < half_chunk.
            Fp tw = twiddles[tw_offsets[s] + j];

            Fp hi = shared_data[hi_local];
            Fp low = shared_data[low_local];

            Fp temp = low * tw;
            shared_data[low_local] = hi - temp;
            shared_data[hi_local] = hi + temp;
        }
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    }

    // Write back from threadgroup to global memory
    for (uint32_t i = tg_pos; i < BLOCK_SIZE; i += tg_size) {
        input[block_start + i] = shared_data[i];
    }
}

/// Fused ICFFT interpolation: performs CFFT_FUSED_STAGES butterfly stages in shared memory.
///
/// Same structure as cfft_butterfly_fused but uses the inverse butterfly:
///   temp = hi + low
///   low  = (hi - low) * twiddle
///   hi   = temp
///
/// For ICFFT, the layers go from large half_chunk to small, so the fused stages
/// are the LAST stages (smallest half_chunks). We process them in order
/// start_layer, start_layer+1, ... where half_chunk decreases.
///
/// Parameters:
/// - input: Array of field elements (modified in-place)
/// - twiddles: Flattened interpolation twiddle factors
/// - start_layer_shift: half_chunk_shift for the first layer to fuse
///   (subsequent layers have shift-1, shift-2, etc.)
/// - tw_offsets: Array of CFFT_FUSED_STAGES offsets into the flattened twiddles array
template<typename Fp>
[[kernel]] void icfft_butterfly_fused(
    device Fp* input                     [[ buffer(0) ]],
    constant Fp* twiddles                [[ buffer(1) ]],
    constant uint32_t& start_layer_shift [[ buffer(2) ]],
    constant uint32_t* tw_offsets        [[ buffer(3) ]],
    uint32_t tg_id                       [[ threadgroup_position_in_grid ]],
    uint32_t tg_pos                      [[ thread_position_in_threadgroup ]],
    uint32_t tg_size                     [[ threads_per_threadgroup ]],
    threadgroup Fp* shared_data          [[ threadgroup(0) ]]
)
{
    const uint32_t BLOCK_SIZE = 1u << CFFT_FUSED_STAGES; // 16
    const uint32_t HALF_BLOCK = BLOCK_SIZE >> 1;          // 8

    uint32_t block_start = tg_id * BLOCK_SIZE;

    // Load block from global to threadgroup memory
    for (uint32_t i = tg_pos; i < BLOCK_SIZE; i += tg_size) {
        shared_data[i] = input[block_start + i];
    }
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    // Process CFFT_FUSED_STAGES inverse butterfly layers
    // The half_chunk for each fused sub-stage within the block:
    // sub-stage 0: half_chunk = HALF_BLOCK (= 8)
    // sub-stage 1: half_chunk = 4
    // sub-stage k: half_chunk = HALF_BLOCK >> k
    for (uint32_t s = 0; s < CFFT_FUSED_STAGES; s++) {
        uint32_t local_half_chunk_shift = (CFFT_FUSED_STAGES - 1) - s;
        uint32_t local_half_chunk = 1u << local_half_chunk_shift;

        for (uint32_t t = tg_pos; t < HALF_BLOCK; t += tg_size) {
            uint32_t chunk_idx = t >> local_half_chunk_shift;
            uint32_t j = t & (local_half_chunk - 1u);

            uint32_t hi_local = chunk_idx * (local_half_chunk << 1u) + j;
            uint32_t low_local = hi_local + local_half_chunk;

            Fp tw = twiddles[tw_offsets[s] + j];

            Fp hi = shared_data[hi_local];
            Fp low = shared_data[low_local];

            Fp temp = hi + low;
            shared_data[low_local] = (hi - low) * tw;
            shared_data[hi_local] = temp;
        }
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    }

    // Write back from threadgroup to global memory
    for (uint32_t i = tg_pos; i < BLOCK_SIZE; i += tg_size) {
        input[block_start + i] = shared_data[i];
    }
}
