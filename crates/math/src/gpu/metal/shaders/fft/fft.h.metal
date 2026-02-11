// Radix-2 Decimation-In-Time (DIT) FFT butterfly kernel for Metal.
//
// Implements the core butterfly operation of the Cooley-Tukey FFT algorithm.
// Each kernel invocation processes one butterfly operation for a given stage.

#pragma once

#include <metal_stdlib>

/// Radix-2 DIT butterfly operation.
///
/// Performs the butterfly computation for one stage of the FFT:
///   a' = a + w*b
///   b' = a - w*b
///
/// where w is the appropriate twiddle factor for this butterfly's group.
///
/// Parameters:
/// - input: Array of field elements (modified in-place)
/// - twiddles: Pre-computed twiddle factors in bit-reversed order
/// - stage: Current FFT stage (0 to log2(n)-1)
/// - thread_count: Total number of butterflies (n/2)
/// - thread_pos: This thread's butterfly index
///
/// Memory layout:
/// For stage s, butterflies are grouped. Each group has 2^s elements.
/// Within a group, pairs (i, i + group_size/2) are processed together.
template<typename Fp>
[[kernel]] void radix2_dit_butterfly(
    device Fp* input          [[ buffer(0) ]],
    constant Fp* twiddles     [[ buffer(1) ]],
    constant uint32_t& stage  [[ buffer(2) ]],
    uint32_t thread_count     [[ threads_per_grid ]],
    uint32_t thread_pos       [[ thread_position_in_grid ]]
)
{
    // Calculate group size and position
    uint32_t half_group_size = thread_count >> stage; // thread_count / group_count
    uint32_t group = thread_pos >> metal::ctz(half_group_size); // thread_pos / half_group_size

    uint32_t pos_in_group = thread_pos & (half_group_size - 1); // thread_pos % half_group_size
    uint32_t i = thread_pos * 2 - pos_in_group; // multiply quotient by 2

    // Load twiddle factor and input values
    Fp w = twiddles[group];
    Fp a = input[i];
    Fp b = input[i + half_group_size];

    // Butterfly computation
    Fp res_1 = a + w*b;
    Fp res_2 = a - w*b;

    // Store results
    input[i]                    = res_1; // --\/--
    input[i + half_group_size]  = res_2; // --/\--
}

// ===========================================================================
// Threadgroup-cached variant: twiddles are cooperatively loaded into shared
// memory before the butterfly, reducing global memory traffic.
// ===========================================================================

/// Radix-2 DIT butterfly with threadgroup-cached twiddles.
///
/// Same computation as radix2_dit_butterfly but twiddles[0..tw_count]
/// are loaded into threadgroup memory cooperatively before use.
/// Beneficial when multiple threads in the same threadgroup access
/// the same twiddle values (which happens at early FFT stages where
/// the number of groups is small).
template<typename Fp>
[[kernel]] void radix2_dit_butterfly_tg(
    device Fp* input          [[ buffer(0) ]],
    constant Fp* twiddles     [[ buffer(1) ]],
    constant uint32_t& stage  [[ buffer(2) ]],
    constant uint32_t& tw_count [[ buffer(3) ]],
    uint32_t thread_count     [[ threads_per_grid ]],
    uint32_t thread_pos       [[ thread_position_in_grid ]],
    uint32_t tg_pos           [[ thread_position_in_threadgroup ]],
    uint32_t tg_size          [[ threads_per_threadgroup ]],
    threadgroup Fp* shared_tw [[ threadgroup(0) ]]
)
{
    // Cooperatively load twiddles into threadgroup memory
    for (uint32_t i = tg_pos; i < tw_count; i += tg_size) {
        shared_tw[i] = twiddles[i];
    }
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    uint32_t half_group_size = thread_count >> stage;
    uint32_t group = thread_pos >> metal::ctz(half_group_size);
    uint32_t pos_in_group = thread_pos & (half_group_size - 1);
    uint32_t i = thread_pos * 2 - pos_in_group;

    Fp w = shared_tw[group];
    Fp a = input[i];
    Fp b = input[i + half_group_size];

    input[i]                    = a + w*b;
    input[i + half_group_size]  = a - w*b;
}

// ===========================================================================
// Fused multi-stage kernel: processes FFT_FUSED_STAGES consecutive butterfly
// stages (the LAST K stages of the FFT) in a single dispatch using
// threadgroup memory for the data block.
// ===========================================================================

/// Number of butterfly stages fused into a single kernel dispatch.
constant constexpr uint32_t FFT_FUSED_STAGES = 4;

/// Fused radix-2 DIT butterfly: processes the last FFT_FUSED_STAGES stages
/// in shared memory.
///
/// Each threadgroup handles a contiguous block of (1 << FFT_FUSED_STAGES)
/// elements. At the fused stages, the butterfly stride (half_group_size)
/// is small enough that all pairs fall within the same block.
///
/// Parameters:
/// - input: Array of field elements (modified in-place)
/// - twiddles: Pre-computed twiddle factors in bit-reversed order
/// - start_stage: The first stage to process (fuses start_stage .. start_stage + FFT_FUSED_STAGES)
template<typename Fp>
[[kernel]] void radix2_dit_butterfly_fused(
    device Fp* input              [[ buffer(0) ]],
    constant Fp* twiddles         [[ buffer(1) ]],
    constant uint32_t& start_stage [[ buffer(2) ]],
    uint32_t tg_id                [[ threadgroup_position_in_grid ]],
    uint32_t tg_pos               [[ thread_position_in_threadgroup ]],
    uint32_t tg_size              [[ threads_per_threadgroup ]],
    threadgroup Fp* shared_data   [[ threadgroup(0) ]]
)
{
    const uint32_t BLOCK_SIZE = 1u << FFT_FUSED_STAGES; // 16
    const uint32_t HALF_BLOCK = BLOCK_SIZE >> 1;          // 8

    uint32_t block_start = tg_id * BLOCK_SIZE;

    // Load block from global to threadgroup memory
    for (uint32_t i = tg_pos; i < BLOCK_SIZE; i += tg_size) {
        shared_data[i] = input[block_start + i];
    }
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    // Process FFT_FUSED_STAGES butterfly stages locally.
    // For sub-stage k, the local half_group_size decreases:
    //   k=0: half_group_size = HALF_BLOCK (= 8)
    //   k=1: half_group_size = 4
    //   k=K-1: half_group_size = 1
    for (uint32_t k = 0; k < FFT_FUSED_STAGES; k++) {
        uint32_t local_hgs_shift = (FFT_FUSED_STAGES - 1) - k;
        uint32_t local_hgs = 1u << local_hgs_shift;

        for (uint32_t t = tg_pos; t < HALF_BLOCK; t += tg_size) {
            uint32_t local_chunk = t >> local_hgs_shift;
            uint32_t local_pos = t & (local_hgs - 1u);

            uint32_t local_i = local_chunk * (local_hgs << 1u) + local_pos;
            uint32_t local_j = local_i + local_hgs;

            // Global group index for twiddle lookup:
            // At global stage (start_stage + k), the block at tg_id contains
            // 2^k chunks, so global_group = tg_id * 2^k + local_chunk
            uint32_t global_group = (tg_id << k) + local_chunk;

            Fp w = twiddles[global_group];
            Fp a = shared_data[local_i];
            Fp b = shared_data[local_j];

            shared_data[local_i] = a + w * b;
            shared_data[local_j] = a - w * b;
        }
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    }

    // Write back from threadgroup to global memory
    for (uint32_t i = tg_pos; i < BLOCK_SIZE; i += tg_size) {
        input[block_start + i] = shared_data[i];
    }
}
