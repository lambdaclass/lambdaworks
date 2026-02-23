// Radix-2 DIT FFT butterfly kernel for extension fields with base field twiddles.
//
// This kernel supports FFT operations where:
// - Input coefficients are in an extension field (Fp2, Fp3, etc.)
// - Twiddle factors are in the base field
//
// The butterfly operation is:
//   a' = a + w*b    (where a, b are extension elements, w is base field)
//   b' = a - w*b
//
// Uses scalar multiplication (extension * base -> extension) for efficiency.

#pragma once

#include <metal_stdlib>

/// Radix-2 DIT butterfly for extension field elements with base field twiddles.
///
/// Template parameters:
/// - FpExt: Extension field element type (e.g., Fp2Goldilocks, Fp3Goldilocks)
/// - FpBase: Base field element type (e.g., Fp64Goldilocks)
///
/// The FpExt type must provide:
/// - operator+(FpExt) -> FpExt
/// - operator-(FpExt) -> FpExt
/// - scalar_mul(FpBase) -> FpExt
template<typename FpExt, typename FpBase>
[[kernel]] void radix2_dit_butterfly_ext(
    device FpExt* input          [[ buffer(0) ]],
    constant FpBase* twiddles    [[ buffer(1) ]],
    constant uint32_t& stage     [[ buffer(2) ]],
    uint32_t thread_count        [[ threads_per_grid ]],
    uint32_t thread_pos          [[ thread_position_in_grid ]]
)
{
    // Calculate group size and position
    uint32_t half_group_size = thread_count >> stage;
    uint32_t group = thread_pos >> metal::ctz(half_group_size);

    uint32_t pos_in_group = thread_pos & (half_group_size - 1);
    uint32_t i = thread_pos * 2 - pos_in_group;

    // Load base field twiddle and extension field input values
    FpBase w = twiddles[group];
    FpExt a = input[i];
    FpExt b = input[i + half_group_size];

    // Butterfly computation using scalar multiplication
    FpExt wb = b.scalar_mul(w);  // extension * base -> extension
    FpExt res_1 = a + wb;
    FpExt res_2 = a - wb;

    // Store results
    input[i] = res_1;
    input[i + half_group_size] = res_2;
}

// ===========================================================================
// Threadgroup-cached variant for extension fields: base-field twiddles are
// cooperatively loaded into shared memory before the butterfly.
// ===========================================================================

/// Radix-2 DIT butterfly for extension fields with threadgroup-cached twiddles.
///
/// Twiddles are in the base field and loaded into threadgroup memory.
/// Input/output elements are in the extension field.
template<typename FpExt, typename FpBase>
[[kernel]] void radix2_dit_butterfly_tg_ext(
    device FpExt* input          [[ buffer(0) ]],
    constant FpBase* twiddles    [[ buffer(1) ]],
    constant uint32_t& stage     [[ buffer(2) ]],
    constant uint32_t& tw_count  [[ buffer(3) ]],
    uint32_t thread_count        [[ threads_per_grid ]],
    uint32_t thread_pos          [[ thread_position_in_grid ]],
    uint32_t tg_pos              [[ thread_position_in_threadgroup ]],
    uint32_t tg_size             [[ threads_per_threadgroup ]],
    threadgroup FpBase* shared_tw [[ threadgroup(0) ]]
)
{
    // Cooperatively load base-field twiddles into threadgroup memory
    for (uint32_t i = tg_pos; i < tw_count; i += tg_size) {
        shared_tw[i] = twiddles[i];
    }
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    uint32_t half_group_size = thread_count >> stage;
    uint32_t group = thread_pos >> metal::ctz(half_group_size);
    uint32_t pos_in_group = thread_pos & (half_group_size - 1);
    uint32_t i = thread_pos * 2 - pos_in_group;

    FpBase w = shared_tw[group];
    FpExt a = input[i];
    FpExt b = input[i + half_group_size];

    FpExt wb = b.scalar_mul(w);
    input[i]                    = a + wb;
    input[i + half_group_size]  = a - wb;
}

// ===========================================================================
// Fused multi-stage kernel for extension fields: processes num_stages
// consecutive butterfly stages (the LAST K stages) in a single dispatch
// using threadgroup memory for the extension field data block.
// Twiddles remain in the base field (read from global memory).
// ===========================================================================

/// Fused radix-2 DIT butterfly for extension fields.
///
/// Each threadgroup handles a contiguous block of (1 << num_stages)
/// extension field elements in shared memory. Twiddle factors are
/// base-field elements read from global memory.
template<typename FpExt, typename FpBase>
[[kernel]] void radix2_dit_butterfly_fused_ext(
    device FpExt* input              [[ buffer(0) ]],
    constant FpBase* twiddles        [[ buffer(1) ]],
    constant uint32_t& start_stage   [[ buffer(2) ]],
    constant uint32_t& num_stages    [[ buffer(3) ]],
    uint32_t tg_id                   [[ threadgroup_position_in_grid ]],
    uint32_t tg_pos                  [[ thread_position_in_threadgroup ]],
    uint32_t tg_size                 [[ threads_per_threadgroup ]],
    threadgroup FpExt* shared_data   [[ threadgroup(0) ]]
)
{
    const uint32_t BLOCK_SIZE = 1u << num_stages;
    const uint32_t HALF_BLOCK = BLOCK_SIZE >> 1;

    uint32_t block_start = tg_id * BLOCK_SIZE;

    // Load block from global to threadgroup memory
    for (uint32_t i = tg_pos; i < BLOCK_SIZE; i += tg_size) {
        shared_data[i] = input[block_start + i];
    }
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    // Process num_stages butterfly stages locally
    for (uint32_t k = 0; k < num_stages; k++) {
        uint32_t local_hgs_shift = (num_stages - 1) - k;
        uint32_t local_hgs = 1u << local_hgs_shift;

        for (uint32_t t = tg_pos; t < HALF_BLOCK; t += tg_size) {
            uint32_t local_chunk = t >> local_hgs_shift;
            uint32_t local_pos = t & (local_hgs - 1u);

            uint32_t local_i = local_chunk * (local_hgs << 1u) + local_pos;
            uint32_t local_j = local_i + local_hgs;

            // Global group index for twiddle lookup
            uint32_t global_group = (tg_id << k) + local_chunk;

            FpBase w = twiddles[global_group];
            FpExt a = shared_data[local_i];
            FpExt b = shared_data[local_j];

            FpExt wb = b.scalar_mul(w);
            shared_data[local_i] = a + wb;
            shared_data[local_j] = a - wb;
        }
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    }

    // Write back from threadgroup to global memory
    for (uint32_t i = tg_pos; i < BLOCK_SIZE; i += tg_size) {
        input[block_start + i] = shared_data[i];
    }
}

/// Bit-reverse permutation for extension field elements.
template<typename FpExt>
[[kernel]] void bitrev_permutation_ext(
    device FpExt* input          [[ buffer(0) ]],
    device FpExt* output         [[ buffer(1) ]],
    uint32_t thread_count        [[ threads_per_grid ]],
    uint32_t thread_pos          [[ thread_position_in_grid ]]
)
{
    uint32_t log_n = metal::ctz(thread_count);
    uint32_t rev_pos = 0;

    // Bit-reverse the position
    for (uint32_t i = 0; i < log_n; i++) {
        rev_pos |= ((thread_pos >> i) & 1) << (log_n - 1 - i);
    }

    output[rev_pos] = input[thread_pos];
}
