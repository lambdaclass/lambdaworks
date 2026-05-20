// bowers_lde.h.metal
// Templated kernel bodies for the Bowers-G fused LDE pipeline.
//   * middle: generic DIF butterfly at an arbitrary stage
//   * head:   fuses the first K stages in threadgroup memory, applying the
//             coset multiply on load
//
// Both bodies match the CPU reference `ntt_bowers_butterflies` exactly.
// Block 0 always uses w = 1; all other blocks use w = twiddles_bitrev[block].
//
// Type parameters:
//   Fp   - element type of the data (base field Fp64, or extension Fp3)
//   TwFp - element type of twiddles and coset powers (always the base field)

#pragma once
#include <metal_stdlib>
#include "bowers_butterfly.h.metal"

// Helper: read twiddle for a given block index, with block-0 specialization.
template<typename TwFp>
inline TwFp bowers_twiddle_for_block(constant TwFp* twiddles_bitrev, uint32_t block) {
    return (block == 0u) ? TwFp(1u) : twiddles_bitrev[block];
}

// ---- middle stages: arbitrary stage --------------------------------------
template<typename Fp, typename TwFp>
inline void bowers_lde_middle_body(
    device Fp* data,
    constant TwFp* twiddles_bitrev,
    uint32_t stage,
    uint32_t thread_pos
) {
    uint32_t a_idx, b_idx, tw_idx;
    bowers_butterfly_indices(thread_pos, stage, a_idx, b_idx, tw_idx);
    Fp a = data[a_idx];
    Fp b = data[b_idx];
    TwFp w = bowers_twiddle_for_block<TwFp>(twiddles_bitrev, tw_idx);
    bowers_dif_butterfly<Fp, TwFp>(a, b, w);
    data[a_idx] = a;
    data[b_idx] = b;
}

// ---- fused head: first num_stages stages (0..num_stages) in threadgroup mem
// Each threadgroup owns one contiguous block of BLOCK = (1 << num_stages)
// elements. Coset multiply (base-field scalar) is applied during the load.
//
// At sub-stage k in [0, num_stages):
//   local_half        = 1 << k
//   global_block      = (tg_id << (num_stages - 1 - k)) + local_block
template<typename Fp, typename TwFp>
inline void bowers_lde_head_body(
    device Fp* data,
    constant TwFp* twiddles_bitrev,
    constant TwFp* coset_powers,
    uint32_t num_stages,
    uint32_t tg_id,
    uint32_t tg_pos,
    uint32_t tg_size,
    threadgroup Fp* shared_data
) {
    uint32_t BLOCK = 1u << num_stages;
    uint32_t HALF_BLOCK = BLOCK >> 1u;
    uint32_t block_start = tg_id * BLOCK;

    // Load block from global to threadgroup memory WITH coset multiply.
    for (uint32_t i = tg_pos; i < BLOCK; i += tg_size) {
        Fp x = data[block_start + i];
        TwFp c = coset_powers[block_start + i];
        shared_data[i] = x * c;
    }
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    for (uint32_t k = 0u; k < num_stages; k++) {
        uint32_t local_half = 1u << k;
        uint32_t local_mask = local_half - 1u;
        uint32_t shift_to_global = num_stages - 1u - k;
        uint32_t tg_block_offset = tg_id << shift_to_global;

        for (uint32_t t = tg_pos; t < HALF_BLOCK; t += tg_size) {
            uint32_t local_block = t >> k;
            uint32_t pos = t & local_mask;
            uint32_t la = (local_block << (k + 1u)) + pos;
            uint32_t lb = la + local_half;

            uint32_t global_block = tg_block_offset + local_block;
            TwFp w = bowers_twiddle_for_block<TwFp>(twiddles_bitrev, global_block);

            Fp a = shared_data[la];
            Fp b = shared_data[lb];
            bowers_dif_butterfly<Fp, TwFp>(a, b, w);
            shared_data[la] = a;
            shared_data[lb] = b;
        }
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    }

    // Write back.
    for (uint32_t i = tg_pos; i < BLOCK; i += tg_size) {
        data[block_start + i] = shared_data[i];
    }
}
