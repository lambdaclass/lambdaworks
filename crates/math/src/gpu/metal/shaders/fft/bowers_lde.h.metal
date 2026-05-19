// bowers_lde.h.metal
// Templated kernel bodies for the Bowers-G fused LDE pipeline.
//   * stage0: per-thread = coset_powers[a_idx] * a, coset_powers[b_idx] * b,
//             then DIF butterfly at stage 0
//   * middle: generic DIF butterfly at stage `stage`
//   * tail:   fuses last K stages in threadgroup memory; one threadgroup per block

#pragma once
#include <metal_stdlib>
#include "bowers_butterfly.h.metal"

// ---- stage 0: coset-fold + first DIF butterfly ---------------------------
template<typename Fp, typename TwFp>
inline void bowers_lde_stage0_body(
    device Fp* data,                 // n elements for one column
    constant TwFp* twiddles_bitrev,  // n/2 elements (base-field for Fp3 path)
    constant Fp* coset_powers,       // n elements
    uint32_t n,
    uint32_t thread_pos
) {
    uint32_t a_idx, b_idx, tw_idx;
    bowers_butterfly_indices(thread_pos, 0u, n, a_idx, b_idx, tw_idx);
    Fp a = data[a_idx] * coset_powers[a_idx];
    Fp b = data[b_idx] * coset_powers[b_idx];
    TwFp w = twiddles_bitrev[tw_idx];
    bowers_dif_butterfly<Fp, TwFp>(a, b, w);
    data[a_idx] = a;
    data[b_idx] = b;
}

// ---- middle stages: generic DIF butterfly --------------------------------
template<typename Fp, typename TwFp>
inline void bowers_lde_middle_body(
    device Fp* data,
    constant TwFp* twiddles_bitrev,
    uint32_t n,
    uint32_t stage,
    uint32_t thread_pos
) {
    uint32_t a_idx, b_idx, tw_idx;
    bowers_butterfly_indices(thread_pos, stage, n, a_idx, b_idx, tw_idx);
    Fp a = data[a_idx];
    Fp b = data[b_idx];
    TwFp w = twiddles_bitrev[tw_idx];
    bowers_dif_butterfly<Fp, TwFp>(a, b, w);
    data[a_idx] = a;
    data[b_idx] = b;
}

// ---- tail: last K stages fused in threadgroup memory ---------------------
// Each threadgroup owns one block of BLOCK = (1 << num_stages) elements.
// start_stage is the global stage index of the first fused stage.
template<typename Fp, typename TwFp>
inline void bowers_lde_tail_body(
    device Fp* data,
    constant TwFp* twiddles_bitrev,
    uint32_t n,
    uint32_t start_stage,
    uint32_t num_stages,
    uint32_t tg_id,
    uint32_t tg_pos,
    uint32_t tg_size,
    threadgroup Fp* shared_data
) {
    uint32_t BLOCK = 1u << num_stages;
    uint32_t HALF_BLOCK = BLOCK >> 1;
    uint32_t block_start = tg_id * BLOCK;

    // Load block from global to threadgroup memory
    for (uint32_t i = tg_pos; i < BLOCK; i += tg_size) {
        shared_data[i] = data[block_start + i];
    }
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    // Run num_stages DIF butterfly stages locally.
    for (uint32_t k = 0; k < num_stages; k++) {
        uint32_t local_half = HALF_BLOCK >> k;
        for (uint32_t t = tg_pos; t < HALF_BLOCK; t += tg_size) {
            uint32_t local_block = t / local_half;
            uint32_t pos = t & (local_half - 1);
            uint32_t la = local_block * (local_half << 1) + pos;
            uint32_t lb = la + local_half;

            // Global twiddle index for this stage:
            uint32_t global_stage = start_stage + k;
            uint32_t global_num_blocks = 1u << global_stage;
            uint32_t global_block = (tg_id << k) + local_block;
            TwFp w = twiddles_bitrev[global_num_blocks + global_block];

            Fp a = shared_data[la];
            Fp b = shared_data[lb];
            bowers_dif_butterfly<Fp, TwFp>(a, b, w);
            shared_data[la] = a;
            shared_data[lb] = b;
        }
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    }

    // Write back
    for (uint32_t i = tg_pos; i < BLOCK; i += tg_size) {
        data[block_start + i] = shared_data[i];
    }
}
