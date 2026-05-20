// bowers_butterfly.h.metal
// Templated DIF butterfly for the Bowers-G NTT layout, matching the CPU
// reference `ntt_bowers_butterflies` in `crates/math/src/fft/cpu/ntt_bowers_goldilocks.rs`.
//
// Stage convention (matches CPU's `log_half = 0..log_n` outer loop):
//   stage 0   -> smallest butterflies (half = 1)
//   stage k   -> half = 2^k, num_blocks = n / 2^(k+1)
//   stage log_n - 1 -> one giant butterfly (half = n/2)
//
// Twiddle layout: bitrev table of length n/2 (matches `compute_bowers_twiddles`
// and `bowers_twiddles_goldilocks`). Index is `block` directly. Block 0 always
// uses w = 1 (the caller is responsible for this special-case).
//
// Butterfly:  a' = a + b ;  b' = (a - b) * w
// Sequential twiddle access within a stage gives perfect coalescing on GPU.

#pragma once
#include <metal_stdlib>

template<typename Fp, typename TwFp>
inline void bowers_dif_butterfly(thread Fp& a, thread Fp& b, TwFp w) {
    Fp sum  = a + b;
    Fp diff = (a - b) * w;
    a = sum;
    b = diff;
}

// Per-thread index calculation. One thread = one butterfly; grid size is n/2.
inline void bowers_butterfly_indices(
    uint32_t thread_pos, uint32_t stage,
    thread uint32_t& a_idx, thread uint32_t& b_idx, thread uint32_t& tw_idx
) {
    uint32_t half_sz    = 1u << stage;
    uint32_t block      = thread_pos >> stage;
    uint32_t pos_in_blk = thread_pos & (half_sz - 1u);
    a_idx  = (block << (stage + 1u)) + pos_in_blk;
    b_idx  = a_idx + half_sz;
    tw_idx = block;
}
