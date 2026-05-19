// bowers_butterfly.h.metal
// Templated DIF butterfly for the Bowers-G NTT layout.
//
// Twiddles are stored in bit-reversed order:
//   stage k uses twiddles[2^k .. 2^(k+1))
//   The first useful twiddle is at index 1 (stage 0 has num_blocks = 1 and
//   that single twiddle is at index 1; index 0 is unused / always 1).
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

// Stage parameters for a column of length n at stage `stage` (0 = outermost,
// matching `ntt_bowers_butterflies`'s `log_half = 0` smallest-half stage):
//
//   half        = n >> (stage + 1)
//   num_blocks  = 1 << stage
//   block       = thread_pos / half
//   pos_in_blk  = thread_pos & (half - 1)
//   a_idx       = block * (2*half) + pos_in_blk
//   b_idx       = a_idx + half
//   twiddle_idx = num_blocks + block        (bitrev layout)
//
// Each thread does exactly one butterfly; grid size is n/2.
inline void bowers_butterfly_indices(
    uint32_t thread_pos, uint32_t stage, uint32_t n,
    thread uint32_t& a_idx, thread uint32_t& b_idx, thread uint32_t& tw_idx
) {
    uint32_t half_sz    = n >> (stage + 1);
    uint32_t num_blocks = 1u << stage;
    uint32_t block      = thread_pos / half_sz;
    uint32_t pos_in_blk = thread_pos & (half_sz - 1);
    a_idx  = block * (half_sz << 1) + pos_in_blk;
    b_idx  = a_idx + half_sz;
    tw_idx = num_blocks + block;
}
