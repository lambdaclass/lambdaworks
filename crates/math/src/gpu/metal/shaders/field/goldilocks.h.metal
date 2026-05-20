// Goldilocks 64-bit prime field instantiation for Metal shaders.
//
// Instantiates FFT kernels for the Goldilocks field used by Plonky2/Plonky3.
// Prime: p = 2^64 - 2^32 + 1 = 0xFFFFFFFF00000001
//
// Two-adicity: 32 (p - 1 = 2^32 * (2^32 - 1))
// Primitive 2^32-th root of unity: 1753635133440165772

#pragma once

#include "fp_u64.h.metal"

#include "../fft/fft.h.metal"
#include "../fft/twiddles.h.metal"
#include "../fft/permutation.h.metal"

// Goldilocks field type alias
typedef Fp64Goldilocks FpGoldilocks;

// Explicit template instantiations with host-callable names.
// These allow Rust code to find the kernels by name.

template [[ host_name("radix2_dit_butterfly_Goldilocks") ]]
[[kernel]] void radix2_dit_butterfly<FpGoldilocks>(
    device FpGoldilocks*,
    constant FpGoldilocks*,
    constant uint32_t&,
    uint32_t,
    uint32_t
);

template [[ host_name("calc_twiddles_Goldilocks") ]]
[[kernel]] void calc_twiddles<FpGoldilocks>(
    device FpGoldilocks*,
    constant FpGoldilocks&,
    uint
);

template [[ host_name("calc_twiddles_inv_Goldilocks") ]]
[[kernel]] void calc_twiddles_inv<FpGoldilocks>(
    device FpGoldilocks*,
    constant FpGoldilocks&,
    uint
);

template [[ host_name("calc_twiddles_bitrev_Goldilocks") ]]
[[kernel]] void calc_twiddles_bitrev<FpGoldilocks>(
    device FpGoldilocks*,
    constant FpGoldilocks&,
    uint,
    uint
);

template [[ host_name("calc_twiddles_bitrev_inv_Goldilocks") ]]
[[kernel]] void calc_twiddles_bitrev_inv<FpGoldilocks>(
    device FpGoldilocks*,
    constant FpGoldilocks&,
    uint,
    uint
);

template [[ host_name("bitrev_permutation_Goldilocks") ]]
[[kernel]] void bitrev_permutation<FpGoldilocks>(
    device FpGoldilocks*,
    device FpGoldilocks*,
    uint,
    uint
);

// Threadgroup-cached radix-2 DIT butterfly for Goldilocks
template [[ host_name("radix2_dit_butterfly_tg_Goldilocks") ]]
[[kernel]] void radix2_dit_butterfly_tg<FpGoldilocks>(
    device FpGoldilocks*,
    constant FpGoldilocks*,
    constant uint32_t&,
    constant uint32_t&,
    uint32_t,
    uint32_t,
    uint32_t,
    uint32_t,
    threadgroup FpGoldilocks*
);

// Fused multi-stage radix-2 DIT butterfly for Goldilocks
template [[ host_name("radix2_dit_butterfly_fused_Goldilocks") ]]
[[kernel]] void radix2_dit_butterfly_fused<FpGoldilocks>(
    device FpGoldilocks*,
    constant FpGoldilocks*,
    constant uint32_t&,
    constant uint32_t&,
    uint32_t,
    uint32_t,
    uint32_t,
    threadgroup FpGoldilocks*
);

// ---- Bowers-G NTT + fused LDE kernel instantiations ----
#include "../fft/bowers_lde.h.metal"

// Stage 0 (smallest stride) + coset fold. Used only when no fused head.
[[ host_name("bowers_lde_stage0_Goldilocks") ]]
[[kernel]] void bowers_lde_stage0_Goldilocks(
    device   FpGoldilocks* data            [[ buffer(0) ]],
    constant FpGoldilocks* twiddles_bitrev [[ buffer(1) ]],
    constant FpGoldilocks* coset_powers    [[ buffer(2) ]],
    uint32_t               thread_pos      [[ thread_position_in_grid ]]
) {
    bowers_lde_stage0_body<FpGoldilocks>(
        data, twiddles_bitrev, coset_powers, thread_pos
    );
}

// Middle stages (arbitrary stage index).
[[ host_name("bowers_lde_middle_Goldilocks") ]]
[[kernel]] void bowers_lde_middle_Goldilocks(
    device   FpGoldilocks* data            [[ buffer(0) ]],
    constant FpGoldilocks* twiddles_bitrev [[ buffer(1) ]],
    constant uint32_t&     stage           [[ buffer(2) ]],
    uint32_t               thread_pos      [[ thread_position_in_grid ]]
) {
    bowers_lde_middle_body<FpGoldilocks, FpGoldilocks>(
        data, twiddles_bitrev, stage, thread_pos
    );
}

// Fused head: first num_stages stages in threadgroup memory with coset multiply on load.
[[ host_name("bowers_lde_head_Goldilocks") ]]
[[kernel]] void bowers_lde_head_Goldilocks(
    device   FpGoldilocks* data            [[ buffer(0) ]],
    constant FpGoldilocks* twiddles_bitrev [[ buffer(1) ]],
    constant FpGoldilocks* coset_powers    [[ buffer(2) ]],
    constant uint32_t&     num_stages      [[ buffer(3) ]],
    uint32_t               tg_id           [[ threadgroup_position_in_grid ]],
    uint32_t               tg_pos          [[ thread_position_in_threadgroup ]],
    uint32_t               tg_size         [[ threads_per_threadgroup ]],
    threadgroup FpGoldilocks* shared_data  [[ threadgroup(0) ]]
) {
    bowers_lde_head_body<FpGoldilocks, FpGoldilocks>(
        data, twiddles_bitrev, coset_powers, num_stages, tg_id, tg_pos, tg_size, shared_data
    );
}
