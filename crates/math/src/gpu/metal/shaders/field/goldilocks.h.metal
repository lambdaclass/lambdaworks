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
