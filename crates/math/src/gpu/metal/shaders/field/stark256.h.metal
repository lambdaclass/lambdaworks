// Stark252 prime field instantiation for Metal shaders.
//
// Instantiates the Fp256 template for the Stark252 field used by StarkWare/Cairo.
// Modulus: 0x800000000000011000000000000000000000000000000000000000000000001
//        = 2^251 + 17 * 2^192 + 1

#pragma once

#include "fp_u256.h.metal"

#include "../fft/fft.h.metal"
#include "../fft/twiddles.h.metal"
#include "../fft/permutation.h.metal"

// Stark252 prime field type alias
typedef Fp256<
    // N = Prime modulus (little-endian limbs)
    /* N_0 */ 576460752303423505,
    /* N_1 */ 0,
    /* N_2 */ 0,
    /* N_3 */ 1,
    // R_SQUARED = R^2 mod N where R = 2^256
    /* R_SQUARED_0 */ 576413109808302096,
    /* R_SQUARED_1 */ 18446744073700081664,
    /* R_SQUARED_2 */ 5151653887,
    /* R_SQUARED_3 */ 18446741271209837569,
    // N_PRIME = -N^(-1) mod R
    /* N_PRIME_0 */ 576460752303423504,
    /* N_PRIME_1 */ 18446744073709551615,
    /* N_PRIME_2 */ 18446744073709551615,
    /* N_PRIME_3 */ 18446744073709551615
> FpStark256;

// Explicit template instantiations with host-callable names.
// These allow Rust code to find the kernels by name.

template [[ host_name("radix2_dit_butterfly_stark256") ]]
[[kernel]] void radix2_dit_butterfly<FpStark256>(
    device FpStark256*,
    constant FpStark256*,
    constant uint32_t&,
    uint32_t,
    uint32_t
);

template [[ host_name("calc_twiddles_stark256") ]]
[[kernel]] void calc_twiddles<FpStark256>(
    device FpStark256*,
    constant FpStark256&,
    uint
);

template [[ host_name("calc_twiddles_inv_stark256") ]]
[[kernel]] void calc_twiddles_inv<FpStark256>(
    device FpStark256*,
    constant FpStark256&,
    uint
);

template [[ host_name("calc_twiddles_bitrev_stark256") ]]
[[kernel]] void calc_twiddles_bitrev<FpStark256>(
    device FpStark256*,
    constant FpStark256&,
    uint,
    uint
);

template [[ host_name("calc_twiddles_bitrev_inv_stark256") ]]
[[kernel]] void calc_twiddles_bitrev_inv<FpStark256>(
    device FpStark256*,
    constant FpStark256&,
    uint,
    uint
);

template [[ host_name("bitrev_permutation_stark256") ]]
[[kernel]] void bitrev_permutation<FpStark256>(
    device FpStark256*,
    device FpStark256*,
    uint,
    uint
);

// Threadgroup-cached radix-2 DIT butterfly for Stark252
template [[ host_name("radix2_dit_butterfly_tg_stark256") ]]
[[kernel]] void radix2_dit_butterfly_tg<FpStark256>(
    device FpStark256*,
    constant FpStark256*,
    constant uint32_t&,
    constant uint32_t&,
    uint32_t,
    uint32_t,
    uint32_t,
    uint32_t,
    threadgroup FpStark256*
);

// Fused multi-stage radix-2 DIT butterfly for Stark252
template [[ host_name("radix2_dit_butterfly_fused_stark256") ]]
[[kernel]] void radix2_dit_butterfly_fused<FpStark256>(
    device FpStark256*,
    constant FpStark256*,
    constant uint32_t&,
    constant uint32_t&,
    uint32_t,
    uint32_t,
    uint32_t,
    threadgroup FpStark256*
);
