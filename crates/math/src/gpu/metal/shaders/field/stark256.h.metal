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
namespace {
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
    > Fp;
}

// Explicit template instantiations with host-callable names.
// These allow Rust code to find the kernels by name.

template [[ host_name("radix2_dit_butterfly_stark256") ]]
[[kernel]] void radix2_dit_butterfly<Fp>(
    device Fp*,
    constant Fp*,
    constant uint32_t&,
    uint32_t,
    uint32_t
);

template [[ host_name("calc_twiddles_stark256") ]]
[[kernel]] void calc_twiddles<Fp>(
    device Fp*,
    constant Fp&,
    uint
);

template [[ host_name("calc_twiddles_inv_stark256") ]]
[[kernel]] void calc_twiddles_inv<Fp>(
    device Fp*,
    constant Fp&,
    uint
);

template [[ host_name("calc_twiddles_bitrev_stark256") ]]
[[kernel]] void calc_twiddles_bitrev<Fp>(
    device Fp*,
    constant Fp&,
    uint,
    uint
);

template [[ host_name("calc_twiddles_bitrev_inv_stark256") ]]
[[kernel]] void calc_twiddles_bitrev_inv<Fp>(
    device Fp*,
    constant Fp&,
    uint,
    uint
);

template [[ host_name("bitrev_permutation_stark256") ]]
[[kernel]] void bitrev_permutation<Fp>(
    device Fp*,
    device Fp*,
    uint,
    uint
);
