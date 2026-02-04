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
namespace {
    typedef Fp64Goldilocks Fp;
}

// Explicit template instantiations with host-callable names.
// These allow Rust code to find the kernels by name.

template [[ host_name("radix2_dit_butterfly_goldilocks") ]]
[[kernel]] void radix2_dit_butterfly<Fp>(
    device Fp*,
    constant Fp*,
    constant uint32_t&,
    uint32_t,
    uint32_t
);

template [[ host_name("calc_twiddles_goldilocks") ]]
[[kernel]] void calc_twiddles<Fp>(
    device Fp*,
    constant Fp&,
    uint
);

template [[ host_name("calc_twiddles_inv_goldilocks") ]]
[[kernel]] void calc_twiddles_inv<Fp>(
    device Fp*,
    constant Fp&,
    uint
);

template [[ host_name("calc_twiddles_bitrev_goldilocks") ]]
[[kernel]] void calc_twiddles_bitrev<Fp>(
    device Fp*,
    constant Fp&,
    uint,
    uint
);

template [[ host_name("calc_twiddles_bitrev_inv_goldilocks") ]]
[[kernel]] void calc_twiddles_bitrev_inv<Fp>(
    device Fp*,
    constant Fp&,
    uint,
    uint
);

template [[ host_name("bitrev_permutation_goldilocks") ]]
[[kernel]] void bitrev_permutation<Fp>(
    device Fp*,
    device Fp*,
    uint,
    uint
);
