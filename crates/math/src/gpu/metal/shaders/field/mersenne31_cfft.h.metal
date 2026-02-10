// Mersenne31 Circle FFT kernel instantiation for Metal.
//
// Instantiates CFFT butterfly kernels for the Mersenne31 field.
// These host_name attributes allow Rust to look up kernels by name.

#pragma once

#include "mersenne31.h.metal"
#include "../fft/cfft.h.metal"

// CFFT evaluation butterfly for Mersenne31
template [[ host_name("cfft_butterfly_mersenne31") ]]
[[kernel]] void cfft_butterfly<FpMersenne31>(
    device FpMersenne31*,
    constant FpMersenne31*,
    constant uint32_t&,
    constant uint32_t&,
    uint32_t
);

// ICFFT interpolation butterfly for Mersenne31
template [[ host_name("icfft_butterfly_mersenne31") ]]
[[kernel]] void icfft_butterfly<FpMersenne31>(
    device FpMersenne31*,
    constant FpMersenne31*,
    constant uint32_t&,
    constant uint32_t&,
    uint32_t
);
