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

// Threadgroup-cached CFFT evaluation butterfly for Mersenne31
template [[ host_name("cfft_butterfly_tg_mersenne31") ]]
[[kernel]] void cfft_butterfly_tg<FpMersenne31>(
    device FpMersenne31*,
    constant FpMersenne31*,
    constant uint32_t&,
    constant uint32_t&,
    constant uint32_t&,
    uint32_t,
    uint32_t,
    uint32_t,
    threadgroup FpMersenne31*
);

// Threadgroup-cached ICFFT interpolation butterfly for Mersenne31
template [[ host_name("icfft_butterfly_tg_mersenne31") ]]
[[kernel]] void icfft_butterfly_tg<FpMersenne31>(
    device FpMersenne31*,
    constant FpMersenne31*,
    constant uint32_t&,
    constant uint32_t&,
    constant uint32_t&,
    uint32_t,
    uint32_t,
    uint32_t,
    threadgroup FpMersenne31*
);

// Fused multi-stage CFFT evaluation for Mersenne31
template [[ host_name("cfft_butterfly_fused_mersenne31") ]]
[[kernel]] void cfft_butterfly_fused<FpMersenne31>(
    device FpMersenne31*,
    constant FpMersenne31*,
    constant uint32_t&,
    constant uint32_t*,
    uint32_t,
    uint32_t,
    uint32_t,
    threadgroup FpMersenne31*
);

// Fused multi-stage ICFFT interpolation for Mersenne31
template [[ host_name("icfft_butterfly_fused_mersenne31") ]]
[[kernel]] void icfft_butterfly_fused<FpMersenne31>(
    device FpMersenne31*,
    constant FpMersenne31*,
    constant uint32_t&,
    constant uint32_t*,
    uint32_t,
    uint32_t,
    uint32_t,
    threadgroup FpMersenne31*
);
