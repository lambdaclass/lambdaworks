// Goldilocks Fp2 extension field kernel instantiations for Metal shaders.
//
// Instantiates FFT kernels for Goldilocks quadratic extension field (Fp2).
// This enables FFT operations with:
// - Coefficients in Fp2 (extension field)
// - Twiddles in Fp (base Goldilocks field)

#pragma once

#include "fp_u64.h.metal"
#include "fp2_goldilocks.h.metal"

#include "../fft/fft_extension.h.metal"
#include "../fft/permutation.h.metal"

// Type aliases for clarity
namespace {
    typedef Fp64Goldilocks FpBase;
    typedef Fp2Goldilocks FpExt;
}

// ============================================================
// FFT Kernel Instantiations for Goldilocks Fp2
// ============================================================

// Butterfly kernel for extension field FFT with base field twiddles
template [[ host_name("radix2_dit_butterfly_goldilocks_fp2") ]]
[[kernel]] void radix2_dit_butterfly_ext<FpExt, FpBase>(
    device FpExt*,
    constant FpBase*,
    constant uint32_t&,
    uint32_t,
    uint32_t
);

// Bit-reverse permutation for extension field elements
template [[ host_name("bitrev_permutation_goldilocks_fp2") ]]
[[kernel]] void bitrev_permutation_ext<FpExt>(
    device FpExt*,
    device FpExt*,
    uint32_t,
    uint32_t
);
