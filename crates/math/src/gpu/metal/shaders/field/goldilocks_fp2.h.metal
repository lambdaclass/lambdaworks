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
typedef Fp64Goldilocks FpBaseGoldilocks;
typedef Fp2Goldilocks FpExtFp2;

// ============================================================
// FFT Kernel Instantiations for Goldilocks Fp2
// ============================================================

// Butterfly kernel for extension field FFT with base field twiddles
template [[ host_name("radix2_dit_butterfly_Goldilocks_fp2") ]]
[[kernel]] void radix2_dit_butterfly_ext<FpExtFp2, FpBaseGoldilocks>(
    device FpExtFp2*,
    constant FpBaseGoldilocks*,
    constant uint32_t&,
    uint32_t,
    uint32_t
);

// Bit-reverse permutation for extension field elements
template [[ host_name("bitrev_permutation_Goldilocks_fp2") ]]
[[kernel]] void bitrev_permutation_ext<FpExtFp2>(
    device FpExtFp2*,
    device FpExtFp2*,
    uint32_t,
    uint32_t
);
