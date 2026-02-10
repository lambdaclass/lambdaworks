// CUDA kernel instantiation for Goldilocks extension field FFT operations
//
// IMPORTANT: For extension field FFT:
// - Twiddle factors (roots of unity) are in the BASE field (Fp)
// - Input/output coefficients are in the EXTENSION field (Fp2 or Fp3)
//
// This is because primitive roots of unity exist in Fp (with two-adicity 32),
// and we're computing FFT of polynomials whose coefficients are in the extension.

#include "./fp_goldilocks.cuh"
#include "./fp2_goldilocks.cuh"
#include "./fp3_goldilocks.cuh"
#include "../fft/fft_extension.cuh"
#include "../fft/bitrev_permutation.cuh"
#include "../utils.h"

// =====================================================
// QUADRATIC EXTENSION (Fp2) KERNELS
// =====================================================

extern "C"
{
    // Radix-2 DIT butterfly for Fp2 with base field twiddles
    // Input: Fp2 coefficients, Fp twiddles
    __global__ void radix2_dit_butterfly_fp2(goldilocks::Fp2 *input,
                                              const goldilocks::Fp *twiddles,
                                              const int stage,
                                              const int butterfly_count)
    {
        _radix2_dit_butterfly_ext<goldilocks::Fp2, goldilocks::Fp>(
            input, twiddles, stage, butterfly_count);
    }

    // Bit-reversal permutation for Fp2
    __global__ void bitrev_permutation_fp2(const goldilocks::Fp2 *input,
                                            goldilocks::Fp2 *result,
                                            const int len)
    {
        _bitrev_permutation_ext<goldilocks::Fp2>(input, result, len);
    }
}

// =====================================================
// CUBIC EXTENSION (Fp3) KERNELS
// =====================================================

extern "C"
{
    // Radix-2 DIT butterfly for Fp3 with base field twiddles
    // Input: Fp3 coefficients, Fp twiddles
    __global__ void radix2_dit_butterfly_fp3(goldilocks::Fp3 *input,
                                              const goldilocks::Fp *twiddles,
                                              const int stage,
                                              const int butterfly_count)
    {
        _radix2_dit_butterfly_ext<goldilocks::Fp3, goldilocks::Fp>(
            input, twiddles, stage, butterfly_count);
    }

    // Bit-reversal permutation for Fp3
    __global__ void bitrev_permutation_fp3(const goldilocks::Fp3 *input,
                                            goldilocks::Fp3 *result,
                                            const int len)
    {
        _bitrev_permutation_ext<goldilocks::Fp3>(input, result, len);
    }
}

// NOTE: Twiddle factor calculation uses the base field kernels from goldilocks.cu
// since roots of unity are in Fp, not in the extension fields.
// Use calc_twiddles / calc_twiddles_bitrev from goldilocks.cu for extension FFTs.
