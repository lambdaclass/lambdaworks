// CUDA kernel instantiation for Goldilocks field FFT operations
// p = 2^64 - 2^32 + 1 (the "Goldilocks" prime)
//
// This field is particularly efficient for FFT because:
// - 64-bit arithmetic (native on modern GPUs)
// - Two-adicity of 32 (supports FFT up to 2^32 elements)
// - EPSILON reduction trick avoids expensive division

#include "./fp_goldilocks.cuh"
#include "../fft/fft.cuh"
#include "../fft/twiddles.cuh"
#include "../fft/bitrev_permutation.cuh"
#include "../utils.h"

extern "C"
{
    // Radix-2 Decimation-In-Time butterfly operation
    // Each thread computes one butterfly: (a, b) -> (a + w*b, a - w*b)
    __global__ void radix2_dit_butterfly(goldilocks::Fp *input,
                                         const goldilocks::Fp *twiddles,
                                         const int stage,
                                         const int butterfly_count)
    {
        _radix2_dit_butterfly<goldilocks::Fp>(input, twiddles, stage, butterfly_count);
    }

    // Calculate twiddle factors: omega^0, omega^1, omega^2, ...
    // For IFFT, pass omega.inverse() instead of omega
    __global__ void calc_twiddles(goldilocks::Fp *result,
                                  const goldilocks::Fp &_omega,
                                  const int count)
    {
        _calc_twiddles<goldilocks::Fp>(result, _omega, count);
    }

    // Calculate twiddle factors in bit-reversed order
    // More cache-friendly for certain FFT implementations
    __global__ void calc_twiddles_bitrev(goldilocks::Fp *result,
                                         const goldilocks::Fp &_omega,
                                         const int count)
    {
        _calc_twiddles_bitrev<goldilocks::Fp>(result, _omega, count);
    }

    // Bit-reversal permutation for FFT input/output reordering
    __global__ void bitrev_permutation(const goldilocks::Fp *input,
                                       goldilocks::Fp *result,
                                       const int len)
    {
        _bitrev_permutation<goldilocks::Fp>(input, result, len);
    }
}
