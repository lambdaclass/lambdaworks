// FFT butterfly for extension fields
// Key difference from base field FFT:
// - Input coefficients are in the extension field (FpExt)
// - Twiddle factors (roots of unity) are in the base field (FpBase)
//
// This works because primitive roots of unity exist in the base field,
// and we're computing FFT of polynomials with extension field coefficients.

#pragma once

// Radix-2 DIT butterfly for extension fields
// FpExt: Extension field type (e.g., Fp2, Fp3)
// FpBase: Base field type (e.g., Fp)
// Requires: FpExt has mul_by_fp(FpBase) method for scalar multiplication
template <class FpExt, class FpBase>
inline __device__ void _radix2_dit_butterfly_ext(FpExt *input,
                                                  const FpBase *twiddles,
                                                  const int stage,
                                                  const int butterfly_count)
{
    int thread_pos = blockDim.x * blockIdx.x + threadIdx.x;

    if (thread_pos >= butterfly_count) return;

    int half_group_size = butterfly_count >> stage;
    int group = thread_pos / half_group_size;

    int pos_in_group = thread_pos & (half_group_size - 1);
    int i = thread_pos * 2 - pos_in_group;

    // Twiddle factor from base field
    FpBase w = twiddles[group];

    // Input elements from extension field
    FpExt a = input[i];
    FpExt b = input[i + half_group_size];

    // w * b: scalar multiplication (base field * extension field)
    FpExt wb = b.mul_by_fp(w);

    // Butterfly: (a + w*b, a - w*b)
    FpExt res_1 = a + wb;
    FpExt res_2 = a - wb;

    input[i] = res_1;
    input[i + half_group_size] = res_2;
}

// Bit-reversal permutation for extension fields
template <class FpExt>
inline __device__ void _bitrev_permutation_ext(const FpExt *input, FpExt *result, const int len)
{
    unsigned thread_pos = blockDim.x * blockIdx.x + threadIdx.x;
    if (thread_pos >= len) return;

    // Compute bit-reversed index
    unsigned rev = 0;
    unsigned n = len;
    unsigned pos = thread_pos;
    while (n > 1) {
        rev = (rev << 1) | (pos & 1);
        pos >>= 1;
        n >>= 1;
    }

    result[thread_pos] = input[rev];
}
