// Radix-2 DIT FFT butterfly kernel for extension fields with base field twiddles.
//
// This kernel supports FFT operations where:
// - Input coefficients are in an extension field (Fp2, Fp3, etc.)
// - Twiddle factors are in the base field
//
// The butterfly operation is:
//   a' = a + w*b    (where a, b are extension elements, w is base field)
//   b' = a - w*b
//
// Uses scalar multiplication (extension * base -> extension) for efficiency.

#pragma once

#include <metal_stdlib>

/// Radix-2 DIT butterfly for extension field elements with base field twiddles.
///
/// Template parameters:
/// - FpExt: Extension field element type (e.g., Fp2Goldilocks, Fp3Goldilocks)
/// - FpBase: Base field element type (e.g., Fp64Goldilocks)
///
/// The FpExt type must provide:
/// - operator+(FpExt) -> FpExt
/// - operator-(FpExt) -> FpExt
/// - scalar_mul(FpBase) -> FpExt
template<typename FpExt, typename FpBase>
[[kernel]] void radix2_dit_butterfly_ext(
    device FpExt* input          [[ buffer(0) ]],
    constant FpBase* twiddles    [[ buffer(1) ]],
    constant uint32_t& stage     [[ buffer(2) ]],
    uint32_t thread_count        [[ threads_per_grid ]],
    uint32_t thread_pos          [[ thread_position_in_grid ]]
)
{
    // Calculate group size and position
    uint32_t half_group_size = thread_count >> stage;
    uint32_t group = thread_pos >> metal::ctz(half_group_size);

    uint32_t pos_in_group = thread_pos & (half_group_size - 1);
    uint32_t i = thread_pos * 2 - pos_in_group;

    // Load base field twiddle and extension field input values
    FpBase w = twiddles[group];
    FpExt a = input[i];
    FpExt b = input[i + half_group_size];

    // Butterfly computation using scalar multiplication
    FpExt wb = b.scalar_mul(w);  // extension * base -> extension
    FpExt res_1 = a + wb;
    FpExt res_2 = a - wb;

    // Store results
    input[i] = res_1;
    input[i + half_group_size] = res_2;
}

/// Bit-reverse permutation for extension field elements.
template<typename FpExt>
[[kernel]] void bitrev_permutation_ext(
    device FpExt* input          [[ buffer(0) ]],
    device FpExt* output         [[ buffer(1) ]],
    uint32_t thread_count        [[ threads_per_grid ]],
    uint32_t thread_pos          [[ thread_position_in_grid ]]
)
{
    uint32_t log_n = metal::ctz(thread_count);
    uint32_t rev_pos = 0;

    // Bit-reverse the position
    for (uint32_t i = 0; i < log_n; i++) {
        rev_pos |= ((thread_pos >> i) & 1) << (log_n - 1 - i);
    }

    output[rev_pos] = input[thread_pos];
}
