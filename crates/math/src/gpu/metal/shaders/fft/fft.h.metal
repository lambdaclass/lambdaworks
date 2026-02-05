// Radix-2 Decimation-In-Time (DIT) FFT butterfly kernel for Metal.
//
// Implements the core butterfly operation of the Cooley-Tukey FFT algorithm.
// Each kernel invocation processes one butterfly operation for a given stage.

#pragma once

#include <metal_stdlib>

/// Radix-2 DIT butterfly operation.
///
/// Performs the butterfly computation for one stage of the FFT:
///   a' = a + w*b
///   b' = a - w*b
///
/// where w is the appropriate twiddle factor for this butterfly's group.
///
/// Parameters:
/// - input: Array of field elements (modified in-place)
/// - twiddles: Pre-computed twiddle factors in bit-reversed order
/// - stage: Current FFT stage (0 to log2(n)-1)
/// - thread_count: Total number of butterflies (n/2)
/// - thread_pos: This thread's butterfly index
///
/// Memory layout:
/// For stage s, butterflies are grouped. Each group has 2^s elements.
/// Within a group, pairs (i, i + group_size/2) are processed together.
template<typename Fp>
[[kernel]] void radix2_dit_butterfly(
    device Fp* input          [[ buffer(0) ]],
    constant Fp* twiddles     [[ buffer(1) ]],
    constant uint32_t& stage  [[ buffer(2) ]],
    uint32_t thread_count     [[ threads_per_grid ]],
    uint32_t thread_pos       [[ thread_position_in_grid ]]
)
{
    // Calculate group size and position
    uint32_t half_group_size = thread_count >> stage; // thread_count / group_count
    uint32_t group = thread_pos >> metal::ctz(half_group_size); // thread_pos / half_group_size

    uint32_t pos_in_group = thread_pos & (half_group_size - 1); // thread_pos % half_group_size
    uint32_t i = thread_pos * 2 - pos_in_group; // multiply quotient by 2

    // Load twiddle factor and input values
    Fp w = twiddles[group];
    Fp a = input[i];
    Fp b = input[i + half_group_size];

    // Butterfly computation
    Fp res_1 = a + w*b;
    Fp res_2 = a - w*b;

    // Store results
    input[i]                    = res_1; // --\/--
    input[i + half_group_size]  = res_2; // --/\--
}
