// Bit-reverse permutation kernel for Metal FFT.
//
// Reorders array elements by reversing the bit-representation of their indices.
// This is a key step in the Cooley-Tukey FFT algorithm.

#pragma once
#include "util.h.metal"

/// Bit-reverse permutation kernel.
///
/// Copies input[i] to result[bit_reverse(i)] for all i in [0, size).
/// Each thread handles one element.
///
/// Parameters:
/// - input: Source array
/// - result: Destination array (must be different from input)
/// - index: Thread's position in grid
/// - size: Total number of elements (must be power of 2)
template<typename Fp>
[[kernel]] void bitrev_permutation(
    device Fp* input [[ buffer(0) ]],
    device Fp* result [[ buffer(1) ]],
    uint index [[ thread_position_in_grid ]],
    uint size [[ threads_per_grid ]]
)
{
    result[index] = input[reverse_index(index, size)];
}
