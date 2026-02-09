// FFT utility functions for Metal shaders.
//
// Provides helper functions for bit manipulation used in FFT algorithms.

#pragma once
#include <metal_stdlib>

/// Reverses the `log2(size)` least significant bits of `i`.
///
/// Used for bit-reverse permutation in FFT algorithms.
/// Example: reverse_index(1, 8) = 4 (binary: 001 -> 100)
uint32_t reverse_index(uint32_t i, uint64_t size) {
    if (size == 1) {
        return i;
    } else {
        return metal::reverse_bits(i) >> (metal::clz(size) + 1);
    }
}
