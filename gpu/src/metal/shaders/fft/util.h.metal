#pragma once
#include <metal_stdlib>

/// Reverses the `log2(size)` first bits of `i`
uint32_t reverse_index(uint32_t i, uint64_t size) {
    if (size == 1) { // TODO: replace this statement with an alternative solution.
        return i;
    } else {
        return metal::reverse_bits(i) >> (metal::clz(size) + 1);
    }
}
