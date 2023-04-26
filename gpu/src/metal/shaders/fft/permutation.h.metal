#pragma once
#include "util.h.metal"

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

