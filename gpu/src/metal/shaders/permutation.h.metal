#pragma once
#include "util.h.metal"

[[kernel]]
void bitrev_permutation(
    device uint32_t* input [[ buffer(0) ]],
    device uint32_t* result [[ buffer(1) ]],
    uint index [[ thread_position_in_grid ]],
    uint size [[ threads_per_grid ]]
)
{
    result[index] = input[reverse_index(index, size)];
}
