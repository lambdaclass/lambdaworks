#pragma once

#include "../utils.h"

template <class Fp>
inline __device__ void _bitrev_permutation(const Fp *input, Fp *result, const int len)
{
    unsigned thread_pos = blockDim.x * blockIdx.x + threadIdx.x;
    if (thread_pos >= len) return;
    // TODO: guard is not needed for inputs of len >=block_size * 2, if len is pow of two

    result[thread_pos] = input[reverse_index(thread_pos, len)];
};
