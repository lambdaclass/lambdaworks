#pragma once

#include "../utils.h"

template <class Fp>
inline __device__ void _bitrev_permutation(const Fp *input, Fp *result)
{
  unsigned index = threadIdx.x;
  unsigned size = blockDim.x;

  result[index] = input[reverse_index(index, size)];
};
