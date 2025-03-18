#pragma once

#include "../utils.h"

// NOTE: In order to calculate the inverse twiddles, call with _omega = _omega.inverse()
template <class Fp>
inline __device__ void _calc_twiddles(Fp *result, const Fp &_omega, const int count)
{
    unsigned thread_pos = blockDim.x * blockIdx.x + threadIdx.x;
    if (thread_pos >= count) return;
    // TODO: guard is not needed for count >=block_size * 2, if count is pow of two

    Fp omega = _omega;
    result[thread_pos] = omega.pow(thread_pos);
};

// NOTE: In order to calculate the inverse twiddles, call with _omega = _omega.inverse()
template <class Fp>
inline __device__ void _calc_twiddles_bitrev(Fp *result, const Fp &_omega, const int count)
{
    unsigned thread_pos = blockDim.x * blockIdx.x + threadIdx.x;
    if (thread_pos >= count) return;
    // TODO: guard is not needed for count >=block_size * 2, if count is pow of two

    Fp omega = _omega;
    result[thread_pos] = omega.pow(reverse_index(thread_pos, count));
};


