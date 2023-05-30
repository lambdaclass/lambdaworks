#pragma once

#include "../utils.h"

// NOTE: In order to calculate the inverse twiddles, call with _omega = _omega.inverse()
template <class Fp>
inline __global__ void _calc_twiddles(Fp *result, const Fp *_omega)
{
    int index = threadIdx.x;

    p256::Fp omega = _omega[0];
    result[index] = omega.pow((unsigned)index);
};

// NOTE: In order to calculate the inverse twiddles, call with _omega = _omega.inverse()
template <class Fp>
inline __global__ void _calc_twiddles_bitrev(Fp *result, const Fp *_omega)
{
    int index = threadIdx.x;
    int size = blockDim.x;

    p256::Fp omega = _omega[0];
    result[index] = omega.pow(reverse_index((unsigned)index, (unsigned)size));
};


