#include "./fields/fp_u256.cuh"
#include "./utils.h"

extern "C"
{
    // NOTE: In order to calculate the inverse twiddles, call with _omega = _omega.inverse()
    __global__ void calc_twiddles(p256::Fp *result,
                                  const p256::Fp &_omega)
    {
        unsigned index = threadIdx.x;

        p256::Fp omega = _omega;
        result[index] = omega.pow(index);
    };

    // NOTE: In order to calculate the inverse twiddles, call with _omega = _omega.inverse()
    __global__ void calc_twiddles_bitrev(p256::Fp *result,
                                         const p256::Fp &_omega)
    {
        unsigned index = threadIdx.x;
        unsigned size = blockDim.x;

        p256::Fp omega = _omega;
        result[index] = omega.pow(reverse_index(index, size));
    };

}
