#include "./fp_u256.cuh"
#include "./utils.h"

extern "C"
{

    __global__ void calc_twiddles(p256::Fp *result,
                                  const p256::Fp &_omega)
    {
        const uint index = blockIdx.x * blockDim.x + threadIdx.x;

        p256::Fp omega = _omega;
        result[index] = omega.pow(index);
    };

    __global__ void calc_twiddles_inv(p256::Fp *result,
                                      const p256::Fp &_omega)
    {
        const uint index = blockIdx.x * blockDim.x + threadIdx.x;

        p256::Fp omega = _omega;
        result[index] = omega.pow(index).inverse();
    };

    __global__ void calc_twiddles_bitrev(p256::Fp *result,
                                         const p256::Fp &_omega)
    {
        const uint index = blockIdx.x * blockDim.x + threadIdx.x;
        const uint size = blockDim.x;

        p256::Fp omega = _omega;
        result[index] = omega.pow(reverse_index(index, size));
    };

    __global__ void calc_twiddles_bitrev_inv(p256::Fp *result,
                                             const p256::Fp &_omega)
    {
        const uint index = blockIdx.x * blockDim.x + threadIdx.x;
        const uint size = blockDim.x;

        p256::Fp omega = _omega;
        result[index] = omega.pow(reverse_index(index, size)).inverse();
    };
}
