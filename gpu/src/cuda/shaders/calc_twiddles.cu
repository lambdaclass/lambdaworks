#include "./fp_u256.cuh"

extern "C" __global__ void calc_twiddles(p256::Fp *result,
                                         const p256::Fp &_omega)
{
  const uint index = blockIdx.x * blockDim.x + threadIdx.x;

  p256::Fp omega = _omega;
  result[index] = omega.pow(index);
};
