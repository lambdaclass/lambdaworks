#include "./fp_u256.cuh"
#include "./utils.cuh"

extern "C" __global__ void calc_twiddles_bitrev(p256::Fp *result,
                                                const p256::Fp &_omega)
{
  const uint index = blockIdx.x * blockDim.x + threadIdx.x;
  const uint size = blockDim.x;

  p256::Fp omega = _omega;
  result[index] = omega.pow(reverse_index(index, size));
};
