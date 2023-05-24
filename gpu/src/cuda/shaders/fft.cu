#include "./fp_u256.cuh"

extern "C" __global__ void radix2_dit_butterfly(p256::Fp *input,
                                                const p256::Fp *twiddles)
{
  int group = blockIdx.x;
  int pos_in_group = threadIdx.x;
  int half_group_size = blockDim.x;

  int i = group * half_group_size * 2 + pos_in_group;

  p256::Fp w = twiddles[group];
  p256::Fp a = input[i];
  p256::Fp b = input[i + half_group_size];

  p256::Fp res_1 = a + w * b;
  p256::Fp res_2 = a - w * b;

  input[i] = res_1;                   // --\/--
  input[i + half_group_size] = res_2; // --/\--
};
