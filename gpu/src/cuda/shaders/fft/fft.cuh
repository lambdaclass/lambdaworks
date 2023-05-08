#pragma once

template <class Fp>
inline __device__ void radix2_dit_butterfly(Fp *input,
                                            const Fp *twiddles)
{
  int group = blockIdx.x;
  int pos_in_group = threadIdx.x;
  int half_group_size = blockDim.x;

  int i = group * half_group_size * 2 + pos_in_group;

  Fp w = twiddles[group];
  Fp a = input[i];
  Fp b = input[i + half_group_size];

  Fp res_1 = a + w * b;
  Fp res_2 = a - w * b;

  input[i] = res_1;                   // --\/--
  input[i + half_group_size] = res_2; // --/\--
};
