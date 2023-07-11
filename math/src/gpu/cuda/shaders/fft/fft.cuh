#pragma once

template <class Fp>
inline __device__ void _radix2_dit_butterfly(Fp *input,
                                             const Fp *twiddles,
                                             const int &stage,
                                             const int &butterfly_count)
{
    if (blockIdx.x >= butterfly_count) return;

    long group_count = 1 << stage;
    long half_group_size = butterfly_count / group_count;
    long group = threadIdx.x / half_group_size;

    long pos_in_group = threadIdx.x % half_group_size;
    long i = threadIdx.x * 2 - pos_in_group; // multiply quotient by 2

    Fp w = twiddles[group];
    Fp a = input[i];
    Fp b = input[i + half_group_size];

    Fp res_1 = a + w * b;
    Fp res_2 = a - w * b;

    input[i] = res_1;                   // --\/--
    input[i + half_group_size] = res_2; // --/\--
};
