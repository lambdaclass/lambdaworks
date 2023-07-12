#pragma once

template <class Fp>
inline __device__ void _radix2_dit_butterfly(Fp *input,
                                             const Fp *twiddles,
                                             const int stage,
                                             const int butterfly_count)
{
    int thread_pos = blockDim.x * blockIdx.x + threadIdx.x;

    if (thread_pos >= butterfly_count) return;
    // TODO: guard is not needed for inputs of len >=block_size * 2, only if len is pow of two

    int half_group_size = butterfly_count >> stage;
    int group = thread_pos / half_group_size;

    int pos_in_group = thread_pos & (half_group_size - 1);
    int i = thread_pos * 2 - pos_in_group; // multiply quotient by 2

    Fp w = twiddles[group];
    Fp a = input[i];
    Fp b = input[i + half_group_size];

    Fp res_1 = a + w * b;
    Fp res_2 = a - w * b;

    input[i] = res_1;                   // --\/--
    input[i + half_group_size] = res_2; // --/\--
};
