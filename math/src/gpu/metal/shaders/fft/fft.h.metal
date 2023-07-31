#pragma once

#include <metal_stdlib>

template<typename Fp>
[[kernel]] void radix2_dit_butterfly(
    device Fp* input          [[ buffer(0) ]],
    constant Fp* twiddles     [[ buffer(1) ]],
    constant uint32_t& stage  [[ buffer(2) ]],
    uint32_t thread_count     [[ threads_per_grid ]],
    uint32_t thread_pos       [[ thread_position_in_grid ]]
)
{
    uint32_t half_group_size = thread_count >> stage; // thread_count / group_count
    uint32_t group = thread_pos >> metal::ctz(half_group_size); // thread_pos / half_group_size

    uint32_t pos_in_group = thread_pos & (half_group_size - 1); // thread_pos % half_group_size
    uint32_t i = thread_pos * 2 - pos_in_group; // multiply quotient by 2

    Fp w = twiddles[group];
    Fp a = input[i];
    Fp b = input[i + half_group_size];

    Fp res_1 = a + w*b;
    Fp res_2 = a - w*b;

    input[i]                    = res_1; // --\/--
    input[i + half_group_size]  = res_2; // --/\--
}
