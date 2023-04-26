#pragma once

#include <metal_stdlib>
#include "fp_u256.h.metal"

template<typename Fp>
[[kernel]] void radix2_dit_butterfly(
    device Fp* input [[ buffer(0) ]],
    constant Fp* twiddles [[ buffer(1) ]],
    uint32_t group [[ threadgroup_position_in_grid ]],
    uint32_t pos_in_group [[ thread_position_in_threadgroup ]],
    uint32_t half_group_size [[ threads_per_threadgroup ]]
)
{
  uint32_t i = group * half_group_size * 2 + pos_in_group;

  Fp w = twiddles[group];
  Fp a = input[i];
  Fp b = input[i + half_group_size];

  Fp res_1 = a + w*b;
  Fp res_2 = a - w*b;

  input[i]                    = res_1; // --\/--
  input[i + half_group_size]  = res_2; // --/\--
}

