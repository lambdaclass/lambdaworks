#pragma once

#include <metal_stdlib>
#include "fp_u256.h.metal"

template<typename Fp>
[[kernel]] void radix2_dit_butterfly(
    device Fp* input [[ buffer(0) ]],
    constant Fp* twiddles [[ buffer(1) ]],
    constant uint32_t& group_size [[ buffer(2) ]],
    uint32_t pos_in_group [[ thread_position_in_threadgroup ]],
    uint32_t group [[ threadgroup_position_in_grid ]],
    uint32_t global_tid [[ thread_position_in_grid ]]
)
{
  uint32_t i = group * group_size + pos_in_group;
  uint32_t distance = group_size / 2;

  Fp w = twiddles[group];
  Fp a = input[i];
  Fp b = input[i + distance];

  Fp res_1 = a + w*b;
  Fp res_2 = a - w*b;

  input[i]             = res_1; // --\/--
  input[i + distance]  = res_2; // --/\--
}

template [[ host_name("radix2_dit_butterfly") ]] 
[[kernel]] void radix2_dit_butterfly<p256::Fp>(
    device p256::Fp*, 
    constant p256::Fp*, 
    constant uint32_t&, 
    uint32_t, 
    uint32_t, 
    uint32_t
);