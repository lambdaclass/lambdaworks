#include <metal_stdlib>

#include "fp_u256.h.metal"
#include "fp.h.metal"

[[kernel]]
void radix2_dit_butterfly(
    device uint32_t* input [[ buffer(0) ]],
    constant uint32_t* twiddles [[ buffer(1) ]],
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

  input[i]             = (a + w*b).asUInt32(); // --\/--
  input[i + distance]  = (a - w*b).asUInt32(); // --/\--
}

[[kernel]]
void radix2_dit_butterfly_u256(
    device p256::Fp* input [[ buffer(0) ]],
    constant p256::Fp* twiddles [[ buffer(1) ]],
    constant uint32_t& group_size [[ buffer(2) ]],
    uint32_t pos_in_group [[ thread_position_in_threadgroup ]],
    uint32_t group [[ threadgroup_position_in_grid ]],
    uint32_t global_tid [[ thread_position_in_grid ]]
)
{
  uint32_t i = group * group_size + pos_in_group;
  uint32_t distance = group_size / 2;

  p256::Fp w = twiddles[group];
  p256::Fp a = input[i];
  p256::Fp b = input[i + distance];

  p256::Fp res_1 = a + w*b;
  p256::Fp res_2 = a - w*b;

  input[i]             = res_1; // --\/--
  input[i + distance]  = res_2; // --/\--
}

[[kernel]]
void calc_twiddle(
    constant uint32_t& _omega [[ buffer(0) ]],
    device uint32_t* result  [[ buffer(1) ]],
    uint index [[ thread_position_in_grid ]]
)
{
    Fp omega = _omega;
    result[index] = pow(omega, index).asUInt32();
}

[[kernel]]
void calc_twiddle_u256(
    constant p256::Fp& _omega [[ buffer(0) ]],
    device p256::Fp* result  [[ buffer(1) ]],
    uint index [[ thread_position_in_grid ]]
)
{
    p256::Fp omega = _omega;
    result[index] = omega.pow(index);
}