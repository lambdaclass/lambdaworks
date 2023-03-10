#include <metal_stdlib>
#include "fp.h.metal"

[[kernel]]
void radix2_dit_butterfly(
    device uint32_t* input [[ buffer(0) ]]
    constant uint32_t* twiddles [[ buffer(1) ]],
    uint32_t group_count [[ threadgroups_per_grid ]],
    uint32_t i [[ thread_position_in_grid ]],
    uint32_t group [[ threadgroup_position_in_grid ]],
    uint32_t butterflies [[ grid_size ]] // one butterfly per thread.
)
{
  uint32_t distance = butterflies / group_count;

  Fp w = twiddles[group];
  Fp a = input[i];
  Fp b = input[i + distance];

  input[i]            = a + w*b // --\/--
  input[i + distance] = a - w*b // --/\--
}

kernel void gen_twiddles(
    constant uint32_t& _omega [[ buffer(0) ]],
    device uint32_t* result  [[ buffer(1) ]],
    uint index [[ thread_position_in_grid ]]
)
{
    Fp omega = _omega;
    result[index] = pow(omega, index).asUInt32();
}
