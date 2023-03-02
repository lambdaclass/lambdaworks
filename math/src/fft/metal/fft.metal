#include <metal_stdlib>
#include "fp.h.metal"

kernel void gen_twiddles(
    constant uint32_t& _omega [[ buffer(0) ]],
    device uint32_t* result  [[ buffer(1) ]],
    uint index [[ thread_position_in_grid ]]
)
{
    Fp omega = _omega;
    result[index] = pow(omega, index).asUInt32();
}
