#pragma once
#include "fp.h.metal"
#include "util.h.metal"
#include "fp_u256.h.metal"

template<typename Fp>
[[kernel]] void calc_twiddles(
    constant Fp& _omega [[ buffer(0) ]],
    device Fp* result  [[ buffer(1) ]],
    uint index [[ thread_position_in_grid ]]
)
{
    Fp omega = _omega;
    result[index] = omega.pow(index);
}

template [[ host_name("calc_twiddles") ]] 
[[kernel]] void calc_twiddles<p256::Fp>(
    constant p256::Fp&, 
    device p256::Fp*, 
    uint
);


[[kernel]]
void calc_twiddle_inv(
    device uint32_t* result  [[ buffer(0) ]],
    constant uint32_t& _omega [[ buffer(1) ]],
    uint index [[ thread_position_in_grid ]]
)
{
    Fp omega = _omega;
    result[index] = inv(pow(omega, index)).asUInt32();
}

[[kernel]]
void calc_twiddle_bitrev(
    device uint32_t* result  [[ buffer(0) ]],
    constant uint32_t& _omega [[ buffer(1) ]],
    uint index [[ thread_position_in_grid ]],
    uint size [[ threads_per_grid ]]
)
{
    Fp omega = _omega;
    result[index] = pow(omega, reverse_index(index, size)).asUInt32();
}

[[kernel]]
void calc_twiddle_bitrev_inv(
    device uint32_t* result  [[ buffer(0) ]],
    constant uint32_t& _omega [[ buffer(1) ]],
    uint index [[ thread_position_in_grid ]],
    uint size [[ threads_per_grid ]]
)
{
    Fp omega = _omega;
    result[index] = inv(pow(omega, reverse_index(index, size))).asUInt32();
}
