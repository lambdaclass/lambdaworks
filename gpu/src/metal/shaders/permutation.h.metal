#pragma once
#include "util.h.metal"
#include "fp_u256.h.metal"

template<typename Fp>
[[kernel]] void bitrev_permutation(
    device Fp* input [[ buffer(0) ]],
    device Fp* result [[ buffer(1) ]],
    uint index [[ thread_position_in_grid ]],
    uint size [[ threads_per_grid ]]
)
{
    result[index] = input[reverse_index(index, size)];
}

template [[ host_name("bitrev_permutation") ]] 
[[kernel]] void bitrev_permutation<p256::Fp>(
    device p256::Fp*, 
    device p256::Fp*, 
    uint, 
    uint
);
