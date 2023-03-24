#include <metal_stdlib>
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

/// Reverses the `log2(size)` first bits of `i`
uint32_t reverse_index(uint32_t i, uint64_t size) {
    if (size == 1) { // TODO: replace this statement with an alternative solution.
        return i;
    } else {
        return metal::reverse_bits(i) >> (32 - metal::ctz(size));
    }
}

[[kernel]]
void calc_twiddle(
    device uint32_t* result  [[ buffer(0) ]],
    constant uint32_t& _omega [[ buffer(1) ]],
    uint index [[ thread_position_in_grid ]]
)
{
    Fp omega = _omega;
    result[index] = pow(omega, index).asUInt32();
}

[[kernel]]
void calc_twiddle_inv(
    device uint32_t* result  [[ buffer(0) ]],
    constant uint32_t& _omega [[ buffer(1) ]],
    uint index [[ thread_position_in_grid ]],
    uint size [[ threads_per_grid ]]
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
