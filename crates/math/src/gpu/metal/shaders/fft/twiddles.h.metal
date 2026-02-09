// Twiddle factor generation kernels for Metal FFT.
//
// Twiddle factors are powers of the primitive root of unity used in FFT.
// These kernels generate twiddles in parallel on the GPU.

#pragma once
#include "util.h.metal"

/// Generate twiddle factors in natural order.
///
/// Computes result[i] = omega^i for i in [0, size).
/// omega is the primitive root of unity of order 2*size.
///
/// Parameters:
/// - result: Output array for twiddle factors
/// - _omega: Primitive root of unity
/// - index: Thread's position in grid
template<typename Fp>
[[kernel]] void calc_twiddles(
    device Fp* result  [[ buffer(0) ]],
    constant Fp& _omega [[ buffer(1) ]],
    uint index [[ thread_position_in_grid ]]
)
{
    Fp omega = _omega;
    result[index] = omega.pow(index);
}

/// Generate inverse twiddle factors in natural order.
///
/// Computes result[i] = omega^(-i) for i in [0, size).
///
/// Parameters:
/// - result: Output array for twiddle factors
/// - _omega: Primitive root of unity
/// - index: Thread's position in grid
template<typename Fp>
[[kernel]] void calc_twiddles_inv(
    device Fp* result  [[ buffer(0) ]],
    constant Fp& _omega [[ buffer(1) ]],
    uint index [[ thread_position_in_grid ]]
)
{
    Fp omega = _omega;
    result[index] = omega.pow(index).inverse();
}

/// Generate twiddle factors in bit-reversed order.
///
/// Computes result[i] = omega^(bit_reverse(i)) for i in [0, size).
/// This ordering is used for in-place FFT algorithms.
///
/// Parameters:
/// - result: Output array for twiddle factors
/// - _omega: Primitive root of unity
/// - index: Thread's position in grid
/// - size: Total number of twiddle factors
template<typename Fp>
[[kernel]] void calc_twiddles_bitrev(
    device Fp* result  [[ buffer(0) ]],
    constant Fp& _omega [[ buffer(1) ]],
    uint index [[ thread_position_in_grid ]],
    uint size [[ threads_per_grid ]]
)
{
    Fp omega = _omega;
    result[index] = omega.pow(reverse_index(index, size));
}

/// Generate inverse twiddle factors in bit-reversed order.
///
/// Computes result[i] = omega^(-bit_reverse(i)) for i in [0, size).
///
/// Parameters:
/// - result: Output array for twiddle factors
/// - _omega: Primitive root of unity
/// - index: Thread's position in grid
/// - size: Total number of twiddle factors
template<typename Fp>
[[kernel]] void calc_twiddles_bitrev_inv(
    device Fp* result  [[ buffer(0) ]],
    constant Fp& _omega [[ buffer(1) ]],
    uint index [[ thread_position_in_grid ]],
    uint size [[ threads_per_grid ]]
)
{
    Fp omega = _omega;
    result[index] = omega.pow(reverse_index(index, size)).inverse();
}
