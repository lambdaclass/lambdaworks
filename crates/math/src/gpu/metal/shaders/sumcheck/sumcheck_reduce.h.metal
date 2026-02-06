// Sumcheck reduction kernel: reduces partial sums to a single value.
//
// Standard parallel reduction pattern. Single threadgroup launch.

#ifndef sumcheck_reduce_h
#define sumcheck_reduce_h

#include <metal_stdlib>
using namespace metal;

template <typename Fp>
[[kernel]] void sumcheck_reduce(
    device Fp* data          [[buffer(0)]],  // Input partial sums, output in data[0]
    constant uint32_t& count [[buffer(1)]],  // Number of partial sums
    uint lid                 [[thread_position_in_threadgroup]],
    uint tg_size             [[threads_per_threadgroup]]
) {
    threadgroup Fp shared_mem[1024];

    // Load into shared memory
    shared_mem[lid] = (lid < count) ? data[lid] : Fp::zero();
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (lid < s) {
            shared_mem[lid] = shared_mem[lid] + shared_mem[lid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0) {
        data[0] = shared_mem[0];
    }
}

#endif /* sumcheck_reduce_h */
