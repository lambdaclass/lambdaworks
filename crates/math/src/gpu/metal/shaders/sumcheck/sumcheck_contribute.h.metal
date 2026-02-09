// Sumcheck contribution kernel: computes round polynomial evaluations.
//
// For a given evaluation point t, each thread handles one index j in [0, half),
// computing: prod_{k=0}^{num_factors-1} (table_k[j] + t * (table_k[j+half] - table_k[j]))
// Then a threadgroup tree-reduction sums contributions within each threadgroup.
//
// Output: one partial sum per threadgroup.

#ifndef sumcheck_contribute_h
#define sumcheck_contribute_h

#include <metal_stdlib>
using namespace metal;

template <typename Fp>
[[kernel]] void sumcheck_contribute(
    device const Fp* tables       [[buffer(0)]],  // Interleaved: factor0[0..table_len], factor1[0..table_len], ...
    device Fp* partial_sums       [[buffer(1)]],  // Output: one partial sum per threadgroup
    constant uint32_t& half       [[buffer(2)]],  // Half of current table length
    constant uint32_t& table_len  [[buffer(3)]],  // Current table length per factor
    constant uint32_t& num_factors [[buffer(4)]], // Number of factor polynomials
    constant Fp& eval_point       [[buffer(5)]],  // Evaluation point t
    uint tid                      [[thread_position_in_grid]],
    uint lid                      [[thread_position_in_threadgroup]],
    uint gid                      [[threadgroup_position_in_grid]],
    uint tg_size                  [[threads_per_threadgroup]]
) {
    threadgroup Fp shared_mem[1024];

    Fp contribution = Fp::zero();

    if (tid < half) {
        contribution = Fp::one();
        for (uint32_t k = 0; k < num_factors; k++) {
            uint32_t base = k * table_len;
            Fp lo = tables[base + tid];
            Fp hi = tables[base + tid + half];
            Fp diff = hi - lo;
            Fp interpolated = lo + eval_point * diff;
            contribution = contribution * interpolated;
        }
    }

    shared_mem[lid] = contribution;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction within threadgroup
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (lid < s) {
            shared_mem[lid] = shared_mem[lid] + shared_mem[lid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0) {
        partial_sums[gid] = shared_mem[0];
    }
}

#endif /* sumcheck_contribute_h */
