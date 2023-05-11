#pragma once

#include <metal_stdlib>

// NOTE: 
//   - the number of threadgroups dispatched needs to equal the number of windows.
//   - `cs` and `hidings` need to have a minimum length of `buflen`
//   - `group_buckets` needs to have a minimum length of `group_size` * (2^`window_size` - 1)
//   - `result` needs to have a length equal to the number of threadgroups
template<typename Fp, typename EcPoint>
[[kernel]] void calculate_Gjs(
    constant const Fp* cs [[ buffer(0) ]],
    constant const EcPoint* hidings [[ buffer(1) ]],
    constant const uint32_t& _window_size [[ buffer(2) ]],
    constant const uint32_t& _buflen [[ buffer(3) ]],
    threadgroup EcPoint* group_buckets [[ threadgroup(0) ]],
    device EcPoint* result [[ buffer(4) ]],
    uint32_t group [[ threadgroup_position_in_grid ]],
    uint32_t group_size [[ threads_per_threadgroup ]],
    uint32_t pos_in_group [[ thread_position_in_threadgroup ]]
) {
    uint32_t window_size = _window_size;
    uint32_t buflen = _buflen;

    uint32_t buckets_size = 1 << (window_size - 1);

    uint32_t gbuckets_len = buckets_size * group_size;
    uint32_t gbuckets_tid = pos_in_group * buckets_size;

    uint32_t windows_shl = window_size * group;
    uint32_t windows_mask = (1 << window_size) - 1;

    // Calculate all Bij in parallel
    for (uint32_t i = pos_in_group; i < buflen; i += group_size) {
        uint32_t m_ij = (cs[i] >> windows_shl) & windows_mask;
        if (m_ij == 0) {
            continue;
        }
        uint32_t index = m_ij - 1 + gbuckets_tid;
        group_buckets[index] += hidings[i];
    }

    // Synchronize threadgroup memory
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    // We only use buckets_size threads at most here
    if (pos_in_group >= buckets_size) {
        return;
    }
    if (group_size > buckets_size) {
        group_size = buckets_size;
    }

    // Sum all Bij in parallel, splitting work between threads
    for (uint32_t i = pos_in_group; i < buckets_size; i += group_size) {
        for (uint32_t j = i + buckets_size; j < gbuckets_len; j += buckets_size) {
            group_buckets[pos_in_group] += group_buckets[j];
        }
        uint32_t m_ij = i + 1;
        group_buckets[pos_in_group] *= m_ij;
    }
    // Synchronize threadgroup memory
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    // TODO: parallelize using reduction
    if (pos_in_group != 0) {
        return;
    }

    for (uint32_t i = 1; i < buckets_size; i++) {
        group_buckets[0] += group_buckets[i];
    }

    result[group] = group_buckets[0];
}

// TODO: perform reduction to sum result in parallel (in a different kernel)
