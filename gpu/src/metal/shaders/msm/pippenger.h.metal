#pragma once

#include <metal_stdlib>

// NOTE: 
//   - the number of threadgroups dispatched needs to equal the number of windows.
//   - `cs` and `hidings` need to have a minimum length of `buflen`
//   - `group_buckets` needs to have a minimum length of `group_size` * (2^`window_size` - 1)
//   - `result` needs to have a length equal to the number of threadgroups
template<typename Fp, typename ECPoint>
[[kernel]] void calculate_Gjs(
    constant const Fp* cs [[ buffer(0) ]],
    constant const ECPoint* hidings [[ buffer(1) ]],
    constant const uint32_t& _window_size [[ buffer(2) ]],
    constant const uint64_t& _buflen [[ buffer(3) ]],
    threadgroup ECPoint* group_buckets [[ threadgroup(0) ]],
    device ECPoint* result [[ buffer(4) ]],
    uint32_t group [[ threadgroup_position_in_grid ]],
    uint32_t group_size [[ threads_per_threadgroup ]],
    uint32_t pos_in_group [[ thread_position_in_threadgroup ]]
) {
    uint32_t window_size = _window_size;
    uint64_t buflen = _buflen;

    uint32_t windows_mask = (1 << window_size) - 1;
    uint32_t buckets_size = windows_mask;

    uint32_t gbuckets_len = windows_mask * group_size;
    uint32_t gbuckets_tid = windows_mask * pos_in_group;

    uint32_t windows_shl = window_size * group;

    // Calculate all Bij in parallel
    for (uint64_t i = pos_in_group; i < buflen; i += group_size) {
        Fp fp = cs[i];
        uint32_t w = fp >> windows_shl;
        uint32_t m_ij = w & windows_mask;
        if (m_ij == 0) {
            continue;
        }
        uint32_t index = m_ij - 1 + gbuckets_tid;
        ECPoint point = group_buckets[index];
        point += hidings[i];
        group_buckets[index] = point;
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
        ECPoint sum = group_buckets[pos_in_group];

        for (uint32_t j = i + buckets_size; j < gbuckets_len; j += buckets_size) {
            sum += group_buckets[j];
        }

        uint32_t m_ij = i + 1;
        sum *= m_ij;
        group_buckets[pos_in_group] = sum;
    }
    // Synchronize threadgroup memory
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    // TODO: parallelize using reduction
    if (pos_in_group != 0) {
        return;
    }

    ECPoint sum;

    for (uint32_t i = 1; i < buckets_size; i++) {
        sum += group_buckets[i];
    }

    result[group] = sum;
}

// TODO: perform reduction to sum result in parallel (in a different kernel)
