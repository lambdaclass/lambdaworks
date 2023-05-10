#pragma once

#include <metal_stdlib>

template<typename Fp, typename EcPoint>
[[kernel]] void pippenger(
    constant const Fp* cs [[ buffer(0) ]],
    constant const EcPoint* hidings [[ buffer(1) ]],
    // TODO: change device for threadgroup
    device EcPoint* buckets [[ buffer(2) ]],
    constant const uint32_t& _window_size [[ buffer(3) ]],
    constant const uint32_t& buflen [[ buffer(4) ]],
    device uint32_t* result [[ buffer(5) ]],
    uint32_t group [[ threadgroup_position_in_grid ]],
    uint32_t group_size [[ threads_per_threadgroup ]],
    uint32_t pos_in_group [[ thread_position_in_threadgroup ]]
)
{
    // TODO: parallelize inside a group
    if (pos_in_group > 0) {
        return;
    }
    uint32_t window_size = _window_size;
    // TODO: simply rename arguments
    uint32_t window_idx = group;
    uint32_t bucket_size = 1 << (window_size - 1);
    device EcPoint *group_buckets = buckets + (group * bucket_size);

    uint32_t windows_shl = window_size * window_idx;
    uint32_t windows_mask = (1 << window_size) - 1;

    for (uint32_t i = 0; i < buflen; i++) {
        uint32_t m_ij = (cs[i] >> windows_shl) & windows_mask;
        if (m_ij == 0) {
            continue;
        }
        EcPoint p = hidings[i];
        group_buckets[m_ij - 1] += p;
    }

    for (uint32_t m_ij = 0; m_ij < bucket_size; m_ij++) {
        group_buckets[m_ij] = (m_ij + 1) * group_buckets[m_ij];
    }

    // TODO: zero
    EcPoint sum;

    for (uint32_t m_ij = 0; m_ij < bucket_size; m_ij++) {
        sum += group_buckets[m_ij];
    }

    result[group] = sum;
}
