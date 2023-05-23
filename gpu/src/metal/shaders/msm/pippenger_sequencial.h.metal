#pragma once
#include "../fields/bls12381.h.metal"
#include "../fields/fp_bls12381.h.metal"
#include "../fields/unsigned_int.h.metal"

namespace {
    typedef UnsignedInteger<12> u384;
    typedef FpBLS12381 FE;
    typedef ECPoint<FE, 0> Point;
}

[[kernel]] void org_buckets(
    constant const uint32_t& _window_idx  [[ buffer(0) ]],
    constant const u384* _k               [[ buffer(1) ]],
    constant const Point* _p              [[ buffer(2) ]],
    device Point* buckets_matrix          [[ buffer(3) ]],
    const uint32_t thread_id      [[ thread_position_in_grid ]],
    const uint32_t thread_count   [[ threads_per_grid ]]
)
{
    constexpr uint32_t WINDOW_SIZE = 4; // set to this for now
    constexpr uint32_t NUM_LIMBS = 12;  // u384

    uint32_t window_idx = _window_idx;
    u384 k = _k[thread_id];
    Point p = _p[thread_id];

    uint32_t buckets_len = (1 << WINDOW_SIZE) - 1;

    uint32_t window_unmasked = (k >> (window_idx * WINDOW_SIZE)).m_limbs[NUM_LIMBS - 1];
    uint32_t m_ij = window_unmasked & ((1 << WINDOW_SIZE) - 1);
    if (m_ij != 0) {
        uint64_t idx = (m_ij - 1);
        Point bucket = buckets_matrix[thread_id * buckets_len + idx];
        buckets_matrix[thread_id * buckets_len + idx] = bucket + p;
    }
}
