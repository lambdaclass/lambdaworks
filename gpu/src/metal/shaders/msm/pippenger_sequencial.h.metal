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
    device Point* buckets                 [[ buffer(3) ]],
    const uint32_t iter_id      [[ thread_position_in_grid ]]
)
{
    constexpr uint32_t WINDOW_SIZE = 4; // set to this for now
    constexpr uint32_t NUM_LIMBS = 12;  // u384

    uint32_t window_idx = _window_idx;
    u384 k = _k[iter_id];
    Point p = _p[iter_id];

    uint32_t window_unmasked = (k >> (window_idx * WINDOW_SIZE)).m_limbs[NUM_LIMBS - 1];
    uint32_t m_ij = window_unmasked & ((1 << WINDOW_SIZE) - 1);
    if (m_ij != 0) {
        uint64_t idx = (m_ij - 1);
        Point bucket = buckets[idx];
        buckets[idx] = bucket + p;
    }
}


// NOTE: 
//   - result should be size 2^s - 1
//   - result should be initialized with the point at infinity
template<typename Fp, typename ECPoint>
[[kernel]] void calculate_points_sum(
    constant const Fp* cs [[ buffer(0) ]],
    constant const ECPoint* hidings [[ buffer(1) ]],
    constant const uint32_t& _buflen [[ buffer(2) ]],
    constant const uint32_t& _window_idx [[ buffer(3) ]],
    device ECPoint* result [[ buffer(4) ]]
) {
    const uint32_t WINDOWS_SIZE = 4;
    uint32_t bucket_size = (1 << WINDOWS_SIZE) - 1;
    const uint32_t windows_idx = _window_idx;
    const uint32_t offset = WINDOWS_SIZE * windows_idx;
    uint64_t buflen = _buflen;

    for(uint32_t i = 0; i < buflen; i++){
        Fp k = cs[i];
        uint32_t m_ij = (uint32_t)(k >> offset) & bucket_size;
        if (m_ij != 0) {
            uint32_t bucket_index = m_ij - 1;
            ECPoint aux = result[bucket_index];
            ECPoint point = hidings[i];
            result[bucket_index] = aux + point;
        }
    }
}

template<typename ECPoint>
[[kernel]]
void calculate_window(
    device ECPoint* buckets [[ buffer(0) ]],
    constant const uint32_t& _buckets_len [[ buffer(1) ]],
    device ECPoint* partial_sums [[ buffer(2) ]],
    device ECPoint& output [[ buffer(3) ]]
)
{
    uint32_t buckets_len = _buckets_len;

    partial_sums[0] = buckets[buckets_len - 1];
    for (uint32_t i = 1; i < buckets_len; i++) {
        ECPoint acc = partial_sums[i - 1];
        ECPoint bucket = buckets[buckets_len - i - 1];

        partial_sums[i] = acc + bucket;
    }

    ECPoint acc {};
    for (uint32_t i = 0; i < buckets_len; i++) {
        acc += partial_sums[i];
    }

    output = acc;
}

template<typename ECPoint>
[[kernel]]
void reduce_windows(
    constant const ECPoint* windows [[ buffer(0) ]],
    constant const uint32_t& _windows_len [[ buffer(1) ]],
    device ECPoint& output [[ buffer(2) ]]
)
{
    uint32_t windows_len = _windows_len;

    ECPoint acc {};
    for (uint32_t i = 0; i < windows_len; i++) {
        ECPoint window = windows[i];
        acc += window * (1 << 4);
    }

    output = acc;
}
