#pragma once

// NOTE: 
//   - result should be size 2^s - 1
//   - result should be initialized with the point at infinity
template<typename ECPoint>
[[kernel]] void calculate_points_sum(
    constant const ECPoint* hidings [[ buffer(0) ]],
    constant const uint64_t& _buflen [[ buffer(1) ]],
    device ECPoint* result [[ buffer(2) ]]
) {
    const uint32_t WINDOWS_SIZE = 4;
    uint32_t bucket_size = (1 << WINDOWS_SIZE) - 1;

    uint64_t buflen = _buflen;
    uint32_t bucket_index;
    ECPoint point;
    ECPoint aux;
    for(uint64_t i = 0; i < buflen; i++){
        bucket_index = i / bucket_size;
        point = hidings[i];
        aux = result[bucket_index];
        result[bucket_index] = aux + point;
    }
}

template<typename ECPoint>
[[kernel]]
void calculate_window(
    constant const ECPoint* buckets [[ buffer(0) ]],
    constant const uint64_t* buckets_len [[ buffer(1) ]],
    constant const ECPoint& output [[ buffer(2) ]],

)
{
    metal::array<ECPoint, buckets_len> partial_sums {}; // b_0, b_0 + b_1, b_0 + b_1 + b_2...
    partial_sums[0] = buckets[0];
    for (uint64_t i = 1; i < buckets_len; i++) {
        partial_sums[i] = partial_sums[i - 1] + buckets[i];
    }

    ECPoint acc {};
    for (uint64_t i = 0; i < buckets_len; i++) {
        acc += partial_sums[i];
    }

    output = acc;
}

template<typename ECPoint>
[[kernel]]
void reduce_windows(
    constant const ECPoint* windows [[ buffer(0) ]],
    constant const uint64_t* windows_len [[ buffer(1) ]],
    constant const ECPoint& output [[ buffer(2) ]],

)
{
    ECPoint acc {};
    for (uint64_t i = 0; i < windows_len; i++) {
        acc += windows[i];
    }

    output = acc;
}
