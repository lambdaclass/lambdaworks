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
    device ECPoint* buckets [[ buffer(0) ]],
    constant const uint64_t& _buckets_len [[ buffer(1) ]],
    device ECPoint* partial_sums [[ buffer(2) ]],
    device ECPoint& output [[ buffer(3) ]]
)
{
    uint64_t buckets_len = _buckets_len;

    partial_sums[0] = buckets[buckets_len - 1];
    for (uint64_t i = 1; i < buckets_len; i++) {
        ECPoint acc = partial_sums[i - 1];
        ECPoint bucket = buckets[buckets_len - i - 1];

        partial_sums[i] = acc + bucket;
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
    constant const uint64_t& _windows_len [[ buffer(1) ]],
    device ECPoint& output [[ buffer(2) ]]
)
{
    uint64_t windows_len = _windows_len;

    ECPoint acc {};
    for (uint64_t i = 0; i < windows_len; i++) {
        ECPoint window = windows[i];
        acc += window * (1 << (4 * i));
    }

    output = acc;
}
