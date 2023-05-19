#pragma once

// NOTE: 
//   - result should be size 2^s - 1
//   - result should be initialized with the point at infinity
template<typename Fp, typename ECPoint>
[[kernel]] void calculate_points_sum(
    constant const Fp* cs [[ buffer(0) ]],
    constant const ECPoint* hidings [[ buffer(1) ]],
    constant const uint64_t& _buflen [[ buffer(2) ]],
    constant const uint32_t& _window_idx [[ buffer(3) ]],
    device ECPoint* result [[ buffer(4) ]]
) {
    const uint32_t WINDOWS_SIZE = 4;
    uint32_t bucket_size = (1 << WINDOWS_SIZE) - 1;
    const uint32_t windows_idx = _window_idx;
    const uint32_t offset = WINDOWS_SIZE * windows_idx;
    uint64_t buflen = _buflen;

    for(uint64_t i = 0; i < buflen; i++){
        for(uint64_t j = 0; j < buflen; j++){
            Fp k = cs[j];
            uint32_t m_ij = (uint32_t)(k >> offset) & bucket_size;
            if (m_ij != 0) {
                uint32_t bucket_index = m_ij - 1;
                ECPoint aux = result[bucket_index];
                ECPoint point = hidings[i];
                result[bucket_index] = aux + point;
            }
        }
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
