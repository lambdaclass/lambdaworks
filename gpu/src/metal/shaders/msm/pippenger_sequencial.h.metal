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
