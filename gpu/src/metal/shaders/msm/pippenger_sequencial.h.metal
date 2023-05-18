#pragma once


// NOTE: 
//   - the number of threadgroups dispatched needs to equal the number of windows.
//   - `cs` and `hidings` need to have a minimum length of `buflen`
//   - `group_buckets` needs to have a minimum length of `group_size` * (2^`window_size` - 1)
//   - `result` needs to have a length equal to the number of threadgroups
template<typename Fp, typename ECPoint>
[[kernel]] void calculate_Gjs_sequencial(
    constant const Fp* cs [[ buffer(0) ]],
    constant const ECPoint* hidings [[ buffer(1) ]],
    constant const uint32_t& _window_size [[ buffer(2) ]],
    constant const uint64_t& _buflen [[ buffer(3) ]],
    device ECPoint* result [[ buffer(4) ]]
) {
    //uint32_t window_size = _window_size;

    //uint32_t windows_mask = (1 << window_size) - 1;
    // const uint32_t bucket_count = windows_mask; TODO calculate bucket_count dynamically

    metal::array<ECPoint, 15> buckets {};

    for(int i = 0; i < 15; i++){
        result[i] = buckets[i];
    }
}

// TODO: perform reduction to sum result in parallel (in a different kernel)
