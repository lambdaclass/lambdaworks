#include <metal_stdlib>
using namespace metal;

// GPU coefficient striding kernel for break_in_parts.
//
// Given N coefficients [c0, c1, c2, ..., c_{N-1}] and k = num_parts,
// produces k sub-arrays where part i contains:
//   [c_i, c_{i+k}, c_{i+2k}, ...]
//
// Output layout: k sub-arrays of N/k elements each, concatenated.
// part 0: output[0..N/k-1]
// part 1: output[N/k..2*N/k-1]
// ...
//
// Thread gid = output index.
// part = gid / part_len, idx = gid % part_len
// input index = idx * num_parts + part

struct StrideParams {
    uint32_t num_coeffs;   // N = total number of input coefficients
    uint32_t num_parts;    // k = number of parts to stride into
};

[[kernel]] void goldilocks_stride_coefficients(
    device const uint64_t* input   [[ buffer(0) ]],
    device uint64_t* output        [[ buffer(1) ]],
    constant StrideParams& params  [[ buffer(2) ]],
    uint gid                       [[ thread_position_in_grid ]]
) {
    if (gid >= params.num_coeffs) return;

    uint32_t part_len = params.num_coeffs / params.num_parts;
    uint32_t part = gid / part_len;
    uint32_t idx = gid % part_len;

    // Read from interleaved position
    output[gid] = input[idx * params.num_parts + part];
}
