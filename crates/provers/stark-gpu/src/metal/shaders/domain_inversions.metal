// GPU batch domain inversions for DEEP composition (base field).
//
// Computes 4 inversion vectors in parallel using Fermat's little theorem:
//   1/(x_i - z^N), 1/(x_i - z*g^0), 1/(x_i - z*g^1), 1/(x_i - z*g^2)
//
// Each thread processes one domain point, performing 4 independent inversions.
// This replaces the sequential CPU `inplace_batch_inverse` calls.
//
// NOTE: The #include is stripped at runtime; FP_U64_HEADER_SOURCE is prepended.
// #include "fp_u64.h.metal"

#include <metal_stdlib>
using namespace metal;

struct DomainInvParams {
    uint32_t num_rows;
};

/// Compute 4 inversion vectors for DEEP composition.
/// Each thread processes one domain point x_i.
[[kernel]] void compute_domain_inversions(
    device const uint64_t* domain_points  [[ buffer(0) ]],
    constant uint64_t& z_power_raw        [[ buffer(1) ]],
    constant uint64_t& z_shifted_0_raw    [[ buffer(2) ]],
    constant uint64_t& z_shifted_1_raw    [[ buffer(3) ]],
    constant uint64_t& z_shifted_2_raw    [[ buffer(4) ]],
    device uint64_t* inv_z_power_out      [[ buffer(5) ]],
    device uint64_t* inv_z_shifted_0_out  [[ buffer(6) ]],
    device uint64_t* inv_z_shifted_1_out  [[ buffer(7) ]],
    device uint64_t* inv_z_shifted_2_out  [[ buffer(8) ]],
    constant DomainInvParams& params      [[ buffer(9) ]],
    uint tid [[ thread_position_in_grid ]]
) {
    if (tid >= params.num_rows) return;

    Fp64Goldilocks x(domain_points[tid]);
    Fp64Goldilocks zp(z_power_raw);
    Fp64Goldilocks zs0(z_shifted_0_raw);
    Fp64Goldilocks zs1(z_shifted_1_raw);
    Fp64Goldilocks zs2(z_shifted_2_raw);

    inv_z_power_out[tid]     = (x - zp).inverse().canonicalize();
    inv_z_shifted_0_out[tid] = (x - zs0).inverse().canonicalize();
    inv_z_shifted_1_out[tid] = (x - zs1).inverse().canonicalize();
    inv_z_shifted_2_out[tid] = (x - zs2).inverse().canonicalize();
}

// =============================================================================
// Batch Montgomery inversion kernel
// =============================================================================
//
// Uses Montgomery's batch inversion trick: given N elements to invert,
// compute all N inverses using only 1 Fermat inversion + 3(N-1) multiplications.
//
// Each thread processes a chunk of CHUNK_SIZE consecutive domain points
// across all 4 z-offsets. Per chunk per offset:
//   Forward pass:  d[k] = x[base+k] - z; prefix[0]=d[0]; prefix[k]=prefix[k-1]*d[k]
//   One inversion: inv = prefix[last].inverse()
//   Backward pass: out[last]=prefix[last-1]*inv; inv=inv*d[last]; ...
//
// For CHUNK_SIZE=16 per offset: 15 forward muls + 1 inversion (73 muls) + 30 backward muls = 118 muls
// vs 4*73 = 292 muls for 4 individual Fermat inversions per point.
// Net: 4 * 118 = 472 muls for 64 inversions (16 points * 4 offsets) per thread.

struct BatchDomainInvParams {
    uint32_t num_rows;
    uint32_t chunk_size;
};

[[kernel]] void batch_domain_inversions(
    device const uint64_t* domain_points     [[ buffer(0) ]],
    device const uint64_t* z_values          [[ buffer(1) ]],  // 4 packed z-values
    device uint64_t* inv_z_power_out         [[ buffer(2) ]],
    device uint64_t* inv_z_shifted_0_out     [[ buffer(3) ]],
    device uint64_t* inv_z_shifted_1_out     [[ buffer(4) ]],
    device uint64_t* inv_z_shifted_2_out     [[ buffer(5) ]],
    constant BatchDomainInvParams& params    [[ buffer(6) ]],
    uint tid [[ thread_position_in_grid ]]
) {
    uint32_t base_idx = tid * params.chunk_size;
    if (base_idx >= params.num_rows) return;

    uint32_t chunk_len = metal::min(params.chunk_size, params.num_rows - base_idx);

    // Load z-values once
    Fp64Goldilocks z[4] = {
        Fp64Goldilocks(z_values[0]),
        Fp64Goldilocks(z_values[1]),
        Fp64Goldilocks(z_values[2]),
        Fp64Goldilocks(z_values[3])
    };

    // Output buffer pointers (indexed by offset)
    device uint64_t* out_ptrs[4] = {
        inv_z_power_out,
        inv_z_shifted_0_out,
        inv_z_shifted_1_out,
        inv_z_shifted_2_out
    };

    // Thread-local storage for differences and prefix products.
    // Max CHUNK_SIZE = 32 (reasonable for register pressure).
    Fp64Goldilocks diffs[32];
    Fp64Goldilocks prefix[32];

    // Process each z-offset independently using batch inversion
    for (uint32_t off = 0; off < 4; off++) {
        // Forward pass: compute differences and prefix products
        for (uint32_t k = 0; k < chunk_len; k++) {
            diffs[k] = Fp64Goldilocks(domain_points[base_idx + k]) - z[off];
        }
        prefix[0] = diffs[0];
        for (uint32_t k = 1; k < chunk_len; k++) {
            prefix[k] = prefix[k - 1] * diffs[k];
        }

        // One Fermat inversion for the entire chunk
        Fp64Goldilocks inv = prefix[chunk_len - 1].inverse();

        // Backward pass: extract individual inverses
        for (uint32_t k = chunk_len - 1; k > 0; k--) {
            out_ptrs[off][base_idx + k] = (prefix[k - 1] * inv).canonicalize();
            inv = inv * diffs[k];
        }
        out_ptrs[off][base_idx] = inv.canonicalize();
    }
}
