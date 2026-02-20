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
