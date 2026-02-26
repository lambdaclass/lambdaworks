// GPU batch domain inversions for DEEP composition (Fp3 extension field).
//
// Computes 4 inversion vectors in Fp3 (degree-3 Goldilocks extension):
//   1/(x_i - z^N), 1/(x_i - z*g^0), 1/(x_i - z*g^1), 1/(x_i - z*g^2)
//
// Domain points x_i are base field; z-values are Fp3.
// Subtraction is base_field - Fp3 -> Fp3, then Fp3::inverse().
//
// NOTE: Headers (fp_u64.h.metal + fp3_goldilocks.h.metal) are concatenated
// at runtime via combined_fp3_source(). Do NOT #include them here.

#include <metal_stdlib>
using namespace metal;

struct DomainInvFp3Params {
    uint32_t num_rows;
};

/// Read an Fp3 value from a buffer of 3 consecutive u64s.
Fp3Goldilocks read_fp3_param(constant uint64_t* buf, uint offset) {
    return Fp3Goldilocks(
        Fp64Goldilocks(buf[offset]),
        Fp64Goldilocks(buf[offset + 1]),
        Fp64Goldilocks(buf[offset + 2])
    );
}

/// Write an Fp3 value to 3 consecutive u64s in an output buffer.
void write_fp3(device uint64_t* buf, uint base, Fp3Goldilocks val) {
    buf[base]     = val.c0.canonicalize();
    buf[base + 1] = val.c1.canonicalize();
    buf[base + 2] = val.c2.canonicalize();
}

/// Compute 4 Fp3 inversion vectors for DEEP composition.
/// Domain points are base field, z-values are Fp3.
/// Each thread processes one domain point.
[[kernel]] void compute_domain_inversions_fp3(
    device const uint64_t* domain_points     [[ buffer(0) ]],
    constant uint64_t* z_power_fp3           [[ buffer(1) ]],  // 3 u64s
    constant uint64_t* z_shifted_0_fp3       [[ buffer(2) ]],  // 3 u64s
    constant uint64_t* z_shifted_1_fp3       [[ buffer(3) ]],  // 3 u64s
    constant uint64_t* z_shifted_2_fp3       [[ buffer(4) ]],  // 3 u64s
    device uint64_t* inv_z_power_out         [[ buffer(5) ]],  // 3 u64s per element
    device uint64_t* inv_z_shifted_0_out     [[ buffer(6) ]],
    device uint64_t* inv_z_shifted_1_out     [[ buffer(7) ]],
    device uint64_t* inv_z_shifted_2_out     [[ buffer(8) ]],
    constant DomainInvFp3Params& params      [[ buffer(9) ]],
    uint tid [[ thread_position_in_grid ]]
) {
    if (tid >= params.num_rows) return;

    // Embed base field domain point into Fp3
    Fp64Goldilocks x_bf(domain_points[tid]);
    Fp3Goldilocks x(x_bf);

    Fp3Goldilocks zp  = read_fp3_param(z_power_fp3, 0);
    Fp3Goldilocks zs0 = read_fp3_param(z_shifted_0_fp3, 0);
    Fp3Goldilocks zs1 = read_fp3_param(z_shifted_1_fp3, 0);
    Fp3Goldilocks zs2 = read_fp3_param(z_shifted_2_fp3, 0);

    Fp3Goldilocks r0 = (x - zp).inverse();
    Fp3Goldilocks r1 = (x - zs0).inverse();
    Fp3Goldilocks r2 = (x - zs1).inverse();
    Fp3Goldilocks r3 = (x - zs2).inverse();

    uint out_base = tid * 3;
    write_fp3(inv_z_power_out,     out_base, r0);
    write_fp3(inv_z_shifted_0_out, out_base, r1);
    write_fp3(inv_z_shifted_1_out, out_base, r2);
    write_fp3(inv_z_shifted_2_out, out_base, r3);
}
