#include <metal_stdlib>
using namespace metal;

// Goldilocks field (same as in fibonacci_rap_constraints.metal)
constant uint64_t GL_EPSILON = 0xFFFFFFFF;
constant uint64_t GL_PRIME = 0xFFFFFFFF00000001UL;

struct GlFp {
    uint64_t inner;

    GlFp() = default;
    constexpr GlFp(uint64_t v) : inner(v) {}

    static GlFp zero() { return GlFp(0); }

    GlFp operator+(const GlFp rhs) const {
        uint64_t sum = inner + rhs.inner;
        bool over = sum < inner;
        uint64_t sum2 = sum + (over ? GL_EPSILON : 0);
        bool over2 = over && (sum2 < sum);
        return GlFp(sum2 + (over2 ? GL_EPSILON : 0));
    }

    GlFp operator-(const GlFp rhs) const {
        uint64_t diff = inner - rhs.inner;
        bool under = inner < rhs.inner;
        uint64_t diff2 = diff - (under ? GL_EPSILON : 0);
        bool under2 = under && (diff2 > diff);
        return GlFp(diff2 - (under2 ? GL_EPSILON : 0));
    }

    GlFp operator*(const GlFp rhs) const {
        uint64_t lo = inner * rhs.inner;
        uint64_t hi = metal::mulhi(inner, rhs.inner);
        return reduce128(lo, hi);
    }

private:
    static GlFp reduce128(uint64_t lo, uint64_t hi) {
        uint64_t hi_hi = hi >> 32;
        uint64_t hi_lo = hi & GL_EPSILON;
        uint64_t t0 = lo - hi_hi;
        bool borrow = lo < hi_hi;
        t0 = borrow ? t0 - GL_EPSILON : t0;
        uint64_t t1 = (hi_lo << 32) - hi_lo;
        uint64_t result = t0 + t1;
        bool carry = result < t0;
        result = carry ? result + GL_EPSILON : result;
        return GlFp(result);
    }
};

// Parameters for the DEEP composition kernel.
// For Fibonacci RAP: 3 trace polys, 3 offsets, 1 composition part.
struct DeepCompParams {
    uint32_t num_rows;           // LDE domain size
    uint32_t num_trace_polys;    // 3 for Fibonacci RAP (2 main + 1 aux)
    uint32_t num_offsets;        // 3 for Fibonacci RAP
    uint32_t num_comp_parts;     // 1 for Fibonacci RAP
};

// Hardcoded for Fibonacci RAP: 3 trace polys, 3 offsets, 1 composition part.
//
// Buffer layout:
// 0: main_col_0 LDE evals (trace poly 0)
// 1: main_col_1 LDE evals (trace poly 1)
// 2: aux_col_0 LDE evals (trace poly 2)
// 3: comp_part_0 LDE evals
// 4: inv_z_power (pre-computed 1/(x_i - z^N) for each domain point)
// 5: inv_z_shifted_0 (1/(x_i - z*g^0) for each domain point)
// 6: inv_z_shifted_1 (1/(x_i - z*g^1) for each domain point)
// 7: inv_z_shifted_2 (1/(x_i - z*g^2) for each domain point)
// 8: scalars - packed array:
//    [gamma_h0,                          -- composition gamma (1)
//     gamma_t00, gamma_t01, gamma_t02,   -- trace gammas for poly 0 (3 offsets)
//     gamma_t10, gamma_t11, gamma_t12,   -- trace gammas for poly 1 (3 offsets)
//     gamma_t20, gamma_t21, gamma_t22,   -- trace gammas for poly 2 (3 offsets)
//     h_ood_0,                           -- composition OOD eval (1)
//     t_ood_00, t_ood_01, t_ood_02,      -- trace OOD evals for poly 0 (3 offsets)
//     t_ood_10, t_ood_11, t_ood_12,      -- trace OOD for poly 1
//     t_ood_20, t_ood_21, t_ood_22]      -- trace OOD for poly 2
// 9: params
// 10: output
[[kernel]] void deep_composition_eval(
    device const GlFp* trace_poly_0    [[ buffer(0) ]],
    device const GlFp* trace_poly_1    [[ buffer(1) ]],
    device const GlFp* trace_poly_2    [[ buffer(2) ]],
    device const GlFp* comp_part_0     [[ buffer(3) ]],
    device const GlFp* inv_z_power     [[ buffer(4) ]],
    device const GlFp* inv_z_shifted_0 [[ buffer(5) ]],
    device const GlFp* inv_z_shifted_1 [[ buffer(6) ]],
    device const GlFp* inv_z_shifted_2 [[ buffer(7) ]],
    device const GlFp* scalars         [[ buffer(8) ]],  // gammas + OOD evals
    constant DeepCompParams& params    [[ buffer(9) ]],
    device GlFp* output                [[ buffer(10) ]],
    uint tid                           [[ thread_position_in_grid ]]
) {
    if (tid >= params.num_rows) return;

    // Unpack scalars
    // Composition gamma: scalars[0]
    GlFp gamma_h0 = scalars[0];
    // Trace gammas: scalars[1..10] (3 polys * 3 offsets)
    GlFp gamma_t00 = scalars[1];
    GlFp gamma_t01 = scalars[2];
    GlFp gamma_t02 = scalars[3];
    GlFp gamma_t10 = scalars[4];
    GlFp gamma_t11 = scalars[5];
    GlFp gamma_t12 = scalars[6];
    GlFp gamma_t20 = scalars[7];
    GlFp gamma_t21 = scalars[8];
    GlFp gamma_t22 = scalars[9];
    // Composition OOD: scalars[10]
    GlFp h_ood_0 = scalars[10];
    // Trace OOD: scalars[11..20] (3 polys * 3 offsets)
    GlFp t_ood_00 = scalars[11];
    GlFp t_ood_01 = scalars[12];
    GlFp t_ood_02 = scalars[13];
    GlFp t_ood_10 = scalars[14];
    GlFp t_ood_11 = scalars[15];
    GlFp t_ood_12 = scalars[16];
    GlFp t_ood_20 = scalars[17];
    GlFp t_ood_21 = scalars[18];
    GlFp t_ood_22 = scalars[19];

    // Read LDE values at this point
    GlFp tp0 = trace_poly_0[tid];
    GlFp tp1 = trace_poly_1[tid];
    GlFp tp2 = trace_poly_2[tid];
    GlFp hp0 = comp_part_0[tid];

    // Read pre-computed inversions
    GlFp inv_zp = inv_z_power[tid];
    GlFp inv_zs0 = inv_z_shifted_0[tid];
    GlFp inv_zs1 = inv_z_shifted_1[tid];
    GlFp inv_zs2 = inv_z_shifted_2[tid];

    // H terms: gamma_h[k] * (H_k(x) - H_k(z^N)) * inv(x - z^N)
    GlFp h_acc = gamma_h0 * (hp0 - h_ood_0) * inv_zp;

    // Trace terms: gamma_t[j][k] * (t_j(x) - t_j(z*g^k)) * inv(x - z*g^k)
    GlFp t_acc = GlFp::zero();
    // Poly 0 (main col a), offsets 0,1,2
    t_acc = t_acc + gamma_t00 * (tp0 - t_ood_00) * inv_zs0;
    t_acc = t_acc + gamma_t01 * (tp0 - t_ood_01) * inv_zs1;
    t_acc = t_acc + gamma_t02 * (tp0 - t_ood_02) * inv_zs2;
    // Poly 1 (main col b), offsets 0,1,2
    t_acc = t_acc + gamma_t10 * (tp1 - t_ood_10) * inv_zs0;
    t_acc = t_acc + gamma_t11 * (tp1 - t_ood_11) * inv_zs1;
    t_acc = t_acc + gamma_t12 * (tp1 - t_ood_12) * inv_zs2;
    // Poly 2 (aux col z), offsets 0,1,2
    t_acc = t_acc + gamma_t20 * (tp2 - t_ood_20) * inv_zs0;
    t_acc = t_acc + gamma_t21 * (tp2 - t_ood_21) * inv_zs1;
    t_acc = t_acc + gamma_t22 * (tp2 - t_ood_22) * inv_zs2;

    output[tid] = h_acc + t_acc;
}
