// Fp3 DEEP composition kernel for Goldilocks + degree-3 extension.
//
// NOTE: This shader is compiled at runtime by concatenating the Goldilocks
// field header (fp_u64.h.metal) and the Fp3 header (fp3_goldilocks.h.metal)
// before this source. Both Fp64Goldilocks and Fp3Goldilocks are expected to
// be already defined when this code is compiled.
//
// Hardcoded for Fibonacci RAP: 3 trace polys, 3 offsets, 1 composition part.
// Trace LDE values are in base field (1 u64 each).
// Inversions, gammas, OOD evals, and output are in Fp3 (3 u64s each).

struct DeepCompFp3Params {
    uint32_t num_rows;           // LDE domain size
    uint32_t num_trace_polys;    // 3 for Fibonacci RAP (2 main + 1 aux)
    uint32_t num_offsets;        // 3 for Fibonacci RAP
    uint32_t num_comp_parts;     // 1 for Fibonacci RAP
};

// Buffer layout:
// 0: main_col_0 LDE evals (base field, 1 u64 per point)
// 1: main_col_1 LDE evals (base field)
// 2: aux_col_0 LDE evals (base field)
// 3: comp_part_0 LDE evals (Fp3, 3 u64s per point)
// 4: inv_z_power (Fp3, pre-computed 1/(x_i - z^N) for each domain point)
// 5: inv_z_shifted_0 (Fp3, 1/(x_i - z*g^0) for each domain point)
// 6: inv_z_shifted_1 (Fp3, 1/(x_i - z*g^1))
// 7: inv_z_shifted_2 (Fp3, 1/(x_i - z*g^2))
// 8: scalars - packed array of Fp3 elements (3 u64s each):
//    [gamma_h0,                          -- composition gamma (1 Fp3)
//     gamma_t00, gamma_t01, gamma_t02,   -- trace gammas for poly 0 (3 Fp3s)
//     gamma_t10, gamma_t11, gamma_t12,   -- trace gammas for poly 1 (3 Fp3s)
//     gamma_t20, gamma_t21, gamma_t22,   -- trace gammas for poly 2 (3 Fp3s)
//     h_ood_0,                           -- composition OOD eval (1 Fp3)
//     t_ood_00, t_ood_01, t_ood_02,      -- trace OOD evals for poly 0 (3 Fp3s)
//     t_ood_10, t_ood_11, t_ood_12,      -- trace OOD for poly 1
//     t_ood_20, t_ood_21, t_ood_22]      -- trace OOD for poly 2
// 9: params
// 10: output (Fp3, 3 u64s per point)
[[kernel]] void deep_composition_fp3_eval(
    device const uint64_t* trace_poly_0    [[ buffer(0) ]],
    device const uint64_t* trace_poly_1    [[ buffer(1) ]],
    device const uint64_t* trace_poly_2    [[ buffer(2) ]],
    device const uint64_t* comp_part_0     [[ buffer(3) ]],  // Fp3: 3 u64s per point
    device const uint64_t* inv_z_power     [[ buffer(4) ]],  // Fp3: 3 u64s per point
    device const uint64_t* inv_z_shifted_0 [[ buffer(5) ]],  // Fp3: 3 u64s per point
    device const uint64_t* inv_z_shifted_1 [[ buffer(6) ]],
    device const uint64_t* inv_z_shifted_2 [[ buffer(7) ]],
    device const uint64_t* scalars         [[ buffer(8) ]],  // Fp3 gammas + OOD evals
    constant DeepCompFp3Params& params     [[ buffer(9) ]],
    device uint64_t* output                [[ buffer(10) ]], // Fp3: 3 u64s per point
    uint tid                               [[ thread_position_in_grid ]]
) {
    if (tid >= params.num_rows) return;

    // Helper to read a Fp3 from a buffer of packed u64 triples
    auto read_fp3 = [](device const uint64_t* buf, uint idx) -> Fp3Goldilocks {
        uint base = idx * 3;
        return Fp3Goldilocks(
            Fp64Goldilocks(buf[base]),
            Fp64Goldilocks(buf[base + 1]),
            Fp64Goldilocks(buf[base + 2])
        );
    };

    // Helper to read a scalar Fp3 from the scalars array (index = Fp3 element index)
    auto read_scalar = [&scalars](uint idx) -> Fp3Goldilocks {
        uint base = idx * 3;
        return Fp3Goldilocks(
            Fp64Goldilocks(scalars[base]),
            Fp64Goldilocks(scalars[base + 1]),
            Fp64Goldilocks(scalars[base + 2])
        );
    };

    // Unpack scalars (all Fp3)
    Fp3Goldilocks gamma_h0 = read_scalar(0);
    Fp3Goldilocks gamma_t00 = read_scalar(1);
    Fp3Goldilocks gamma_t01 = read_scalar(2);
    Fp3Goldilocks gamma_t02 = read_scalar(3);
    Fp3Goldilocks gamma_t10 = read_scalar(4);
    Fp3Goldilocks gamma_t11 = read_scalar(5);
    Fp3Goldilocks gamma_t12 = read_scalar(6);
    Fp3Goldilocks gamma_t20 = read_scalar(7);
    Fp3Goldilocks gamma_t21 = read_scalar(8);
    Fp3Goldilocks gamma_t22 = read_scalar(9);
    Fp3Goldilocks h_ood_0 = read_scalar(10);
    Fp3Goldilocks t_ood_00 = read_scalar(11);
    Fp3Goldilocks t_ood_01 = read_scalar(12);
    Fp3Goldilocks t_ood_02 = read_scalar(13);
    Fp3Goldilocks t_ood_10 = read_scalar(14);
    Fp3Goldilocks t_ood_11 = read_scalar(15);
    Fp3Goldilocks t_ood_12 = read_scalar(16);
    Fp3Goldilocks t_ood_20 = read_scalar(17);
    Fp3Goldilocks t_ood_21 = read_scalar(18);
    Fp3Goldilocks t_ood_22 = read_scalar(19);

    // Read base field trace LDE values (1 u64 each) and embed to Fp3
    Fp3Goldilocks tp0 = Fp3Goldilocks(Fp64Goldilocks(trace_poly_0[tid]));
    Fp3Goldilocks tp1 = Fp3Goldilocks(Fp64Goldilocks(trace_poly_1[tid]));
    Fp3Goldilocks tp2 = Fp3Goldilocks(Fp64Goldilocks(trace_poly_2[tid]));

    // Read Fp3 composition LDE value
    Fp3Goldilocks hp0 = read_fp3(comp_part_0, tid);

    // Read Fp3 pre-computed inversions
    Fp3Goldilocks inv_zp = read_fp3(inv_z_power, tid);
    Fp3Goldilocks inv_zs0 = read_fp3(inv_z_shifted_0, tid);
    Fp3Goldilocks inv_zs1 = read_fp3(inv_z_shifted_1, tid);
    Fp3Goldilocks inv_zs2 = read_fp3(inv_z_shifted_2, tid);

    // H terms: gamma_h[k] * (H_k(x) - H_k(z^N)) * inv(x - z^N)
    Fp3Goldilocks h_acc = gamma_h0 * (hp0 - h_ood_0) * inv_zp;

    // Trace terms: gamma_t[j][k] * (t_j(x) - t_j(z*g^k)) * inv(x - z*g^k)
    Fp3Goldilocks t_acc = Fp3Goldilocks::zero();
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

    // Write Fp3 result (3 u64s)
    Fp3Goldilocks result = h_acc + t_acc;
    uint base_out = tid * 3;
    output[base_out]     = (uint64_t)result.c0;
    output[base_out + 1] = (uint64_t)result.c1;
    output[base_out + 2] = (uint64_t)result.c2;
}
