// FRI fold kernel for Goldilocks Fp3 (degree-3 extension) field.
//
// NOTE: This shader is compiled at runtime by concatenating the Goldilocks
// field header (fp_u64.h.metal) and the Fp3 header (fp3_goldilocks.h.metal)
// before this source. Both Fp64Goldilocks and Fp3Goldilocks are expected to
// be already defined when this code is compiled.
//
// FRI fold operation in Fp3:
//   result[k] = 2 * (coeffs[2k] + beta * coeffs[2k+1])
// where coeffs, beta, and result are all in Fp3 (3 u64s each).

[[kernel]] void goldilocks_fp3_fri_fold(
    device const uint64_t* coeffs  [[ buffer(0) ]],  // Fp3 coefficients: 3 u64s each
    device uint64_t* result        [[ buffer(1) ]],   // Fp3 output: 3 u64s each
    device const uint64_t* beta_buf [[ buffer(2) ]],  // Single Fp3 element (3 u64s)
    constant uint& half_len        [[ buffer(3) ]],
    uint gid [[ thread_position_in_grid ]]
) {
    if (gid >= half_len) return;

    // Read beta (single Fp3 element)
    Fp3Goldilocks b = Fp3Goldilocks(
        Fp64Goldilocks(beta_buf[0]),
        Fp64Goldilocks(beta_buf[1]),
        Fp64Goldilocks(beta_buf[2])
    );

    // Read even and odd Fp3 coefficients
    uint even_base = (2 * gid) * 3;
    Fp3Goldilocks even = Fp3Goldilocks(
        Fp64Goldilocks(coeffs[even_base]),
        Fp64Goldilocks(coeffs[even_base + 1]),
        Fp64Goldilocks(coeffs[even_base + 2])
    );

    uint odd_base = (2 * gid + 1) * 3;
    Fp3Goldilocks odd = Fp3Goldilocks(
        Fp64Goldilocks(coeffs[odd_base]),
        Fp64Goldilocks(coeffs[odd_base + 1]),
        Fp64Goldilocks(coeffs[odd_base + 2])
    );

    // FRI fold: result = 2 * (even + beta * odd)
    Fp3Goldilocks two = Fp3Goldilocks(Fp64Goldilocks(2));
    Fp3Goldilocks folded = two * (even + b * odd);

    // Write result (3 u64s)
    uint out_base = gid * 3;
    result[out_base]     = (uint64_t)folded.c0;
    result[out_base + 1] = (uint64_t)folded.c1;
    result[out_base + 2] = (uint64_t)folded.c2;
}
