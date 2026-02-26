// GPU FRI domain inverse precomputation for eval-domain fold.
//
// Computes inv_x[i] = h_inv * omega_inv^{bitrev(i)} for i = 0..half_len-1.
// These are the inverses of the first half of the coset domain points
// in bit-reversed order: x_{bitrev(i)} = h * omega^{bitrev(i)}.
//
// Since h_inv and omega_inv are known constants, this avoids expensive
// per-element Fermat inversions entirely — just multiplications.
//
// For subsequent FRI layers, use the squaring kernel instead:
// inv_x_next[i] = inv_x[i]^2  (since 1/x^2 = (1/x)^2).
//
// NOTE: fp_u64.h.metal is concatenated at runtime. Do NOT #include it.

struct FriDomainInvParams {
    uint32_t half_len;
    uint32_t log_half_len;
};

// Bit-reverse an index with `bits` significant bits.
uint bitrev(uint idx, uint bits) {
    uint r = 0;
    for (uint b = 0; b < bits; b++) {
        r = (r << 1) | (idx & 1);
        idx >>= 1;
    }
    return r;
}

kernel void compute_fri_domain_inverses(
    constant Fp64Goldilocks& h_inv_ref      [[ buffer(0) ]],
    constant Fp64Goldilocks& omega_inv_ref  [[ buffer(1) ]],
    device Fp64Goldilocks* inv_x            [[ buffer(2) ]],
    constant FriDomainInvParams& params     [[ buffer(3) ]],
    uint gid [[ thread_position_in_grid ]]
) {
    if (gid >= params.half_len) return;

    // Copy from constant address space to thread-local
    Fp64Goldilocks h_inv = h_inv_ref;
    Fp64Goldilocks omega_inv = omega_inv_ref;

    // Compute omega_inv^{bitrev(gid)} via repeated squaring
    uint exp = bitrev(gid, params.log_half_len);
    Fp64Goldilocks omega_pow = Fp64Goldilocks::one();
    Fp64Goldilocks base = omega_inv;
    while (exp > 0) {
        if (exp & 1) {
            omega_pow = omega_pow * base;
        }
        base = base * base;
        exp >>= 1;
    }

    inv_x[gid] = (h_inv * omega_pow).canonicalize();
}
