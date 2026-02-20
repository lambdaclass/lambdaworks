// Evaluation-domain FRI fold kernel for Goldilocks field (bit-reversed order).
//
// NOTE: fp_u64.h.metal is concatenated at runtime. Do NOT #include it.
//
// Given evaluations in bit-reversed order on a coset domain, the paired
// elements (x, -x) are adjacent at positions (2i, 2i+1). This is because
// bit-reversing maps natural indices (j, j+N/2) to positions that differ
// only in the LSB, making them consecutive.
//
// Formula (matching verifier convention):
//   result[i] = (evals[2*i] + evals[2*i + 1])
//             + beta * (evals[2*i] - evals[2*i + 1]) * inv_x[i]
//
// where inv_x[i] = 1/x for the domain point at bit-reversed position 2*i.
// Output is in bit-reversed order for the next FRI layer's domain.

struct FriFoldEvalParams {
    uint32_t half_len;
};

kernel void goldilocks_fri_fold_eval(
    device const Fp64Goldilocks* evals   [[ buffer(0) ]],
    device Fp64Goldilocks* result        [[ buffer(1) ]],
    constant Fp64Goldilocks& beta        [[ buffer(2) ]],
    device const Fp64Goldilocks* inv_x   [[ buffer(3) ]],
    constant FriFoldEvalParams& params   [[ buffer(4) ]],
    uint gid [[ thread_position_in_grid ]]
) {
    if (gid >= params.half_len) return;

    Fp64Goldilocks f_pos = evals[2 * gid];
    Fp64Goldilocks f_neg = evals[2 * gid + 1];
    Fp64Goldilocks b = beta;
    Fp64Goldilocks ix = inv_x[gid];

    Fp64Goldilocks sum = f_pos + f_neg;
    Fp64Goldilocks diff = f_pos - f_neg;

    result[gid] = (sum + b * diff * ix).canonicalize();
}
