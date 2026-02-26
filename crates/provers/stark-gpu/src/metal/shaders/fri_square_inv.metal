// GPU FRI inverse stride-2 squaring for subsequent FRI layers.
//
// Given inv_x[i] = 1/x[2*i] from the previous layer (where x[2*i] is the
// domain point for the first element of each fold pair), computes the domain
// inverses for the next layer's fold:
//
//   inv_x_next[j] = inv_x_prev[2*j]^2
//
// Why stride-2: The fold kernel pairs evals[2*i] and evals[2*i+1], using
// inv_x[i] = 1/x[2*i]. After folding, result[i] sits at domain point x[2*i]^2.
// The NEXT fold pairs result[2*j] and result[2*j+1], needing 1/result_domain[2*j]
// = 1/(x[4*j]^2) = (1/x[4*j])^2 = inv_x_prev[2*j]^2.
//
// Output has half the length of input.
//
// NOTE: fp_u64.h.metal is concatenated at runtime. Do NOT #include it.

struct FriSquareInvParams {
    uint32_t len;   // output length (= half of input length)
};

kernel void fri_square_inverses(
    device const Fp64Goldilocks* inv_x_prev  [[ buffer(0) ]],
    device Fp64Goldilocks* inv_x_next        [[ buffer(1) ]],
    constant FriSquareInvParams& params      [[ buffer(2) ]],
    uint gid [[ thread_position_in_grid ]]
) {
    if (gid >= params.len) return;

    Fp64Goldilocks v = inv_x_prev[2 * gid];
    inv_x_next[gid] = (v * v).canonicalize();
}
