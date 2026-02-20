// GPU FRI inverse squaring for subsequent FRI layers.
//
// Given inv_x[i] = 1/x_i from the previous layer, computes
// inv_x_next[i] = inv_x[i]^2 = 1/x_i^2.
//
// This works because after folding, the next layer's domain points
// are x_i^2, so their inverses are (1/x_i)^2 = inv_x[i]^2.
//
// NOTE: fp_u64.h.metal is concatenated at runtime. Do NOT #include it.

struct FriSquareInvParams {
    uint32_t len;
};

kernel void fri_square_inverses(
    device const Fp64Goldilocks* inv_x_prev  [[ buffer(0) ]],
    device Fp64Goldilocks* inv_x_next        [[ buffer(1) ]],
    constant FriSquareInvParams& params      [[ buffer(2) ]],
    uint gid [[ thread_position_in_grid ]]
) {
    if (gid >= params.len) return;

    Fp64Goldilocks v = inv_x_prev[gid];
    inv_x_next[gid] = (v * v).canonicalize();
}
