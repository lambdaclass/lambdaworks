// FRI fold kernel for Goldilocks field.
//
// NOTE: This shader is compiled at runtime by concatenating the Goldilocks
// field header (fp_u64.h.metal) before this source. The Fp64Goldilocks class
// is expected to be already defined when this code is compiled.
//
// Values from device/constant address space buffers must be copied into
// thread-local variables before calling Fp64Goldilocks arithmetic methods,
// since those methods operate in the default (thread) address space.
//
// FRI fold operation: result[k] = 2 * (coeffs[2k] + beta * coeffs[2k+1])
kernel void goldilocks_fri_fold(
    device const Fp64Goldilocks* coeffs  [[ buffer(0) ]],
    device Fp64Goldilocks* result        [[ buffer(1) ]],
    constant Fp64Goldilocks& beta        [[ buffer(2) ]],
    constant uint& half_len              [[ buffer(3) ]],
    uint gid [[ thread_position_in_grid ]]
) {
    if (gid >= half_len) return;
    // Copy from device/constant to thread-local for arithmetic
    Fp64Goldilocks even = coeffs[2 * gid];
    Fp64Goldilocks odd  = coeffs[2 * gid + 1];
    Fp64Goldilocks b    = beta;
    Fp64Goldilocks two  = Fp64Goldilocks(2);
    result[gid] = two * (even + b * odd);
}
