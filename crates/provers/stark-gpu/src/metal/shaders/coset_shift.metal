// Coset shift and scalar multiplication kernels for Goldilocks field.
//
// NOTE: This shader is compiled at runtime by concatenating the Goldilocks
// field header (fp_u64.h.metal) before this source. The Fp64Goldilocks class
// is expected to be already defined when this code is compiled.
//
// Values from device/constant address space buffers must be copied into
// thread-local variables before calling Fp64Goldilocks arithmetic methods,
// since those methods operate in the default (thread) address space.

/// Coset shift: output[k] = input[k] * offset^k for k < input_len, zero for k >= input_len.
///
/// This multiplies each polynomial coefficient by the appropriate power of the
/// coset offset, preparing it for FFT evaluation on an offset domain.
/// Elements beyond input_len are zero-padded (for blowup factor padding).
kernel void goldilocks_coset_shift(
    device const Fp64Goldilocks* input   [[ buffer(0) ]],
    device Fp64Goldilocks* output        [[ buffer(1) ]],
    constant Fp64Goldilocks& offset      [[ buffer(2) ]],
    constant uint& input_len             [[ buffer(3) ]],
    constant uint& output_len            [[ buffer(4) ]],
    uint gid [[ thread_position_in_grid ]]
) {
    if (gid >= output_len) return;
    if (gid < input_len) {
        // Copy from device/constant to thread-local for arithmetic
        Fp64Goldilocks coeff = input[gid];
        Fp64Goldilocks off = offset;
        Fp64Goldilocks power = off.pow(gid);
        output[gid] = coeff * power;
    } else {
        output[gid] = Fp64Goldilocks::zero();
    }
}

/// Scale all elements: output[k] = input[k] * scalar.
///
/// Simple element-wise scalar multiplication on a GPU buffer.
kernel void goldilocks_scale(
    device const Fp64Goldilocks* input  [[ buffer(0) ]],
    device Fp64Goldilocks* output       [[ buffer(1) ]],
    constant Fp64Goldilocks& scalar     [[ buffer(2) ]],
    constant uint& len                  [[ buffer(3) ]],
    uint gid [[ thread_position_in_grid ]]
) {
    if (gid >= len) return;
    // Copy from device/constant to thread-local for arithmetic
    Fp64Goldilocks val = input[gid];
    Fp64Goldilocks sc = scalar;
    output[gid] = val * sc;
}

/// Cyclic multiply: output[i] = input[i] * pattern[i % pattern_len].
///
/// Used for combining base zerofier (small cyclic pattern) with end-exemptions
/// evaluations (large buffer), keeping the entire zerofier computation on GPU.
kernel void goldilocks_cyclic_mul(
    device const Fp64Goldilocks* input   [[ buffer(0) ]],
    device Fp64Goldilocks* output        [[ buffer(1) ]],
    device const Fp64Goldilocks* pattern [[ buffer(2) ]],
    constant uint& len                   [[ buffer(3) ]],
    constant uint& pattern_len           [[ buffer(4) ]],
    uint gid [[ thread_position_in_grid ]]
) {
    if (gid >= len) return;
    Fp64Goldilocks val = input[gid];
    Fp64Goldilocks pat = pattern[gid % pattern_len];
    output[gid] = val * pat;
}
