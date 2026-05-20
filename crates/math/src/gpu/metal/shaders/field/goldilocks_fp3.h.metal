// Goldilocks Fp3 extension field kernel instantiations for Metal shaders.
//
// Instantiates FFT kernels for Goldilocks cubic extension field (Fp3).
// This enables FFT operations with:
// - Coefficients in Fp3 (extension field)
// - Twiddles in Fp (base Goldilocks field)

#pragma once

#include "fp_u64.h.metal"
#include "fp3_goldilocks.h.metal"

#include "../fft/fft_extension.h.metal"
#include "../fft/permutation.h.metal"

// Type aliases for clarity
typedef Fp64Goldilocks FpBaseGoldilocks;
typedef Fp3Goldilocks FpExtFp3;

// ============================================================
// FFT Kernel Instantiations for Goldilocks Fp3
// ============================================================

// Butterfly kernel for extension field FFT with base field twiddles
template [[ host_name("radix2_dit_butterfly_Goldilocks_fp3") ]]
[[kernel]] void radix2_dit_butterfly_ext<FpExtFp3, FpBaseGoldilocks>(
    device FpExtFp3*,
    constant FpBaseGoldilocks*,
    constant uint32_t&,
    uint32_t,
    uint32_t
);

// Threadgroup-cached butterfly for extension field FFT
template [[ host_name("radix2_dit_butterfly_tg_Goldilocks_fp3") ]]
[[kernel]] void radix2_dit_butterfly_tg_ext<FpExtFp3, FpBaseGoldilocks>(
    device FpExtFp3*,
    constant FpBaseGoldilocks*,
    constant uint32_t&,
    constant uint32_t&,
    uint32_t,
    uint32_t,
    uint32_t,
    uint32_t,
    threadgroup FpBaseGoldilocks*
);

// Fused multi-stage butterfly for extension field FFT
template [[ host_name("radix2_dit_butterfly_fused_Goldilocks_fp3") ]]
[[kernel]] void radix2_dit_butterfly_fused_ext<FpExtFp3, FpBaseGoldilocks>(
    device FpExtFp3*,
    constant FpBaseGoldilocks*,
    constant uint32_t&,
    constant uint32_t&,
    uint32_t,
    uint32_t,
    uint32_t,
    threadgroup FpExtFp3*
);

// Bit-reverse permutation for extension field elements
template [[ host_name("bitrev_permutation_Goldilocks_fp3") ]]
[[kernel]] void bitrev_permutation_ext<FpExtFp3>(
    device FpExtFp3*,
    device FpExtFp3*,
    uint32_t,
    uint32_t
);

// ---- Bowers-G NTT + fused LDE kernel instantiations (Fp3) ----
// Data is Fp3; twiddles and coset powers are base-field Goldilocks.
#include "../fft/bowers_lde.h.metal"

[[ host_name("bowers_lde_middle_Goldilocks_fp3") ]]
[[kernel]] void bowers_lde_middle_Goldilocks_fp3(
    device   FpExtFp3*        data            [[ buffer(0) ]],
    constant FpBaseGoldilocks* twiddles_bitrev [[ buffer(1) ]],
    constant uint32_t&        stage           [[ buffer(2) ]],
    uint32_t                  thread_pos      [[ thread_position_in_grid ]]
) {
    bowers_lde_middle_body<FpExtFp3, FpBaseGoldilocks>(
        data, twiddles_bitrev, stage, thread_pos
    );
}

[[ host_name("bowers_lde_head_Goldilocks_fp3") ]]
[[kernel]] void bowers_lde_head_Goldilocks_fp3(
    device   FpExtFp3*        data            [[ buffer(0) ]],
    constant FpBaseGoldilocks* twiddles_bitrev [[ buffer(1) ]],
    constant FpBaseGoldilocks* coset_powers    [[ buffer(2) ]],
    constant uint32_t&        num_stages      [[ buffer(3) ]],
    uint32_t                  tg_id           [[ threadgroup_position_in_grid ]],
    uint32_t                  tg_pos          [[ thread_position_in_threadgroup ]],
    uint32_t                  tg_size         [[ threads_per_threadgroup ]],
    threadgroup FpExtFp3*     shared_data     [[ threadgroup(0) ]]
) {
    bowers_lde_head_body<FpExtFp3, FpBaseGoldilocks>(
        data, twiddles_bitrev, coset_powers, num_stages, tg_id, tg_pos, tg_size, shared_data
    );
}
