#pragma once

#include "fp_u256.h.metal"

#include "../fft/fft.h.metal"
#include "../fft/twiddles.h.metal"
#include "../fft/permutation.h.metal"

// Prime Field of U256 with modulus 0x800000000000011000000000000000000000000000000000000000000000001, used for Starks
namespace {
    typedef Fp256<
    /* =N **/ /*u256(*/ 576460752303423505, 0, 0, 1 /*)*/,
    /* =R_SQUARED **/ /*u256(*/ 576413109808302096, 18446744073700081664, 5151653887, 18446741271209837569 /*)*/,
    /* =N_PRIME **/ /*u256(*/ 576460752303423504, 18446744073709551615, 18446744073709551615, 18446744073709551615 /*)*/
    > Fp;
}

template [[ host_name("radix2_dit_butterfly_stark256") ]] 
[[kernel]] void radix2_dit_butterfly<Fp>(
    device Fp*,
    constant Fp*,
    constant uint32_t&,
    uint32_t,
    uint32_t
);

template [[ host_name("calc_twiddles_stark256") ]] 
[[kernel]] void calc_twiddles<Fp>(
    device Fp*,
    constant Fp&, 
    uint
);

template [[ host_name("calc_twiddles_inv_stark256") ]] 
[[kernel]] void calc_twiddles_inv<Fp>(
    device Fp*,
    constant Fp&, 
    uint
);

template [[ host_name("calc_twiddles_bitrev_stark256") ]] 
[[kernel]] void calc_twiddles_bitrev<Fp>(
    device Fp*,
    constant Fp&, 
    uint,
    uint
);

template [[ host_name("calc_twiddles_bitrev_inv_stark256") ]] 
[[kernel]] void calc_twiddles_bitrev_inv<Fp>(
    device Fp*,
    constant Fp&, 
    uint,
    uint
);

template [[ host_name("bitrev_permutation_stark256") ]] 
[[kernel]] void bitrev_permutation<Fp>(
    device Fp*, 
    device Fp*, 
    uint, 
    uint
);
