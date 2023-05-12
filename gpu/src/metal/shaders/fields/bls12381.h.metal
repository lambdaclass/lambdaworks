#pragma once

#include "fp_u256.h.metal"
#include "ec_point.h.metal"
#include "../test/test_bls12381.h.metal"
#include "../msm/pippenger.h.metal"

// Prime Field of U256 with modulus 0x800000000000011000000000000000000000000000000000000000000000001, used for Starks
namespace {
    typedef Fp256<
    /* =N **/ /*u256(*/ 576460752303423505, 0, 0, 1 /*)*/,
    /* =R_SQUARED **/ /*u256(*/ 576413109808302096, 18446744073700081664, 5151653887, 18446741271209837569 /*)*/,
    /* =N_PRIME **/ /*u256(*/ 576460752303423504, 18446744073709551615, 18446744073709551615, 18446744073709551615 /*)*/
    > FpBLS12381;

    typedef ECPoint<FpBLS12381, 0> BLS12381;
}

template [[ host_name("bls12381_add") ]]
[[kernel]] void add<BLS12381, FpBLS12381>(
    constant FpBLS12381*,
    constant FpBLS12381*,
    device FpBLS12381*
);

template [[ host_name("calculate_Gjs_bls12381") ]]
[[kernel]] void calculate_Gjs<FpBLS12381, BLS12381>(
    constant FpBLS12381*,
    constant BLS12381*,
    constant uint32_t&,
    constant uint64_t&,
    threadgroup BLS12381*,
    device BLS12381*,
    uint32_t,
    uint32_t,
    uint32_t
);
