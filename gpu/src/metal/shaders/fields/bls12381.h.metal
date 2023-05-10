#pragma once

#include "fp_u256.h.metal"
#include "ec_point.h.metal"
#include "../test/test_bls12381.h.metal"

// Prime Field of U256 with modulus 0x800000000000011000000000000000000000000000000000000000000000001, used for Starks
namespace {
    typedef Fp256<
    /* =N **/ /*u256(*/ 576460752303423505, 0, 0, 1 /*)*/,
    /* =R_SQUARED **/ /*u256(*/ 576413109808302096, 18446744073700081664, 5151653887, 18446741271209837569 /*)*/,
    /* =N_PRIME **/ /*u256(*/ 576460752303423504, 18446744073709551615, 18446744073709551615, 18446744073709551615 /*)*/
    > Fp;

    typedef ECPoint<Fp, 5> BLS12381;
}

template [[ host_name("bls12381_add") ]] 
[[kernel]] void add<BLS12381, Fp>(
    constant Fp*,
    constant Fp*,
    device Fp*
);
