#pragma once

#include "fp_u256.h.metal"
#include "fp_bls12381.h.metal"
#include "ec_point.h.metal"
#include "../test/test_bls12381.h.metal"
#include "../msm/pippenger.h.metal"

namespace {
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
