// Goldilocks kernel instantiations for sumcheck protocol.

#pragma once

#include "../field/goldilocks.h.metal"
#include "sumcheck_contribute.h.metal"
#include "sumcheck_reduce.h.metal"
#include "sumcheck_fold.h.metal"

// Contribute kernel
template [[ host_name("sumcheck_contribute_Goldilocks") ]]
[[kernel]] void sumcheck_contribute<FpGoldilocks>(
    device const FpGoldilocks*,
    device FpGoldilocks*,
    constant uint32_t&,
    constant uint32_t&,
    constant uint32_t&,
    constant FpGoldilocks&,
    uint, uint, uint, uint
);

// Reduce kernel
template [[ host_name("sumcheck_reduce_Goldilocks") ]]
[[kernel]] void sumcheck_reduce<FpGoldilocks>(
    device FpGoldilocks*,
    constant uint32_t&,
    uint, uint
);

// Fold kernel
template [[ host_name("sumcheck_fold_Goldilocks") ]]
[[kernel]] void sumcheck_fold<FpGoldilocks>(
    device FpGoldilocks*,
    constant FpGoldilocks&,
    constant uint32_t&,
    uint
);
