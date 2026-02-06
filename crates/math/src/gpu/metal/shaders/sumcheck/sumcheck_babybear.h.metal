// BabyBear kernel instantiations for sumcheck protocol.

#pragma once

#include "../field/babybear.h.metal"
#include "sumcheck_contribute.h.metal"
#include "sumcheck_reduce.h.metal"
#include "sumcheck_fold.h.metal"

// Contribute kernel
template [[ host_name("sumcheck_contribute_BabyBear") ]]
[[kernel]] void sumcheck_contribute<FpBabyBear>(
    device const FpBabyBear*,
    device FpBabyBear*,
    constant uint32_t&,
    constant uint32_t&,
    constant uint32_t&,
    constant FpBabyBear&,
    uint, uint, uint, uint
);

// Reduce kernel
template [[ host_name("sumcheck_reduce_BabyBear") ]]
[[kernel]] void sumcheck_reduce<FpBabyBear>(
    device FpBabyBear*,
    constant uint32_t&,
    uint, uint
);

// Fold kernel
template [[ host_name("sumcheck_fold_BabyBear") ]]
[[kernel]] void sumcheck_fold<FpBabyBear>(
    device FpBabyBear*,
    constant FpBabyBear&,
    constant uint32_t&,
    uint
);
