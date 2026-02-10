// BabyBear 32-bit prime field instantiation for Metal shaders.
//
// Prime: P = 2013265921 = 15 * 2^27 + 1
// Montgomery constants:
//   MU  = 2281701377 (= -P^(-1) mod 2^32)
//   R2  = 1172168163 (= 2^64 mod P)
//   ONE = 268435454  (= 2^32 mod P, i.e., R mod P)
//
// Constants verified against lambdaworks Rust implementation in
// crates/math/src/field/fields/u32_montgomery_backend_prime_field.rs

#pragma once

#include "fp_u32.h.metal"

typedef Fp32<2013265921u, 2281701377u, 1172168163u, 268435454u> FpBabyBear;
