// lambdaworks Metal shader library entry point.
//
// This file includes all Metal shaders and serves as the compilation entry point
// for generating the lib.metallib binary.
//
// Inspired by:
// - Original lambdaworks Metal implementation (pre-PR#993)
// - ICICLE multi-backend architecture
// - ministark Metal GPU implementation
// - Plonky3 Goldilocks implementation
//
// Currently supported fields:
// - Stark252 (256-bit, used by StarkWare/Cairo)
// - Goldilocks (64-bit, p = 2^64 - 2^32 + 1, used by Plonky2/Plonky3)
//   - Fp2 quadratic extension (x^2 - 7)
//   - Fp3 cubic extension (x^3 - 2)
//
// To add a new field:
// 1. Create a new header in shaders/field/ that instantiates the field class
// 2. Add explicit template instantiations for all kernel functions
// 3. Include the header here

// Base field instantiations
#include "shaders/field/stark256.h.metal"
#include "shaders/field/goldilocks.h.metal"

// BabyBear 32-bit prime field
#include "shaders/field/babybear.h.metal"

// Goldilocks extension field instantiations (for FFT with base field twiddles)
#include "shaders/field/goldilocks_fp2.h.metal"
#include "shaders/field/goldilocks_fp3.h.metal"

// Mersenne31 Circle FFT kernels
#include "shaders/field/mersenne31_cfft.h.metal"

// Sumcheck protocol kernels
#include "shaders/sumcheck/sumcheck_babybear.h.metal"
#include "shaders/sumcheck/sumcheck_goldilocks.h.metal"
