// lambdaworks Metal shader library entry point.
//
// This file includes all Metal shaders and serves as the compilation entry point
// for generating the lib.metallib binary.
//
// Inspired by:
// - Original lambdaworks Metal implementation (pre-PR#993)
// - ICICLE multi-backend architecture
// - ministark Metal GPU implementation
//
// Currently supported fields:
// - Stark252 (256-bit, used by StarkWare/Cairo)
//
// To add a new field:
// 1. Create a new header in shaders/field/ that instantiates the Fp256 template
// 2. Add explicit template instantiations for all kernel functions
// 3. Include the header here

// Field instantiations (includes all kernel instantiations)
#include "shaders/field/stark256.h.metal"
