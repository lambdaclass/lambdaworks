// This is necessary because pragma once doesn't work as expected
// and some symbols are being duplicated.

// TODO: Investigate this issue, having .metal sources would be better
// than headers and a unique source.

#include "fields/stark256.h.metal"
#include "fields/bls12381.h.metal"
#include "test/test_bls12381.h.metal"
#include "test/test_unsigned_integer.h.metal"
#include "msm/pippenger.h.metal"
