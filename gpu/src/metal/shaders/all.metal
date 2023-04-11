// This is necessary because pragma once doesn't work as expected
// and some symbols are being duplicated.

// TODO: Investigate this issue, having .metal sources would be better
// than headers and a unique source.

#include "fft.h.metal"
#include "twiddles.h.metal"
#include "permutation.h.metal"
