// This is necessary because pragma once isn't working as expected and
// some symbols are being duplicated

// TODO: Investigate if this issue can be solved, having different .metal sources
// would be better than header files and a unique source file.

#include "fft.h.metal"
#include "twiddles.h.metal"
#include "permutation.h.metal"
