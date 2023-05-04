#pragma once

/// Reverses the `log2(size)` first bits of `i`
__device__ uint reverse_index(uint i, uint size);
