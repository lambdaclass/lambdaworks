#include "./utils.cuh"

/// Reverses the `log2(size)` first bits of `i`
__device__ uint reverse_index(uint i, uint size)
{
    if (size == 1)
    { // TODO: replace this statement with an alternative solution.
        return i;
    }
    else
    {
        return __brev(i) >> (__clz(size) + 1);
    }
}
