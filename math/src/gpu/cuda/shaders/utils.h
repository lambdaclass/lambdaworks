#ifndef UTILS_H
#define UTILS_H

/// Reverses the `log2(size)` first bits of `i`
__device__ unsigned reverse_index(unsigned i, unsigned size)
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

#endif // UTILS_H
