#include "./fp_u256.cuh"
#include "./utils.h"

extern "C"
{
    __global__ void bitrev_permutation(
        const p256::Fp *input,
        p256::Fp *result
    ) {
        unsigned index = threadIdx.x;
        unsigned size = blockDim.x;

        result[index] = input[reverse_index(index, size)];
    };
}
