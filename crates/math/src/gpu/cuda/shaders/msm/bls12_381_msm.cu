// BLS12-381 MSM CUDA kernel instantiation
//
// Compiled to .ptx by build.rs (nvcc -ptx)
// The build system walks crates/math/src/gpu/cuda/shaders/ recursively
// for .cu files, so this file is picked up automatically.
//
// WARNING: The bucket_accumulation kernel has a documented race condition
// when multiple threads write to the same bucket without synchronization.
// This first CUDA port carries the same limitation as the Metal MSM.
// A correctness fix (sorting-based approach) is planned as follow-up.

#include "msm.cuh"

extern "C" {

__global__ void bucket_accumulation_bls12_381(
    const int *scalars,
    const unsigned long long *points,
    unsigned long long *buckets,
    const unsigned int *config
) {
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    bucket_accumulation_384_impl(scalars, points, buckets, config, gid);
}

__global__ void bucket_reduction_bls12_381(
    unsigned long long *buckets,
    unsigned long long *window_sums,
    const unsigned int *config
) {
    unsigned int window_idx = blockIdx.x * blockDim.x + threadIdx.x;
    bucket_reduction_384_impl(buckets, window_sums, config, window_idx);
}

} // extern "C"
