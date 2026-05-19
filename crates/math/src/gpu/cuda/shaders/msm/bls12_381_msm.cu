// BLS12-381 MSM CUDA kernel instantiation
//
// Compiled to .ptx by build.rs (nvcc -ptx)
// The build system walks crates/math/src/gpu/cuda/shaders/ recursively
// for .cu files, so this file is picked up automatically.

#include "msm.cuh"

extern "C" {

__global__ void count_bucket_entries_bls12_381(
    const int *scalars,
    unsigned int *counts,
    const unsigned int *config
) {
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    count_bucket_entries_384_impl(scalars, counts, config, gid);
}

__global__ void scatter_bucket_descriptors_bls12_381(
    const int *scalars,
    unsigned int *write_counts,
    const unsigned int *offsets,
    unsigned int *sorted_descriptors,
    const unsigned int *config
) {
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    scatter_bucket_descriptors_384_impl(scalars, write_counts, offsets, sorted_descriptors, config, gid);
}

__global__ void segmented_bucket_accumulation_bls12_381(
    const unsigned int *sorted_descriptors,
    const unsigned int *offsets,
    const unsigned long long *points,
    unsigned long long *buckets,
    const unsigned int *config
) {
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    segmented_bucket_accumulation_384_impl(sorted_descriptors, offsets, points, buckets, config, gid);
}

__global__ void partial_segment_accumulation_bls12_381(
    const unsigned int *sorted_descriptors,
    const unsigned int *task_starts,
    const unsigned int *task_lengths,
    const unsigned long long *points,
    unsigned long long *partial_buckets,
    const unsigned int *config
) {
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    partial_segment_accumulation_384_impl(
        sorted_descriptors,
        task_starts,
        task_lengths,
        points,
        partial_buckets,
        config,
        gid
    );
}

__global__ void finalize_segment_accumulation_bls12_381(
    const unsigned int *task_offsets,
    unsigned long long *partial_buckets,
    unsigned long long *buckets,
    const unsigned int *config
) {
    unsigned int bucket_idx = blockIdx.x * blockDim.x + threadIdx.x;
    finalize_segment_accumulation_384_impl(task_offsets, partial_buckets, buckets, config, bucket_idx);
}

__global__ void partial_bucket_reduction_bls12_381(
    unsigned long long *buckets,
    unsigned long long *partial_sums,
    unsigned long long *partial_results,
    const unsigned int *config
) {
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    partial_bucket_reduction_384_impl(buckets, partial_sums, partial_results, config, gid);
}

__global__ void finalize_bucket_reduction_bls12_381(
    unsigned long long *partial_sums,
    unsigned long long *partial_results,
    unsigned long long *window_sums,
    const unsigned int *config
) {
    unsigned int window_idx = blockIdx.x * blockDim.x + threadIdx.x;
    finalize_bucket_reduction_384_impl(partial_sums, partial_results, window_sums, config, window_idx);
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
