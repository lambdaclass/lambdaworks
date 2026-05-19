// MSM kernel templates for CUDA (384-bit)
// Ported from lambdaworks Metal MSM kernels

#ifndef MSM384_CUH
#define MSM384_CUH

#include "jacobian384.cuh"

// Configuration struct passed from CPU (as array of unsigned ints)
// config[0] = num_scalars
// config[1] = num_windows
// config[2] = num_buckets
// config[3] = window_size

#define MSM_DESC_SIGN_BIT_384 0x80000000u
#define MSM_DESC_INDEX_MASK_384 0x7fffffffu
#define MSM_SEGMENT_CHUNK_SIZE_384 256u

__device__ void count_bucket_entries_384_impl(
    const int *scalars,
    unsigned int *counts,
    const unsigned int *config,
    unsigned int gid
) {
    unsigned int num_scalars = config[0];
    unsigned int num_windows = config[1];
    unsigned int num_buckets = config[2];
    unsigned int total_digits = num_scalars * num_windows;

    if (gid >= total_digits) return;

    int digit = scalars[gid];
    if (digit == 0) return;

    unsigned int scalar_idx = gid / num_windows;
    unsigned int window_idx = gid - scalar_idx * num_windows;
    unsigned int abs_digit = (digit < 0) ? (unsigned int)(-digit) : (unsigned int)digit;
    unsigned int bucket_idx = abs_digit - 1;

    if (bucket_idx >= num_buckets) return;

    unsigned int key = window_idx * num_buckets + bucket_idx;
    atomicAdd(&counts[key], 1u);
}

__device__ void scatter_bucket_descriptors_384_impl(
    const int *scalars,
    unsigned int *write_counts,
    const unsigned int *offsets,
    unsigned int *sorted_descriptors,
    const unsigned int *config,
    unsigned int gid
) {
    unsigned int num_scalars = config[0];
    unsigned int num_windows = config[1];
    unsigned int num_buckets = config[2];
    unsigned int total_digits = num_scalars * num_windows;

    if (gid >= total_digits) return;

    int digit = scalars[gid];
    if (digit == 0) return;

    unsigned int scalar_idx = gid / num_windows;
    unsigned int window_idx = gid - scalar_idx * num_windows;
    unsigned int abs_digit = (digit < 0) ? (unsigned int)(-digit) : (unsigned int)digit;
    unsigned int bucket_idx = abs_digit - 1;

    if (bucket_idx >= num_buckets) return;

    unsigned int key = window_idx * num_buckets + bucket_idx;
    unsigned int write_idx = atomicAdd(&write_counts[key], 1u);
    unsigned int descriptor = scalar_idx & MSM_DESC_INDEX_MASK_384;

    if (digit < 0) {
        descriptor |= MSM_DESC_SIGN_BIT_384;
    }

    sorted_descriptors[offsets[key] + write_idx] = descriptor;
}

__device__ JacobianPoint384 load_signed_descriptor_point_384(
    const unsigned int *sorted_descriptors,
    const unsigned long long *points,
    unsigned int descriptor_idx
) {
    unsigned int descriptor = sorted_descriptors[descriptor_idx];
    unsigned int scalar_idx = descriptor & MSM_DESC_INDEX_MASK_384;
    JacobianPoint384 p = load_point_384(points, scalar_idx);

    if ((descriptor & MSM_DESC_SIGN_BIT_384) != 0) {
        p = jacobian_neg_384(p, BLS12_381_P);
    }

    return p;
}

__device__ void segmented_bucket_accumulation_384_impl(
    const unsigned int *sorted_descriptors,
    const unsigned int *offsets,
    const unsigned long long *points,
    unsigned long long *buckets,
    const unsigned int *config,
    unsigned int gid
) {
    unsigned int num_windows = config[1];
    unsigned int num_buckets = config[2];
    unsigned int total_buckets = num_windows * num_buckets;

    if (gid >= total_buckets) return;

    unsigned int start = offsets[gid];
    unsigned int end = offsets[gid + 1];
    JacobianPoint384 bucket = jacobian_identity_384();

    for (unsigned int i = start; i < end; i++) {
        JacobianPoint384 p = load_signed_descriptor_point_384(sorted_descriptors, points, i);
        bucket = jacobian_add_384(bucket, p, BLS12_381_P, BLS12_381_INV);
    }

    store_point_384(buckets, gid, bucket);
}

__device__ void partial_segment_accumulation_384_impl(
    const unsigned int *sorted_descriptors,
    const unsigned int *task_starts,
    const unsigned int *task_lengths,
    const unsigned long long *points,
    unsigned long long *partial_buckets,
    const unsigned int *config,
    unsigned int gid
) {
    unsigned int total_tasks = config[0];

    if (gid >= total_tasks) return;

    unsigned int start = task_starts[gid];
    unsigned int len = task_lengths[gid];
    unsigned int end = start + len;
    JacobianPoint384 bucket = jacobian_identity_384();

    for (unsigned int i = start; i < end; i++) {
        JacobianPoint384 p = load_signed_descriptor_point_384(sorted_descriptors, points, i);
        bucket = jacobian_add_384(bucket, p, BLS12_381_P, BLS12_381_INV);
    }

    store_point_384(partial_buckets, gid, bucket);
}

__device__ void finalize_segment_accumulation_384_impl(
    const unsigned int *task_offsets,
    unsigned long long *partial_buckets,
    unsigned long long *buckets,
    const unsigned int *config,
    unsigned int bucket_idx
) {
    unsigned int total_buckets = config[1];

    if (bucket_idx >= total_buckets) return;

    unsigned int start = task_offsets[bucket_idx];
    unsigned int end = task_offsets[bucket_idx + 1];
    JacobianPoint384 bucket = jacobian_identity_384();

    for (unsigned int i = start; i < end; i++) {
        JacobianPoint384 partial = load_point_384(partial_buckets, i);
        bucket = jacobian_add_384(bucket, partial, BLS12_381_P, BLS12_381_INV);
    }

    store_point_384(buckets, bucket_idx, bucket);
}

__device__ JacobianPoint384 jacobian_mul_u32_384(JacobianPoint384 point, unsigned int scalar) {
    JacobianPoint384 result = jacobian_identity_384();

    while (scalar != 0) {
        if ((scalar & 1u) != 0) {
            result = jacobian_add_384(result, point, BLS12_381_P, BLS12_381_INV);
        }
        scalar >>= 1;
        if (scalar != 0) {
            point = jacobian_double_384(point, BLS12_381_P, BLS12_381_INV);
        }
    }

    return result;
}

__device__ void partial_bucket_reduction_384_impl(
    unsigned long long *buckets,
    unsigned long long *partial_sums,
    unsigned long long *partial_results,
    const unsigned int *config,
    unsigned int gid
) {
    unsigned int num_windows = config[0];
    unsigned int num_buckets = config[1];
    unsigned int chunk_size = config[2];
    unsigned int chunks_per_window = config[3];
    unsigned int total_chunks = num_windows * chunks_per_window;

    if (gid >= total_chunks) return;

    unsigned int window_idx = gid / chunks_per_window;
    unsigned int chunk_idx = gid - window_idx * chunks_per_window;
    unsigned int chunk_start = chunk_idx * chunk_size;
    unsigned int chunk_end = chunk_start + chunk_size;

    if (chunk_start >= num_buckets) {
        store_point_384(partial_sums, gid, jacobian_identity_384());
        store_point_384(partial_results, gid, jacobian_identity_384());
        return;
    }
    if (chunk_end > num_buckets) {
        chunk_end = num_buckets;
    }

    unsigned int bucket_base = window_idx * num_buckets;
    JacobianPoint384 running_sum = jacobian_identity_384();
    JacobianPoint384 result = jacobian_identity_384();

    for (int i = (int)chunk_end - 1; i >= (int)chunk_start; i--) {
        JacobianPoint384 bucket = load_point_384(buckets, bucket_base + (unsigned int)i);
        running_sum = jacobian_add_384(running_sum, bucket, BLS12_381_P, BLS12_381_INV);
        result = jacobian_add_384(result, running_sum, BLS12_381_P, BLS12_381_INV);
    }

    store_point_384(partial_sums, gid, running_sum);
    store_point_384(partial_results, gid, result);
}

__device__ void finalize_bucket_reduction_384_impl(
    unsigned long long *partial_sums,
    unsigned long long *partial_results,
    unsigned long long *window_sums,
    const unsigned int *config,
    unsigned int window_idx
) {
    unsigned int num_windows = config[0];
    unsigned int num_buckets = config[1];
    unsigned int chunk_size = config[2];
    unsigned int chunks_per_window = config[3];

    if (window_idx >= num_windows) return;

    JacobianPoint384 higher_chunks_sum = jacobian_identity_384();
    JacobianPoint384 result = jacobian_identity_384();
    unsigned int partial_base = window_idx * chunks_per_window;

    for (int chunk_idx = (int)chunks_per_window - 1; chunk_idx >= 0; chunk_idx--) {
        unsigned int chunk_start = (unsigned int)chunk_idx * chunk_size;
        if (chunk_start >= num_buckets) continue;

        unsigned int chunk_end = chunk_start + chunk_size;
        if (chunk_end > num_buckets) {
            chunk_end = num_buckets;
        }
        unsigned int chunk_len = chunk_end - chunk_start;
        unsigned int partial_idx = partial_base + (unsigned int)chunk_idx;

        JacobianPoint384 carry = jacobian_mul_u32_384(higher_chunks_sum, chunk_len);
        JacobianPoint384 partial_result = load_point_384(partial_results, partial_idx);
        JacobianPoint384 partial_sum = load_point_384(partial_sums, partial_idx);

        result = jacobian_add_384(result, carry, BLS12_381_P, BLS12_381_INV);
        result = jacobian_add_384(result, partial_result, BLS12_381_P, BLS12_381_INV);
        higher_chunks_sum = jacobian_add_384(higher_chunks_sum, partial_sum, BLS12_381_P, BLS12_381_INV);
    }

    store_point_384(window_sums, window_idx, result);
}

// Bucket reduction kernel
// Reduces buckets within a window to a single point.
// No race condition: each thread owns its window exclusively.
__device__ void bucket_reduction_384_impl(
    unsigned long long *buckets,
    unsigned long long *window_sums,
    const unsigned int *config,
    unsigned int window_idx
) {
    unsigned int num_windows = config[0];
    unsigned int num_buckets = config[1];

    if (window_idx >= num_windows) return;

    unsigned int bucket_base = window_idx * num_buckets;
    JacobianPoint384 running_sum = jacobian_identity_384();
    JacobianPoint384 result = jacobian_identity_384();

    for (int i = (int)num_buckets - 1; i >= 0; i--) {
        JacobianPoint384 bucket = load_point_384(buckets, bucket_base + (unsigned int)i);

        running_sum = jacobian_add_384(running_sum, bucket, BLS12_381_P, BLS12_381_INV);
        result = jacobian_add_384(result, running_sum, BLS12_381_P, BLS12_381_INV);
    }

    store_point_384(window_sums, window_idx, result);
}

#endif /* MSM384_CUH */
