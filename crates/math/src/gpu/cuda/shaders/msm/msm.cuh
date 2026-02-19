// MSM kernel templates for CUDA (384-bit)
// Ported from lambdaworks Metal MSM kernels
//
// WARNING: The bucket_accumulation kernel has a RACE CONDITION when multiple
// threads write to the same bucket. See bls12_381_msm.cu for details.

#ifndef MSM384_CUH
#define MSM384_CUH

#include "jacobian384.cuh"

// Configuration struct passed from CPU (as array of unsigned ints)
// config[0] = num_scalars
// config[1] = num_windows
// config[2] = num_buckets
// config[3] = window_size

// Bucket accumulation kernel
// WARNING: This kernel has a RACE CONDITION when multiple threads write to the same bucket.
// For production use, implement one of:
// 1. Sorting-based approach (cuZK paper) - sort (bucket_idx, point) pairs, then scan
// 2. Atomic operations for point addition (complex for 384-bit)
// 3. Per-thread local buckets with tree reduction
//
// Current behavior: The last thread to write to a bucket wins, producing incorrect results
// when multiple points map to the same bucket.
__device__ void bucket_accumulation_384_impl(
    const int *scalars,
    const unsigned long long *points,
    unsigned long long *buckets,
    const unsigned int *config,
    unsigned int gid
) {
    unsigned int num_scalars = config[0];
    unsigned int num_windows = config[1];
    unsigned int num_buckets = config[2];

    // Each thread handles one (scalar, window) pair
    unsigned int scalar_idx = gid / num_windows;
    unsigned int window_idx = gid % num_windows;

    if (scalar_idx >= num_scalars) return;

    // Get the signed digit for this scalar and window
    int digit = scalars[scalar_idx * num_windows + window_idx];

    if (digit == 0) return;

    // Load the point
    JacobianPoint384 p = load_point_384(points, scalar_idx);

    // Determine bucket index and whether to negate
    unsigned int bucket_idx;
    bool negate;
    if (digit > 0) {
        bucket_idx = (unsigned int)(digit - 1);
        negate = false;
    } else {
        bucket_idx = (unsigned int)(-digit - 1);
        negate = true;
    }

    // Negate point if needed
    if (negate) {
        p = jacobian_neg_384(p, BLS12_381_P);
    }

    // Calculate global bucket index
    unsigned int global_bucket_idx = window_idx * num_buckets + bucket_idx;

    // Load current bucket value, add point, store back
    // WARNING: This read-modify-write is NOT atomic and causes race conditions
    JacobianPoint384 bucket = load_point_384(buckets, global_bucket_idx);
    bucket = jacobian_add_384(bucket, p, BLS12_381_P, BLS12_381_INV);
    store_point_384(buckets, global_bucket_idx, bucket);
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

    // Running sum for bucket reduction
    JacobianPoint384 running_sum = jacobian_identity_384();
    // Accumulated result
    JacobianPoint384 result = jacobian_identity_384();

    // Process buckets in reverse order (highest weight first)
    for (int i = (int)num_buckets - 1; i >= 0; i--) {
        JacobianPoint384 bucket = load_point_384(buckets, bucket_base + (unsigned int)i);

        running_sum = jacobian_add_384(running_sum, bucket, BLS12_381_P, BLS12_381_INV);
        result = jacobian_add_384(result, running_sum, BLS12_381_P, BLS12_381_INV);
    }

    // Store the window sum
    store_point_384(window_sums, window_idx, result);
}

#endif /* MSM384_CUH */
