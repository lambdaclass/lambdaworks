//! Metal Shader Source for Sumcheck Operations
//!
//! Contains Metal Shading Language (MSL) kernels for:
//! - Parallel reduction (summing over hypercube)
//! - Challenge application (fixing variables)
//! - Round polynomial computation

/// Metal shader source code for sumcheck operations.
///
/// This shader is designed for 64-bit prime fields (e.g., Goldilocks, BabyBear).
/// For fields with different moduli, the MODULUS constant should be adjusted.
pub const SUMCHECK_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Field modulus - configure for your field
// Default: Goldilocks prime (2^64 - 2^32 + 1)
constant uint64_t MODULUS = 0xFFFFFFFF00000001ULL;

// Modular addition
inline uint64_t mod_add(uint64_t a, uint64_t b) {
    uint64_t sum = a + b;
    bool overflow = sum < a;
    // If overflow occurred, we need to add 2^64 - MODULUS = NEG_ORDER
    if (overflow) {
        sum += NEG_ORDER;  // NEG_ORDER = 2^32 - 1
    }
    if (sum >= MODULUS) {
        sum -= MODULUS;
    }
    return sum;
}

// Modular subtraction
inline uint64_t mod_sub(uint64_t a, uint64_t b) {
    if (a >= b) {
        return a - b;
    }
    return MODULUS - (b - a);
}

// Modular multiplication using 128-bit intermediate
// Note: Metal doesn't have native 128-bit, so we use double-word arithmetic
inline uint64_t mod_mul(uint64_t a, uint64_t b) {
    // Split into 32-bit parts
    uint64_t a_lo = a & 0xFFFFFFFF;
    uint64_t a_hi = a >> 32;
    uint64_t b_lo = b & 0xFFFFFFFF;
    uint64_t b_hi = b >> 32;

    // Compute partial products
    uint64_t p0 = a_lo * b_lo;
    uint64_t p1 = a_lo * b_hi;
    uint64_t p2 = a_hi * b_lo;
    uint64_t p3 = a_hi * b_hi;

    // Combine (simplified reduction for Goldilocks)
    // For Goldilocks: 2^64 = 2^32 - 1 (mod p)
    uint64_t mid = p1 + p2;
    uint64_t carry = (mid < p1) ? 1ULL : 0ULL;

    uint64_t lo = p0 + (mid << 32);
    if (lo < p0) carry += 1;

    uint64_t hi = p3 + (mid >> 32) + carry;

    // Reduction: hi * 2^64 = hi * (2^32 - 1) mod p
    uint64_t reduction = (hi << 32) - hi;
    uint64_t result = mod_add(lo, reduction);

    // Final reduction if needed
    while (result >= MODULUS) {
        result -= MODULUS;
    }

    return result;
}

// Parallel reduction kernel - sums all elements
// Uses shared memory for efficient reduction within threadgroup
kernel void parallel_sum(
    device const uint64_t* input [[buffer(0)]],
    device uint64_t* output [[buffer(1)]],
    device const uint* params [[buffer(2)]],  // [0] = input_size
    threadgroup uint64_t* shared_mem [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint gid [[thread_position_in_grid]],
    uint group_size [[threads_per_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]]
) {
    uint input_size = params[0];

    // Load element (or zero if out of bounds)
    shared_mem[tid] = (gid < input_size) ? input[gid] : 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction within threadgroup
    for (uint stride = group_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_mem[tid] = mod_add(shared_mem[tid], shared_mem[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write result
    if (tid == 0) {
        output[group_id] = shared_mem[0];
    }
}

// Challenge application kernel
// Computes: new[k] = (1-r) * old[k] + r * old[k + half]
kernel void apply_challenge(
    device const uint64_t* input [[buffer(0)]],
    device uint64_t* output [[buffer(1)]],
    device const uint64_t* challenge [[buffer(2)]],  // [0] = r, [1] = 1-r
    device const uint* params [[buffer(3)]],  // [0] = half (output size)
    uint gid [[thread_position_in_grid]]
) {
    uint half = params[0];
    if (gid >= half) return;

    uint64_t r = challenge[0];
    uint64_t one_minus_r = challenge[1];

    uint64_t v0 = input[gid];
    uint64_t v1 = input[gid + half];

    // new = (1-r) * v0 + r * v1
    uint64_t term0 = mod_mul(one_minus_r, v0);
    uint64_t term1 = mod_mul(r, v1);
    output[gid] = mod_add(term0, term1);
}

// Compute round sums kernel
// Computes sum_0 = sum(evals[0..half]) and sum_1 = sum(evals[half..])
// Uses two-phase reduction for better memory access patterns
kernel void compute_round_sums(
    device const uint64_t* input [[buffer(0)]],
    device uint64_t* output [[buffer(1)]],  // [0] = partial sum_0, [1] = partial sum_1
    device const uint* params [[buffer(2)]],  // [0] = half, [1] = elements_per_thread
    threadgroup uint64_t* shared_mem [[threadgroup(0)]],  // [0..group_size] for sum_0, [group_size..] for sum_1
    uint tid [[thread_index_in_threadgroup]],
    uint gid [[thread_position_in_grid]],
    uint group_size [[threads_per_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]]
) {
    uint half = params[0];
    uint elems_per_thread = params[1];

    // Initialize shared memory
    shared_mem[tid] = 0;
    shared_mem[tid + group_size] = 0;

    // Each thread accumulates multiple elements
    uint start = gid * elems_per_thread;
    for (uint i = 0; i < elems_per_thread && (start + i) < half; i++) {
        uint idx = start + i;
        shared_mem[tid] = mod_add(shared_mem[tid], input[idx]);
        shared_mem[tid + group_size] = mod_add(shared_mem[tid + group_size], input[idx + half]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction
    for (uint stride = group_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_mem[tid] = mod_add(shared_mem[tid], shared_mem[tid + stride]);
            shared_mem[tid + group_size] = mod_add(shared_mem[tid + group_size], shared_mem[tid + group_size + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write partial results
    if (tid == 0) {
        output[group_id * 2] = shared_mem[0];
        output[group_id * 2 + 1] = shared_mem[group_size];
    }
}

// Batched polynomial evaluation for multi-factor sumcheck
// Computes products of corresponding elements across factors
kernel void compute_product(
    device const uint64_t* factors [[buffer(0)]],  // Interleaved: [f0[0], f1[0], f0[1], f1[1], ...]
    device uint64_t* output [[buffer(1)]],
    device const uint* params [[buffer(2)]],  // [0] = num_elements, [1] = num_factors
    uint gid [[thread_position_in_grid]]
) {
    uint num_elements = params[0];
    uint num_factors = params[1];

    if (gid >= num_elements) return;

    uint64_t product = 1;
    for (uint f = 0; f < num_factors; f++) {
        product = mod_mul(product, factors[gid * num_factors + f]);
    }
    output[gid] = product;
}
"#;

/// Shader source for BabyBear field (p = 2^31 - 2^27 + 1)
#[allow(dead_code)]
pub const BABYBEAR_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// BabyBear prime: 2^31 - 2^27 + 1 = 2013265921
constant uint32_t MODULUS = 2013265921u;

// Modular addition
inline uint32_t mod_add(uint32_t a, uint32_t b) {
    uint32_t sum = a + b;
    return (sum >= MODULUS) ? (sum - MODULUS) : sum;
}

// Modular subtraction
inline uint32_t mod_sub(uint32_t a, uint32_t b) {
    return (a >= b) ? (a - b) : (MODULUS - b + a);
}

// Modular multiplication
inline uint32_t mod_mul(uint32_t a, uint32_t b) {
    uint64_t prod = uint64_t(a) * uint64_t(b);
    return uint32_t(prod % MODULUS);
}

// Parallel sum kernel for BabyBear
kernel void parallel_sum_bb(
    device const uint32_t* input [[buffer(0)]],
    device uint32_t* output [[buffer(1)]],
    device const uint* params [[buffer(2)]],
    threadgroup uint32_t* shared_mem [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint gid [[thread_position_in_grid]],
    uint group_size [[threads_per_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]]
) {
    uint input_size = params[0];

    shared_mem[tid] = (gid < input_size) ? input[gid] : 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = group_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_mem[tid] = mod_add(shared_mem[tid], shared_mem[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        output[group_id] = shared_mem[0];
    }
}

// Apply challenge for BabyBear
kernel void apply_challenge_bb(
    device const uint32_t* input [[buffer(0)]],
    device uint32_t* output [[buffer(1)]],
    device const uint32_t* challenge [[buffer(2)]],
    device const uint* params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint half = params[0];
    if (gid >= half) return;

    uint32_t r = challenge[0];
    uint32_t one_minus_r = challenge[1];

    uint32_t v0 = input[gid];
    uint32_t v1 = input[gid + half];

    uint32_t term0 = mod_mul(one_minus_r, v0);
    uint32_t term1 = mod_mul(r, v1);
    output[gid] = mod_add(term0, term1);
}
"#;

/// Shader source for Mersenne31 field (p = 2^31 - 1)
#[allow(dead_code)]
pub const MERSENNE31_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Mersenne31 prime: 2^31 - 1
constant uint32_t MODULUS = 0x7FFFFFFF;

// Fast modular reduction for Mersenne primes
inline uint32_t mersenne_reduce(uint64_t x) {
    uint32_t lo = uint32_t(x & MODULUS);
    uint32_t hi = uint32_t(x >> 31);
    uint32_t sum = lo + hi;
    return (sum >= MODULUS) ? (sum - MODULUS) : sum;
}

inline uint32_t mod_add(uint32_t a, uint32_t b) {
    uint32_t sum = a + b;
    return (sum >= MODULUS) ? (sum - MODULUS) : sum;
}

inline uint32_t mod_sub(uint32_t a, uint32_t b) {
    return (a >= b) ? (a - b) : (MODULUS - b + a);
}

inline uint32_t mod_mul(uint32_t a, uint32_t b) {
    return mersenne_reduce(uint64_t(a) * uint64_t(b));
}

// Parallel sum kernel for Mersenne31
kernel void parallel_sum_m31(
    device const uint32_t* input [[buffer(0)]],
    device uint32_t* output [[buffer(1)]],
    device const uint* params [[buffer(2)]],
    threadgroup uint32_t* shared_mem [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint gid [[thread_position_in_grid]],
    uint group_size [[threads_per_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]]
) {
    uint input_size = params[0];

    shared_mem[tid] = (gid < input_size) ? input[gid] : 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = group_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_mem[tid] = mod_add(shared_mem[tid], shared_mem[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        output[group_id] = shared_mem[0];
    }
}

// Apply challenge for Mersenne31
kernel void apply_challenge_m31(
    device const uint32_t* input [[buffer(0)]],
    device uint32_t* output [[buffer(1)]],
    device const uint32_t* challenge [[buffer(2)]],
    device const uint* params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint half = params[0];
    if (gid >= half) return;

    uint32_t r = challenge[0];
    uint32_t one_minus_r = challenge[1];

    uint32_t v0 = input[gid];
    uint32_t v1 = input[gid + half];

    uint32_t term0 = mod_mul(one_minus_r, v0);
    uint32_t term1 = mod_mul(r, v1);
    output[gid] = mod_add(term0, term1);
}
"#;
