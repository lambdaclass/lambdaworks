// 384-bit unsigned integer arithmetic for CUDA
// Adapted from lambdaworks Metal MSM BigInt/BigIntWide pattern
// for BLS12-381 base field (381 bits -> 6 x 64-bit limbs)

#ifndef BIGINT384_CUH
#define BIGINT384_CUH

#define NUM_LIMBS_384 6

struct BigInt384 {
    unsigned long long limbs[NUM_LIMBS_384];
};

struct BigIntWide384 {
    unsigned long long limbs[NUM_LIMBS_384 * 2];
};

__device__ __constant__ BigInt384 BIGINT384_ZERO = {{0, 0, 0, 0, 0, 0}};

__device__ __forceinline__ bool bigint384_is_zero(const BigInt384 &a) {
    for (unsigned i = 0; i < NUM_LIMBS_384; i++) {
        if (a.limbs[i] != 0) return false;
    }
    return true;
}

// Addition with carry out
__device__ __forceinline__ BigInt384 bigint384_add(const BigInt384 &a, const BigInt384 &b, unsigned long long &carry_out) {
    BigInt384 result;
    unsigned long long carry = 0;

    for (unsigned i = 0; i < NUM_LIMBS_384; i++) {
        unsigned long long sum = a.limbs[i] + b.limbs[i] + carry;
        carry = (sum < a.limbs[i]) || (carry && sum == a.limbs[i]) ? 1ULL : 0ULL;
        result.limbs[i] = sum;
    }

    carry_out = carry;
    return result;
}

// Subtraction with borrow out
__device__ __forceinline__ BigInt384 bigint384_sub(const BigInt384 &a, const BigInt384 &b, unsigned long long &borrow_out) {
    BigInt384 result;
    unsigned long long borrow = 0;

    for (unsigned i = 0; i < NUM_LIMBS_384; i++) {
        unsigned long long diff1 = a.limbs[i] - b.limbs[i];
        unsigned long long b1 = (a.limbs[i] < b.limbs[i]) ? 1ULL : 0ULL;
        unsigned long long diff2 = diff1 - borrow;
        unsigned long long b2 = (diff1 < borrow) ? 1ULL : 0ULL;
        result.limbs[i] = diff2;
        borrow = b1 + b2;
    }

    borrow_out = borrow;
    return result;
}

// Compare a >= b
__device__ __forceinline__ bool bigint384_gte(const BigInt384 &a, const BigInt384 &b) {
    for (int i = NUM_LIMBS_384 - 1; i >= 0; i--) {
        if (a.limbs[i] > b.limbs[i]) return true;
        if (a.limbs[i] < b.limbs[i]) return false;
    }
    return true; // equal
}

#endif /* BIGINT384_CUH */
