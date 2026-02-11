// Montgomery field arithmetic for 384-bit fields (CUDA)
// Ported from lambdaworks Metal MSM field operations
// Provides modular arithmetic for BLS12-381 base field Fq

#ifndef FP384_CUH
#define FP384_CUH

#include "bigint384.cuh"

// BLS12-381 prime field modulus p (little-endian limbs)
// p = 0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab
__device__ __constant__ BigInt384 BLS12_381_P = {{
    0xb9feffffffffaaabULL,
    0x1eabfffeb153ffffULL,
    0x6730d2a0f6b0f624ULL,
    0x64774b84f38512bfULL,
    0x4b1ba7b6434bacd7ULL,
    0x1a0111ea397fe69aULL
}};

// Montgomery parameter: -p^(-1) mod 2^64
__device__ __constant__ unsigned long long BLS12_381_INV = 0x89f3fffcfffcfffdULL;

// R^2 mod p where R = 2^384 (little-endian limbs)
__device__ __constant__ BigInt384 BLS12_381_R2 = {{
    0xf4df1f341c341746ULL,
    0x0a76e6a609d104f1ULL,
    0x8de5476c4c95b6d5ULL,
    0x67eb88a9939d83c0ULL,
    0x9a793e85b519952dULL,
    0x11988fe592cae3aaULL
}};

// Montgomery reduction: given T (up to 2*p bits), compute T * R^(-1) mod p
__device__ __forceinline__ BigInt384 mont_reduce_384(BigIntWide384 t, const BigInt384 &p, unsigned long long inv) {
    BigIntWide384 tmp = t;

    for (unsigned i = 0; i < NUM_LIMBS_384; i++) {
        unsigned long long m = tmp.limbs[i] * inv;

        // Add m * p to tmp, shifted by i limbs
        unsigned long long carry = 0;
        for (unsigned j = 0; j < NUM_LIMBS_384; j++) {
            // tmp.limbs[i + j] += m * p.limbs[j] + carry
            unsigned long long hi = __umul64hi(m, p.limbs[j]);
            unsigned long long lo = m * p.limbs[j];

            unsigned long long old = tmp.limbs[i + j];
            unsigned long long sum1 = old + lo;
            unsigned long long c1 = (sum1 < old) ? 1ULL : 0ULL;
            unsigned long long sum2 = sum1 + carry;
            unsigned long long c2 = (sum2 < sum1) ? 1ULL : 0ULL;
            tmp.limbs[i + j] = sum2;
            carry = hi + c1 + c2;
        }

        // Propagate carry
        for (unsigned j = NUM_LIMBS_384; i + j < NUM_LIMBS_384 * 2; j++) {
            unsigned long long sum = tmp.limbs[i + j] + carry;
            carry = (sum < carry) ? 1ULL : 0ULL;
            tmp.limbs[i + j] = sum;
            if (carry == 0) break;
        }
    }

    // Result is in upper half
    BigInt384 result;
    for (unsigned i = 0; i < NUM_LIMBS_384; i++) {
        result.limbs[i] = tmp.limbs[i + NUM_LIMBS_384];
    }

    // Final reduction if result >= p
    unsigned long long borrow;
    BigInt384 reduced = bigint384_sub(result, p, borrow);
    if (borrow == 0) {
        return reduced;
    }
    return result;
}

// Montgomery multiplication: a * b * R^(-1) mod p
__device__ __forceinline__ BigInt384 mont_mul_384(const BigInt384 &a, const BigInt384 &b, const BigInt384 &p, unsigned long long inv) {
    BigIntWide384 product;
    for (unsigned i = 0; i < NUM_LIMBS_384 * 2; i++) {
        product.limbs[i] = 0;
    }

    for (unsigned i = 0; i < NUM_LIMBS_384; i++) {
        unsigned long long carry = 0;
        for (unsigned j = 0; j < NUM_LIMBS_384; j++) {
            unsigned long long hi = __umul64hi(a.limbs[i], b.limbs[j]);
            unsigned long long lo = a.limbs[i] * b.limbs[j];

            unsigned long long old = product.limbs[i + j];
            unsigned long long sum1 = old + lo;
            unsigned long long c1 = (sum1 < old) ? 1ULL : 0ULL;
            unsigned long long sum2 = sum1 + carry;
            unsigned long long c2 = (sum2 < sum1) ? 1ULL : 0ULL;
            product.limbs[i + j] = sum2;
            carry = hi + c1 + c2;
        }
        product.limbs[i + NUM_LIMBS_384] += carry;
    }

    return mont_reduce_384(product, p, inv);
}

// Montgomery squaring
__device__ __forceinline__ BigInt384 mont_square_384(const BigInt384 &a, const BigInt384 &p, unsigned long long inv) {
    return mont_mul_384(a, a, p, inv);
}

// Field addition: (a + b) mod p
__device__ __forceinline__ BigInt384 field_add_384(const BigInt384 &a, const BigInt384 &b, const BigInt384 &p) {
    unsigned long long carry;
    BigInt384 sum = bigint384_add(a, b, carry);

    unsigned long long borrow;
    BigInt384 reduced = bigint384_sub(sum, p, borrow);

    if (carry || borrow == 0) {
        return reduced;
    }
    return sum;
}

// Field subtraction: (a - b) mod p
__device__ __forceinline__ BigInt384 field_sub_384(const BigInt384 &a, const BigInt384 &b, const BigInt384 &p) {
    unsigned long long borrow;
    BigInt384 diff = bigint384_sub(a, b, borrow);

    if (borrow) {
        unsigned long long carry;
        diff = bigint384_add(diff, p, carry);
    }

    return diff;
}

// Field negation: -a mod p
__device__ __forceinline__ BigInt384 field_neg_384(const BigInt384 &a, const BigInt384 &p) {
    if (bigint384_is_zero(a)) return a;
    unsigned long long borrow;
    return bigint384_sub(p, a, borrow);
}

// Field doubling: 2a mod p
__device__ __forceinline__ BigInt384 field_double_384(const BigInt384 &a, const BigInt384 &p) {
    return field_add_384(a, a, p);
}

#endif /* FP384_CUH */
