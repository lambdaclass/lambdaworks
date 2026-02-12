// Jacobian point operations over 384-bit fields (CUDA)
// Ported from lambdaworks Metal MSM point operations
// Uses EFD formulas for point doubling and addition

#ifndef JACOBIAN384_CUH
#define JACOBIAN384_CUH

#include "fp384.cuh"

#define COORDS_PER_POINT_384 3
#define LIMBS_PER_POINT_384 (COORDS_PER_POINT_384 * NUM_LIMBS_384)

struct JacobianPoint384 {
    BigInt384 x;
    BigInt384 y;
    BigInt384 z;
};

// Point at infinity (identity element): z = 0
__device__ __forceinline__ JacobianPoint384 jacobian_identity_384() {
    JacobianPoint384 p;
    p.x = BIGINT384_ZERO;
    p.y = BIGINT384_ZERO;
    p.z = BIGINT384_ZERO;
    return p;
}

// Check if point is identity (z == 0)
__device__ __forceinline__ bool jacobian_is_identity_384(const JacobianPoint384 &p) {
    return bigint384_is_zero(p.z);
}

// Point doubling using 2009-l formula from EFD
// https://hyperelliptic.org/EFD/g1p/auto-shortw-jacobian.html#doubling-dbl-2009-l
__device__ __forceinline__ JacobianPoint384 jacobian_double_384(const JacobianPoint384 &p, const BigInt384 &field_p, unsigned long long inv) {
    if (jacobian_is_identity_384(p)) {
        return p;
    }

    // A = X1^2
    BigInt384 A = mont_square_384(p.x, field_p, inv);
    // B = Y1^2
    BigInt384 B = mont_square_384(p.y, field_p, inv);
    // C = B^2
    BigInt384 C = mont_square_384(B, field_p, inv);

    // D = 2*((X1+B)^2-A-C)
    BigInt384 tmp = field_add_384(p.x, B, field_p);
    tmp = mont_square_384(tmp, field_p, inv);
    tmp = field_sub_384(tmp, A, field_p);
    tmp = field_sub_384(tmp, C, field_p);
    BigInt384 D = field_double_384(tmp, field_p);

    // E = 3*A
    BigInt384 E = field_add_384(A, field_double_384(A, field_p), field_p);

    // F = E^2
    BigInt384 F = mont_square_384(E, field_p, inv);

    // X3 = F-2*D
    BigInt384 X3 = field_sub_384(F, field_double_384(D, field_p), field_p);

    // Y3 = E*(D-X3)-8*C
    BigInt384 Y3 = field_sub_384(D, X3, field_p);
    Y3 = mont_mul_384(E, Y3, field_p, inv);
    BigInt384 C8 = field_double_384(field_double_384(field_double_384(C, field_p), field_p), field_p);
    Y3 = field_sub_384(Y3, C8, field_p);

    // Z3 = 2*Y1*Z1
    BigInt384 Z3 = mont_mul_384(p.y, p.z, field_p, inv);
    Z3 = field_double_384(Z3, field_p);

    JacobianPoint384 result;
    result.x = X3;
    result.y = Y3;
    result.z = Z3;
    return result;
}

// Point addition using 2007-bl formula from EFD
// https://hyperelliptic.org/EFD/g1p/auto-shortw-jacobian.html#addition-add-2007-bl
__device__ __forceinline__ JacobianPoint384 jacobian_add_384(const JacobianPoint384 &p, const JacobianPoint384 &q, const BigInt384 &field_p, unsigned long long inv) {
    if (jacobian_is_identity_384(p)) return q;
    if (jacobian_is_identity_384(q)) return p;

    // Z1Z1 = Z1^2
    BigInt384 Z1Z1 = mont_square_384(p.z, field_p, inv);
    // Z2Z2 = Z2^2
    BigInt384 Z2Z2 = mont_square_384(q.z, field_p, inv);

    // U1 = X1*Z2Z2
    BigInt384 U1 = mont_mul_384(p.x, Z2Z2, field_p, inv);
    // U2 = X2*Z1Z1
    BigInt384 U2 = mont_mul_384(q.x, Z1Z1, field_p, inv);

    // S1 = Y1*Z2*Z2Z2
    BigInt384 S1 = mont_mul_384(p.y, q.z, field_p, inv);
    S1 = mont_mul_384(S1, Z2Z2, field_p, inv);
    // S2 = Y2*Z1*Z1Z1
    BigInt384 S2 = mont_mul_384(q.y, p.z, field_p, inv);
    S2 = mont_mul_384(S2, Z1Z1, field_p, inv);

    // H = U2-U1
    BigInt384 H = field_sub_384(U2, U1, field_p);

    // Handle P == Q case
    if (bigint384_is_zero(H)) {
        BigInt384 S_diff = field_sub_384(S2, S1, field_p);
        if (bigint384_is_zero(S_diff)) {
            return jacobian_double_384(p, field_p, inv);
        } else {
            return jacobian_identity_384();
        }
    }

    // I = (2*H)^2
    BigInt384 I = field_double_384(H, field_p);
    I = mont_square_384(I, field_p, inv);
    // J = H*I
    BigInt384 J = mont_mul_384(H, I, field_p, inv);

    // r = 2*(S2-S1)
    BigInt384 r = field_sub_384(S2, S1, field_p);
    r = field_double_384(r, field_p);

    // V = U1*I
    BigInt384 V = mont_mul_384(U1, I, field_p, inv);

    // X3 = r^2-J-2*V
    BigInt384 X3 = mont_square_384(r, field_p, inv);
    X3 = field_sub_384(X3, J, field_p);
    X3 = field_sub_384(X3, field_double_384(V, field_p), field_p);

    // Y3 = r*(V-X3)-2*S1*J
    BigInt384 Y3 = field_sub_384(V, X3, field_p);
    Y3 = mont_mul_384(r, Y3, field_p, inv);
    BigInt384 tmp2 = mont_mul_384(S1, J, field_p, inv);
    tmp2 = field_double_384(tmp2, field_p);
    Y3 = field_sub_384(Y3, tmp2, field_p);

    // Z3 = ((Z1+Z2)^2-Z1Z1-Z2Z2)*H
    BigInt384 Z3 = field_add_384(p.z, q.z, field_p);
    Z3 = mont_square_384(Z3, field_p, inv);
    Z3 = field_sub_384(Z3, Z1Z1, field_p);
    Z3 = field_sub_384(Z3, Z2Z2, field_p);
    Z3 = mont_mul_384(Z3, H, field_p, inv);

    JacobianPoint384 result;
    result.x = X3;
    result.y = Y3;
    result.z = Z3;
    return result;
}

// Point negation: -(x, y, z) = (x, -y, z)
__device__ __forceinline__ JacobianPoint384 jacobian_neg_384(const JacobianPoint384 &p, const BigInt384 &field_p) {
    JacobianPoint384 result;
    result.x = p.x;
    result.y = field_neg_384(p.y, field_p);
    result.z = p.z;
    return result;
}

// Load a Jacobian point from a buffer
__device__ __forceinline__ JacobianPoint384 load_point_384(const unsigned long long *buffer, unsigned int point_idx) {
    unsigned int base = point_idx * LIMBS_PER_POINT_384;
    JacobianPoint384 p;

    for (unsigned i = 0; i < NUM_LIMBS_384; i++) {
        p.x.limbs[i] = buffer[base + i];
        p.y.limbs[i] = buffer[base + NUM_LIMBS_384 + i];
        p.z.limbs[i] = buffer[base + 2 * NUM_LIMBS_384 + i];
    }

    return p;
}

// Store a Jacobian point to a buffer
__device__ __forceinline__ void store_point_384(unsigned long long *buffer, unsigned int point_idx, const JacobianPoint384 &p) {
    unsigned int base = point_idx * LIMBS_PER_POINT_384;

    for (unsigned i = 0; i < NUM_LIMBS_384; i++) {
        buffer[base + i] = p.x.limbs[i];
        buffer[base + NUM_LIMBS_384 + i] = p.y.limbs[i];
        buffer[base + 2 * NUM_LIMBS_384 + i] = p.z.limbs[i];
    }
}

#endif /* JACOBIAN384_CUH */
