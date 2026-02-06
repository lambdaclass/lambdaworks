//
// Metal MSM (Multi-Scalar Multiplication) Shaders for lambdaworks
//
// Implements Pippenger's algorithm with:
// - Montgomery arithmetic for field operations
// - Jacobian coordinates for point operations
// - Signed digit recoding for bucket reduction
//
// References:
// - cuZK paper for sparse matrix MSM approach
// - mopro Metal MSM for implementation patterns
// - Explicit-Formulas Database for curve operation formulas
//

#include <metal_stdlib>
using namespace metal;

// =============================================================================
// Configuration Constants
// =============================================================================

// Number of 64-bit limbs for BLS12-381 base field Fq (381 bits → 6 × 64 = 384 bits)
constant uint NUM_LIMBS = 6;

// Limbs per Jacobian point coordinate (x, y, z each have NUM_LIMBS limbs)
constant uint LIMBS_PER_COORD = NUM_LIMBS;
constant uint COORDS_PER_POINT = 3;
constant uint LIMBS_PER_POINT = COORDS_PER_POINT * LIMBS_PER_COORD;

// =============================================================================
// BigInt Types and Operations (256-bit unsigned integers)
// =============================================================================

struct BigInt {
    ulong limbs[NUM_LIMBS];
};

struct BigIntWide {
    ulong limbs[NUM_LIMBS * 2];
};

// Zero constant
constant BigInt BIGINT_ZERO = {{0, 0, 0, 0, 0, 0}};

// Check if BigInt is zero
bool bigint_is_zero(BigInt a) {
    for (uint i = 0; i < NUM_LIMBS; i++) {
        if (a.limbs[i] != 0) return false;
    }
    return true;
}

// BigInt addition with carry, returns (result, carry)
BigInt bigint_add(BigInt a, BigInt b, thread ulong& carry_out) {
    BigInt result;
    ulong carry = 0;

    for (uint i = 0; i < NUM_LIMBS; i++) {
        ulong sum = a.limbs[i] + b.limbs[i] + carry;
        // Detect overflow: sum < a.limbs[i] means overflow occurred
        carry = (sum < a.limbs[i]) || (carry && sum == a.limbs[i]) ? 1 : 0;
        result.limbs[i] = sum;
    }

    carry_out = carry;
    return result;
}

// BigInt subtraction with borrow, returns (result, borrow)
BigInt bigint_sub(BigInt a, BigInt b, thread ulong& borrow_out) {
    BigInt result;
    ulong borrow = 0;

    for (uint i = 0; i < NUM_LIMBS; i++) {
        ulong diff1 = a.limbs[i] - b.limbs[i];
        ulong b1 = (a.limbs[i] < b.limbs[i]) ? 1UL : 0UL;
        ulong diff2 = diff1 - borrow;
        ulong b2 = (diff1 < borrow) ? 1UL : 0UL;
        result.limbs[i] = diff2;
        borrow = b1 + b2;
    }

    borrow_out = borrow;
    return result;
}

// Compare a >= b
bool bigint_gte(BigInt a, BigInt b) {
    for (int i = NUM_LIMBS - 1; i >= 0; i--) {
        if (a.limbs[i] > b.limbs[i]) return true;
        if (a.limbs[i] < b.limbs[i]) return false;
    }
    return true; // Equal
}

// =============================================================================
// Montgomery Field Arithmetic
// =============================================================================

// BLS12-381 prime field modulus p (little-endian limbs)
// p = 0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab
constant BigInt BLS12_381_P = {{
    0xb9feffffffffaaab,  // limb 0 (LSB)
    0x1eabfffeb153ffff,
    0x6730d2a0f6b0f624,
    0x64774b84f38512bf,
    0x4b1ba7b6434bacd7,
    0x1a0111ea397fe69a   // limb 5 (MSB)
}};

// Montgomery parameter: -p^(-1) mod 2^64
// Depends only on LSB of modulus, unchanged from 4-limb version
constant ulong BLS12_381_INV = 0x89f3fffcfffcfffd;

// R^2 mod p where R = 2^384 (little-endian limbs)
constant BigInt BLS12_381_R2 = {{
    0xf4df1f341c341746,
    0x0a76e6a609d104f1,
    0x8de5476c4c95b6d5,
    0x67eb88a9939d83c0,
    0x9a793e85b519952d,
    0x11988fe592cae3aa
}};

// Montgomery reduction: given T (up to 2*p bits), compute T * R^(-1) mod p
BigInt mont_reduce(BigIntWide t, BigInt p, ulong inv) {
    BigInt result;
    BigIntWide tmp = t;

    for (uint i = 0; i < NUM_LIMBS; i++) {
        ulong m = tmp.limbs[i] * inv;

        // Add m * p to tmp, shifted by i limbs
        ulong carry = 0;
        for (uint j = 0; j < NUM_LIMBS; j++) {
            // tmp.limbs[i + j] += m * p.limbs[j] + carry
            ulong hi, lo;
            lo = m * p.limbs[j];
            hi = mulhi(m, p.limbs[j]);

            ulong old = tmp.limbs[i + j];
            ulong sum1 = old + lo;
            ulong c1 = (sum1 < old) ? 1UL : 0UL;
            ulong sum2 = sum1 + carry;
            ulong c2 = (sum2 < sum1) ? 1UL : 0UL;
            tmp.limbs[i + j] = sum2;
            carry = hi + c1 + c2;
        }

        // Propagate carry
        for (uint j = NUM_LIMBS; i + j < NUM_LIMBS * 2; j++) {
            ulong sum = tmp.limbs[i + j] + carry;
            carry = (sum < carry) ? 1 : 0;
            tmp.limbs[i + j] = sum;
            if (carry == 0) break;
        }
    }

    // Result is in upper half
    for (uint i = 0; i < NUM_LIMBS; i++) {
        result.limbs[i] = tmp.limbs[i + NUM_LIMBS];
    }

    // Final reduction if result >= p
    ulong borrow;
    BigInt reduced = bigint_sub(result, p, borrow);
    if (borrow == 0) {
        return reduced;
    }
    return result;
}

// Montgomery multiplication: a * b * R^(-1) mod p
BigInt mont_mul(BigInt a, BigInt b, BigInt p, ulong inv) {
    BigIntWide product = {{0}};

    // Multiply a * b
    for (uint i = 0; i < NUM_LIMBS; i++) {
        ulong carry = 0;
        for (uint j = 0; j < NUM_LIMBS; j++) {
            ulong hi = mulhi(a.limbs[i], b.limbs[j]);
            ulong lo = a.limbs[i] * b.limbs[j];

            // Add lo to product[i+j] with carry tracking
            ulong old = product.limbs[i + j];
            ulong sum1 = old + lo;
            ulong c1 = (sum1 < old) ? 1UL : 0UL;
            ulong sum2 = sum1 + carry;
            ulong c2 = (sum2 < sum1) ? 1UL : 0UL;
            product.limbs[i + j] = sum2;
            carry = hi + c1 + c2;
        }
        product.limbs[i + NUM_LIMBS] += carry;
    }

    return mont_reduce(product, p, inv);
}

// Montgomery squaring (optimized)
BigInt mont_square(BigInt a, BigInt p, ulong inv) {
    return mont_mul(a, a, p, inv);
}

// Field addition: (a + b) mod p
BigInt field_add(BigInt a, BigInt b, BigInt p) {
    ulong carry;
    BigInt sum = bigint_add(a, b, carry);

    // Reduce if sum >= p
    ulong borrow;
    BigInt reduced = bigint_sub(sum, p, borrow);

    if (carry || borrow == 0) {
        return reduced;
    }
    return sum;
}

// Field subtraction: (a - b) mod p
BigInt field_sub(BigInt a, BigInt b, BigInt p) {
    ulong borrow;
    BigInt diff = bigint_sub(a, b, borrow);

    if (borrow) {
        ulong carry;
        diff = bigint_add(diff, p, carry);
    }

    return diff;
}

// Field negation: -a mod p
BigInt field_neg(BigInt a, BigInt p) {
    if (bigint_is_zero(a)) return a;
    ulong borrow;
    return bigint_sub(p, a, borrow);
}

// Field doubling: 2a mod p
BigInt field_double(BigInt a, BigInt p) {
    return field_add(a, a, p);
}

// =============================================================================
// Jacobian Point Operations
// =============================================================================

struct JacobianPoint {
    BigInt x;
    BigInt y;
    BigInt z;
};

// Point at infinity (identity element)
JacobianPoint jacobian_identity() {
    JacobianPoint p;
    p.x = BIGINT_ZERO;
    p.y = BIGINT_ZERO;
    p.z = BIGINT_ZERO;
    // Convention: (0:1:0) or all zeros depending on implementation
    // We use z=0 to indicate identity
    return p;
}

// Check if point is identity (z == 0)
bool jacobian_is_identity(JacobianPoint p) {
    return bigint_is_zero(p.z);
}

// Point doubling using 2009-l formula from EFD
// Cost: 1M + 5S + 1*a + 7add + 2*2 + 1*3 + 1*8
// Source: https://hyperelliptic.org/EFD/g1p/auto-shortw-jacobian.html#doubling-dbl-2009-l
JacobianPoint jacobian_double(JacobianPoint p, BigInt field_p, ulong inv) {
    if (jacobian_is_identity(p)) {
        return p;
    }

    // A = X1^2
    BigInt A = mont_square(p.x, field_p, inv);
    // B = Y1^2
    BigInt B = mont_square(p.y, field_p, inv);
    // C = B^2
    BigInt C = mont_square(B, field_p, inv);

    // D = 2*((X1+B)^2-A-C)
    BigInt tmp = field_add(p.x, B, field_p);
    tmp = mont_square(tmp, field_p, inv);
    tmp = field_sub(tmp, A, field_p);
    tmp = field_sub(tmp, C, field_p);
    BigInt D = field_double(tmp, field_p);

    // E = 3*A
    BigInt E = field_add(A, field_double(A, field_p), field_p);

    // F = E^2
    BigInt F = mont_square(E, field_p, inv);

    // X3 = F-2*D
    BigInt X3 = field_sub(F, field_double(D, field_p), field_p);

    // Y3 = E*(D-X3)-8*C
    BigInt Y3 = field_sub(D, X3, field_p);
    Y3 = mont_mul(E, Y3, field_p, inv);
    BigInt C8 = field_double(field_double(field_double(C, field_p), field_p), field_p);
    Y3 = field_sub(Y3, C8, field_p);

    // Z3 = 2*Y1*Z1
    BigInt Z3 = mont_mul(p.y, p.z, field_p, inv);
    Z3 = field_double(Z3, field_p);

    JacobianPoint result;
    result.x = X3;
    result.y = Y3;
    result.z = Z3;
    return result;
}

// Point addition using 2007-bl formula from EFD
// Cost: 11M + 5S + 9add + 4*2
// Source: https://hyperelliptic.org/EFD/g1p/auto-shortw-jacobian.html#addition-add-2007-bl
JacobianPoint jacobian_add(JacobianPoint p, JacobianPoint q, BigInt field_p, ulong inv) {
    if (jacobian_is_identity(p)) return q;
    if (jacobian_is_identity(q)) return p;

    // Z1Z1 = Z1^2
    BigInt Z1Z1 = mont_square(p.z, field_p, inv);
    // Z2Z2 = Z2^2
    BigInt Z2Z2 = mont_square(q.z, field_p, inv);

    // U1 = X1*Z2Z2
    BigInt U1 = mont_mul(p.x, Z2Z2, field_p, inv);
    // U2 = X2*Z1Z1
    BigInt U2 = mont_mul(q.x, Z1Z1, field_p, inv);

    // S1 = Y1*Z2*Z2Z2
    BigInt S1 = mont_mul(p.y, q.z, field_p, inv);
    S1 = mont_mul(S1, Z2Z2, field_p, inv);
    // S2 = Y2*Z1*Z1Z1
    BigInt S2 = mont_mul(q.y, p.z, field_p, inv);
    S2 = mont_mul(S2, Z1Z1, field_p, inv);

    // H = U2-U1
    BigInt H = field_sub(U2, U1, field_p);

    // Handle P == Q case: when H == 0, the addition formula degenerates.
    // If H == 0 and S1 == S2, the points are equal → use doubling.
    // If H == 0 and S1 != S2, the points are inverses → return identity.
    if (bigint_is_zero(H)) {
        BigInt S_diff = field_sub(S2, S1, field_p);
        if (bigint_is_zero(S_diff)) {
            return jacobian_double(p, field_p, inv);
        } else {
            return jacobian_identity();
        }
    }

    // I = (2*H)^2
    BigInt I = field_double(H, field_p);
    I = mont_square(I, field_p, inv);
    // J = H*I
    BigInt J = mont_mul(H, I, field_p, inv);

    // r = 2*(S2-S1)
    BigInt r = field_sub(S2, S1, field_p);
    r = field_double(r, field_p);

    // V = U1*I
    BigInt V = mont_mul(U1, I, field_p, inv);

    // X3 = r^2-J-2*V
    BigInt X3 = mont_square(r, field_p, inv);
    X3 = field_sub(X3, J, field_p);
    X3 = field_sub(X3, field_double(V, field_p), field_p);

    // Y3 = r*(V-X3)-2*S1*J
    BigInt Y3 = field_sub(V, X3, field_p);
    Y3 = mont_mul(r, Y3, field_p, inv);
    BigInt tmp = mont_mul(S1, J, field_p, inv);
    tmp = field_double(tmp, field_p);
    Y3 = field_sub(Y3, tmp, field_p);

    // Z3 = ((Z1+Z2)^2-Z1Z1-Z2Z2)*H
    BigInt Z3 = field_add(p.z, q.z, field_p);
    Z3 = mont_square(Z3, field_p, inv);
    Z3 = field_sub(Z3, Z1Z1, field_p);
    Z3 = field_sub(Z3, Z2Z2, field_p);
    Z3 = mont_mul(Z3, H, field_p, inv);

    JacobianPoint result;
    result.x = X3;
    result.y = Y3;
    result.z = Z3;
    return result;
}

// Point negation: -(x, y, z) = (x, -y, z)
JacobianPoint jacobian_neg(JacobianPoint p, BigInt field_p) {
    JacobianPoint result;
    result.x = p.x;
    result.y = field_neg(p.y, field_p);
    result.z = p.z;
    return result;
}

// =============================================================================
// MSM Kernel: Bucket Accumulation
// =============================================================================

// Configuration struct passed from CPU
struct MSMConfig {
    uint num_scalars;
    uint num_windows;
    uint num_buckets;
    uint window_size;
};

// Load a Jacobian point from the buffer
JacobianPoint load_point(device const ulong* points, uint point_idx) {
    uint base = point_idx * LIMBS_PER_POINT;
    JacobianPoint p;

    for (uint i = 0; i < NUM_LIMBS; i++) {
        p.x.limbs[i] = points[base + i];
        p.y.limbs[i] = points[base + NUM_LIMBS + i];
        p.z.limbs[i] = points[base + 2 * NUM_LIMBS + i];
    }

    return p;
}

// Store a Jacobian point to the buffer
void store_point(device ulong* buffer, uint point_idx, JacobianPoint p) {
    uint base = point_idx * LIMBS_PER_POINT;

    for (uint i = 0; i < NUM_LIMBS; i++) {
        buffer[base + i] = p.x.limbs[i];
        buffer[base + NUM_LIMBS + i] = p.y.limbs[i];
        buffer[base + 2 * NUM_LIMBS + i] = p.z.limbs[i];
    }
}

// Load a Jacobian point from the buckets buffer
JacobianPoint load_bucket(device ulong* buckets, uint bucket_idx) {
    uint base = bucket_idx * LIMBS_PER_POINT;
    JacobianPoint p;

    for (uint i = 0; i < NUM_LIMBS; i++) {
        p.x.limbs[i] = buckets[base + i];
        p.y.limbs[i] = buckets[base + NUM_LIMBS + i];
        p.z.limbs[i] = buckets[base + 2 * NUM_LIMBS + i];
    }

    return p;
}

// Store a Jacobian point to the buckets buffer
void store_bucket(device ulong* buckets, uint bucket_idx, JacobianPoint p) {
    uint base = bucket_idx * LIMBS_PER_POINT;

    for (uint i = 0; i < NUM_LIMBS; i++) {
        buckets[base + i] = p.x.limbs[i];
        buckets[base + NUM_LIMBS + i] = p.y.limbs[i];
        buckets[base + 2 * NUM_LIMBS + i] = p.z.limbs[i];
    }
}

// Bucket accumulation kernel
// WARNING: This kernel has a RACE CONDITION when multiple threads write to the same bucket.
// For production use, implement one of:
// 1. Sorting-based approach (cuZK paper) - sort (bucket_idx, point) pairs, then scan
// 2. Atomic operations for point addition (complex for 256-bit)
// 3. Per-thread local buckets with tree reduction
//
// Current behavior: The last thread to write to a bucket wins, producing incorrect results
// when multiple points map to the same bucket.
kernel void bucket_accumulation(
    device const int* scalars [[buffer(0)]],      // Signed digits [num_scalars * num_windows]
    device const ulong* points [[buffer(1)]],     // Jacobian points [num_scalars * LIMBS_PER_POINT]
    device ulong* buckets [[buffer(2)]],          // Output buckets [num_windows * num_buckets * LIMBS_PER_POINT]
    device const uint* config [[buffer(3)]],      // MSMConfig
    uint gid [[thread_position_in_grid]]
) {
    uint num_scalars = config[0];
    uint num_windows = config[1];
    uint num_buckets = config[2];
    // uint window_size = config[3]; // Not needed in kernel

    // Each thread handles one (scalar, window) pair
    uint scalar_idx = gid / num_windows;
    uint window_idx = gid % num_windows;

    if (scalar_idx >= num_scalars) return;

    // Get the signed digit for this scalar and window
    int digit = scalars[scalar_idx * num_windows + window_idx];

    if (digit == 0) return; // Skip zero digits

    // Load the point
    JacobianPoint p = load_point(points, scalar_idx);

    // Determine bucket index and whether to negate
    uint bucket_idx;
    bool negate;
    if (digit > 0) {
        bucket_idx = uint(digit - 1);
        negate = false;
    } else {
        bucket_idx = uint(-digit - 1);
        negate = true;
    }

    // Negate point if needed
    if (negate) {
        p = jacobian_neg(p, BLS12_381_P);
    }

    // Calculate global bucket index
    uint global_bucket_idx = window_idx * num_buckets + bucket_idx;

    // Load current bucket value, add point, store back
    // WARNING: This read-modify-write is NOT atomic and causes race conditions
    // when multiple threads access the same bucket.
    JacobianPoint bucket = load_bucket(buckets, global_bucket_idx);
    bucket = jacobian_add(bucket, p, BLS12_381_P, BLS12_381_INV);
    store_bucket(buckets, global_bucket_idx, bucket);
}

// =============================================================================
// MSM Kernel: Bucket Reduction
// =============================================================================

// Reduce buckets within a window to a single point
// Uses the formula: sum = n*B[n-1] + (n-1)*B[n-2] + ... + 1*B[0]
//                       = B[n-1] + (B[n-1] + B[n-2]) + (B[n-1] + B[n-2] + B[n-3]) + ...
kernel void bucket_reduction(
    device ulong* buckets [[buffer(0)]],       // Input/output buckets
    device ulong* window_sums [[buffer(1)]],   // Output: one point per window
    device const uint* config [[buffer(2)]],   // [num_windows, num_buckets]
    uint window_idx [[thread_position_in_grid]]
) {
    uint num_windows = config[0];
    uint num_buckets = config[1];

    if (window_idx >= num_windows) return;

    uint bucket_base = window_idx * num_buckets;

    // Running sum for bucket reduction
    JacobianPoint running_sum = jacobian_identity();
    // Accumulated result
    JacobianPoint result = jacobian_identity();

    // Process buckets in reverse order (highest weight first)
    for (int i = int(num_buckets) - 1; i >= 0; i--) {
        JacobianPoint bucket = load_point(buckets, bucket_base + uint(i));

        // Add bucket to running sum
        running_sum = jacobian_add(running_sum, bucket, BLS12_381_P, BLS12_381_INV);

        // Add running sum to result
        result = jacobian_add(result, running_sum, BLS12_381_P, BLS12_381_INV);
    }

    // Store the window sum
    store_point(window_sums, window_idx, result);
}
