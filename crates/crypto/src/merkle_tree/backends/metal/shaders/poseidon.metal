//! Metal shader for Poseidon hash on Stark252 prime field.
//!
//! This shader implements the Poseidon hash function used in StarkNet,
//! operating on the Stark252 prime field (p = 2^251 + 17 * 2^192 + 1).

#include <metal_stdlib>
using namespace metal;

// ============================================================
// 256-bit Unsigned Integer Type (4 x 64-bit limbs)
// ============================================================

struct U256 {
    ulong4 limbs; // limbs[0] is least significant

    U256() : limbs(0) {}
    U256(ulong4 l) : limbs(l) {}
    U256(ulong v) : limbs(ulong4(v, 0, 0, 0)) {}
};

// ============================================================
// Stark252 Prime Field Constants
// ============================================================

// p = 2^251 + 17 * 2^192 + 1
// = 0x800000000000011000000000000000000000000000000000000000000000001
constant ulong4 STARK_PRIME = ulong4(
    0x0000000000000001UL,  // limbs[0] (least significant): the +1
    0x0000000000000000UL,  // limbs[1]
    0x0000000000000000UL,  // limbs[2]
    0x0800000000000011UL   // limbs[3] (most significant): 2^251 + 17*2^192
);

// ============================================================
// 256-bit Arithmetic Operations
// ============================================================

/// Add two 256-bit integers, returns (result, carry)
inline U256 add_u256(U256 a, U256 b, thread bool& carry) {
    U256 result;
    ulong c = 0;

    // Limb 0
    ulong sum0 = a.limbs[0] + b.limbs[0];
    c = (sum0 < a.limbs[0]) ? 1 : 0;
    result.limbs[0] = sum0;

    // Limb 1
    ulong sum1 = a.limbs[1] + b.limbs[1] + c;
    c = (sum1 < a.limbs[1] || (c && sum1 == a.limbs[1])) ? 1 : 0;
    result.limbs[1] = sum1;

    // Limb 2
    ulong sum2 = a.limbs[2] + b.limbs[2] + c;
    c = (sum2 < a.limbs[2] || (c && sum2 == a.limbs[2])) ? 1 : 0;
    result.limbs[2] = sum2;

    // Limb 3
    ulong sum3 = a.limbs[3] + b.limbs[3] + c;
    c = (sum3 < a.limbs[3] || (c && sum3 == a.limbs[3])) ? 1 : 0;
    result.limbs[3] = sum3;

    carry = (c != 0);
    return result;
}

/// Subtract two 256-bit integers, returns (result, borrow)
inline U256 sub_u256(U256 a, U256 b, thread bool& borrow) {
    U256 result;
    ulong br = 0;

    // Limb 0
    result.limbs[0] = a.limbs[0] - b.limbs[0];
    br = (a.limbs[0] < b.limbs[0]) ? 1 : 0;

    // Limb 1: check borrow without overflow in (b.limbs[1] + br)
    result.limbs[1] = a.limbs[1] - b.limbs[1] - br;
    br = (a.limbs[1] < b.limbs[1] || (br && a.limbs[1] == b.limbs[1])) ? 1 : 0;

    // Limb 2
    result.limbs[2] = a.limbs[2] - b.limbs[2] - br;
    br = (a.limbs[2] < b.limbs[2] || (br && a.limbs[2] == b.limbs[2])) ? 1 : 0;

    // Limb 3
    result.limbs[3] = a.limbs[3] - b.limbs[3] - br;
    br = (a.limbs[3] < b.limbs[3] || (br && a.limbs[3] == b.limbs[3])) ? 1 : 0;

    borrow = (br != 0);
    return result;
}

/// Compare a >= b
inline bool gte_u256(U256 a, U256 b) {
    if (a.limbs[3] != b.limbs[3]) return a.limbs[3] > b.limbs[3];
    if (a.limbs[2] != b.limbs[2]) return a.limbs[2] > b.limbs[2];
    if (a.limbs[1] != b.limbs[1]) return a.limbs[1] > b.limbs[1];
    return a.limbs[0] >= b.limbs[0];
}

// ============================================================
// Stark252 Field Operations
// ============================================================

/// Reduce a 256-bit value modulo the Stark prime
inline U256 reduce_stark(U256 a) {
    U256 prime;
    prime.limbs = STARK_PRIME;

    while (gte_u256(a, prime)) {
        bool borrow;
        a = sub_u256(a, prime, borrow);
    }
    return a;
}

/// Modular addition in Stark252 field
inline U256 add_mod(U256 a, U256 b) {
    bool carry;
    U256 sum = add_u256(a, b, carry);

    U256 prime;
    prime.limbs = STARK_PRIME;

    if (carry || gte_u256(sum, prime)) {
        bool borrow;
        sum = sub_u256(sum, prime, borrow);
    }
    return sum;
}

/// Modular subtraction in Stark252 field
inline U256 sub_mod(U256 a, U256 b) {
    bool borrow;
    U256 diff = sub_u256(a, b, borrow);

    if (borrow) {
        bool carry;
        U256 prime;
        prime.limbs = STARK_PRIME;
        diff = add_u256(diff, prime, carry);
    }
    return diff;
}

/// Double a field element (2 * a)
inline U256 double_mod(U256 a) {
    return add_mod(a, a);
}

/// Modular multiplication using binary double-and-add method.
/// Computes a * b mod p by iterating over bits of b.
/// This is a correct reference implementation; a production version
/// would use Montgomery or Barrett reduction for better performance.
inline U256 mul_mod(U256 a, U256 b) {
    U256 result = U256(0);
    U256 current = a;

    // Iterate over bits of b (Stark252 values are < p < 2^252)
    for (int bit = 0; bit < 252; bit++) {
        int limb_idx = bit / 64;
        int bit_idx = bit % 64;

        if ((b.limbs[limb_idx] >> bit_idx) & 1UL) {
            result = add_mod(result, current);
        }
        current = double_mod(current);
    }

    return result;
}

/// Cube a field element (x^3) - used in Poseidon S-box
inline U256 cube_mod(U256 a) {
    U256 a2 = mul_mod(a, a);
    return mul_mod(a2, a);
}

// ============================================================
// Poseidon Round Constants (first few for reference)
// Full constants would be loaded from buffer
// ============================================================

// State size for Poseidon Cairo (rate=2, capacity=1)
constant int STATE_SIZE = 3;
constant int N_FULL_ROUNDS = 8;
constant int N_PARTIAL_ROUNDS = 83;

// ============================================================
// Poseidon Permutation
// ============================================================

/// Optimized mix function for Poseidon Cairo Stark252
/// t = s0 + s1 + s2
/// s0' = t + 2*s0
/// s1' = t - 2*s1
/// s2' = t - 3*s2
inline void mix(thread U256* state) {
    U256 t = add_mod(add_mod(state[0], state[1]), state[2]);

    U256 s0_new = add_mod(t, double_mod(state[0]));
    U256 s1_new = sub_mod(t, double_mod(state[1]));
    U256 s2_double = double_mod(state[2]);
    U256 s2_triple = add_mod(s2_double, state[2]);
    U256 s2_new = sub_mod(t, s2_triple);

    state[0] = s0_new;
    state[1] = s1_new;
    state[2] = s2_new;
}

/// Full round: add constants, apply S-box to all, mix
inline void full_round(
    thread U256* state,
    device const U256* round_constants,
    int rc_index
) {
    // Add round constants and apply S-box (x^3)
    for (int i = 0; i < STATE_SIZE; i++) {
        state[i] = add_mod(state[i], round_constants[rc_index + i]);
        state[i] = cube_mod(state[i]);
    }
    mix(state);
}

/// Partial round: add constant to last element, apply S-box to last, mix
inline void partial_round(
    thread U256* state,
    device const U256* round_constants,
    int rc_index
) {
    // Add constant and apply S-box only to state[2]
    state[2] = add_mod(state[2], round_constants[rc_index]);
    state[2] = cube_mod(state[2]);
    mix(state);
}

/// Poseidon permutation (Hades design)
inline void hades_permutation(
    thread U256* state,
    device const U256* round_constants
) {
    int index = 0;

    // First half of full rounds
    for (int r = 0; r < N_FULL_ROUNDS / 2; r++) {
        full_round(state, round_constants, index);
        index += STATE_SIZE;
    }

    // Partial rounds
    for (int r = 0; r < N_PARTIAL_ROUNDS; r++) {
        partial_round(state, round_constants, index);
        index += 1;
    }

    // Second half of full rounds
    for (int r = 0; r < N_FULL_ROUNDS / 2; r++) {
        full_round(state, round_constants, index);
        index += STATE_SIZE;
    }
}

// ============================================================
// Merkle Tree Hash Kernels
// ============================================================

/// Hash two field elements (for parent node computation)
/// hash(x, y) = permutation([x, y, 2])[0]
kernel void hash_pair(
    device const U256* left [[buffer(0)]],
    device const U256* right [[buffer(1)]],
    device U256* output [[buffer(2)]],
    device const U256* round_constants [[buffer(3)]],
    constant uint& count [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;

    U256 state[STATE_SIZE];
    state[0] = left[gid];
    state[1] = right[gid];
    state[2] = U256(2);  // Domain separator for hash(x, y)

    hades_permutation(state, round_constants);

    output[gid] = state[0];
}

/// Hash a single field element (for leaf hashing)
/// hash_single(x) = permutation([x, 0, 1])[0]
kernel void hash_single(
    device const U256* input [[buffer(0)]],
    device U256* output [[buffer(1)]],
    device const U256* round_constants [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;

    U256 state[STATE_SIZE];
    state[0] = input[gid];
    state[1] = U256(0);
    state[2] = U256(1);  // Domain separator for hash_single

    hades_permutation(state, round_constants);

    output[gid] = state[0];
}

/// Build one level of the Merkle tree (hash pairs of nodes)
/// Takes 2N input nodes and produces N output nodes
kernel void merkle_hash_level(
    device const U256* input [[buffer(0)]],
    device U256* output [[buffer(1)]],
    device const U256* round_constants [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;

    uint left_idx = gid * 2;
    uint right_idx = left_idx + 1;

    U256 state[STATE_SIZE];
    state[0] = input[left_idx];
    state[1] = input[right_idx];
    state[2] = U256(2);

    hades_permutation(state, round_constants);

    output[gid] = state[0];
}
