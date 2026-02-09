// Poseidon hash implementation for CUDA (Stark252 field)
//
// This implements the Cairo/Starknet Poseidon hash using the Hades permutation.
// The round constants are passed in as device memory from Rust.
//
// Parameters:
// - State size: 3 (rate=2, capacity=1)
// - Alpha (S-box): 3
// - Full rounds: 8 (4 before, 4 after partial)
// - Partial rounds: 83

#ifndef poseidon_h
#define poseidon_h

#include "../../../../../math/src/gpu/cuda/shaders/field/fp_u256.cuh"

namespace poseidon {

// Use Stark252 field
using Fp = p256::Fp;

// Poseidon parameters
constexpr int STATE_SIZE = 3;
constexpr int FULL_ROUNDS = 8;
constexpr int PARTIAL_ROUNDS = 83;
constexpr int HALF_FULL_ROUNDS = FULL_ROUNDS / 2;
constexpr int N_ROUND_CONSTANTS = 107;
#define POSEIDON_N_ROUND_CONSTANTS 107

// S-box: x^3
__device__ inline Fp sbox(const Fp& x) {
    return x * x * x;
}

// Optimized MDS mixing for state size 3
// Equivalent to multiplying by the MDS matrix but faster
// t = s[0] + s[1] + s[2]
// s[0]' = t + 2*s[0]
// s[1]' = t - 2*s[1]
// s[2]' = t - 3*s[2]
__device__ inline void mix(Fp* state) {
    Fp t = state[0] + state[1] + state[2];
    Fp s0_doubled = state[0] + state[0];
    Fp s1_doubled = state[1] + state[1];
    Fp s2_tripled = state[2] + state[2] + state[2];

    state[0] = t + s0_doubled;
    state[1] = t - s1_doubled;
    state[2] = t - s2_tripled;
}

// Full round: add constants to all state elements, apply S-box to all, then mix
__device__ inline void full_round(Fp* state, const Fp* round_constants, int round_idx) {
    // Add round constants and apply S-box to all elements
    #pragma unroll
    for (int i = 0; i < STATE_SIZE; i++) {
        state[i] = state[i] + round_constants[round_idx * STATE_SIZE + i];
        state[i] = sbox(state[i]);
    }
    // MDS mixing
    mix(state);
}

// Partial round (optimized): add constant only to position 2, apply S-box only there, then mix
// Uses the optimized constants format from the Poseidon paper
__device__ inline void partial_round(Fp* state, const Fp* round_constants, int const_idx) {
    // Add constant and apply S-box only to state[2]
    state[2] = state[2] + round_constants[const_idx];
    state[2] = sbox(state[2]);
    // MDS mixing
    mix(state);
}

// Hades permutation: the core of Poseidon
// round_constants: array of 107 field elements (optimized format)
__device__ void hades_permutation(Fp* state, const Fp* round_constants) {
    int const_idx = 0;

    // First half of full rounds (4 rounds)
    #pragma unroll
    for (int r = 0; r < HALF_FULL_ROUNDS; r++) {
        full_round(state, round_constants, r);
    }
    const_idx = HALF_FULL_ROUNDS * STATE_SIZE;  // 12

    // Partial rounds (83 rounds)
    // In optimized form, we only add one constant per partial round
    #pragma unroll 4
    for (int r = 0; r < PARTIAL_ROUNDS; r++) {
        partial_round(state, round_constants, const_idx);
        const_idx++;
    }
    // const_idx is now 12 + 83 = 95

    // Second half of full rounds (4 rounds)
    // The optimized constants for these are stored at the end of the array
    #pragma unroll
    for (int r = 0; r < HALF_FULL_ROUNDS; r++) {
        // For the last full rounds, constants are at the end of the array
        int base = POSEIDON_N_ROUND_CONSTANTS - (HALF_FULL_ROUNDS - r) * STATE_SIZE;
        #pragma unroll
        for (int i = 0; i < STATE_SIZE; i++) {
            state[i] = state[i] + round_constants[base + i];
            state[i] = sbox(state[i]);
        }
        mix(state);
    }
}

// Hash two field elements (for Merkle tree parent computation)
// Returns hash in state[0]
__device__ Fp hash_pair(const Fp& left, const Fp& right, const Fp* round_constants) {
    Fp state[STATE_SIZE];
    state[0] = left;
    state[1] = right;
    state[2] = Fp::from_int(2);  // Domain separator for 2-to-1 hash (Montgomery form)

    hades_permutation(state, round_constants);

    return state[0];
}

// Hash a single field element (for leaf hashing)
// Returns hash in state[0]
__device__ Fp hash_single(const Fp& input, const Fp* round_constants) {
    Fp state[STATE_SIZE];
    state[0] = input;
    state[1] = Fp(0);  // Montgomery form of 0 is 0
    state[2] = Fp::from_int(1);  // Domain separator for 1-to-1 hash (Montgomery form)

    hades_permutation(state, round_constants);

    return state[0];
}

} // namespace poseidon

#endif // poseidon_h
