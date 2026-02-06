// Poseidon hash constants for Stark252 field (Cairo/Starknet parameters)
//
// Parameters:
// - State size: 3 (rate=2, capacity=1)
// - Alpha (S-box exponent): 3
// - Full rounds: 8 (4 before partial, 4 after)
// - Partial rounds: 83
//
// The round constants are in the optimized form from the Poseidon paper (Appendix B)

#ifndef poseidon_constants_h
#define poseidon_constants_h

#include "../../../../../../math/src/gpu/cuda/shaders/field/fp_u256.cuh"

// Poseidon parameters
#define POSEIDON_STATE_SIZE 3
#define POSEIDON_RATE 2
#define POSEIDON_CAPACITY 1
#define POSEIDON_ALPHA 3
#define POSEIDON_FULL_ROUNDS 8
#define POSEIDON_PARTIAL_ROUNDS 83
#define POSEIDON_N_ROUND_CONSTANTS 107

namespace poseidon_stark252 {

// StarkWare field for Cairo: p = 2^251 + 17 * 2^192 + 1
using Fp = p256::Fp;

// Round constants in Montgomery form
// These are the optimized constants from the Starknet Poseidon implementation
__constant__ u256 ROUND_CONSTANTS[POSEIDON_N_ROUND_CONSTANTS];

// Host-side constants initialization (called once at startup)
__host__ void init_round_constants();

} // namespace poseidon_stark252

#endif // poseidon_constants_h
