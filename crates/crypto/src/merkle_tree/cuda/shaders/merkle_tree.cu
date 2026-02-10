// CUDA kernels for Merkle tree construction using Poseidon hash
//
// This file implements GPU-accelerated Merkle tree building with two core kernels:
//
// 1. hash_leaves: Parallel hashing of all leaf data (initial layer)
// 2. compute_parents: Parallel computation of parent nodes from child pairs
//
// The tree is built bottom-up, layer by layer:
//   [Leaves] → hash_leaves → [Hashed Leaves]
//   [Layer N] → compute_parents → [Layer N-1]
//   ...repeat until single root node remains
//
// Performance characteristics:
// - Embarrassingly parallel: no thread communication required
// - Memory-bound for large trees (limited by global memory bandwidth)
// - Compute-bound for small trees (Poseidon hash dominates)
// - Typical speedup: 5-10x vs CPU for trees with >16K leaves

#include "./poseidon.cuh"

using Fp = poseidon::Fp;

extern "C" {

/**
 * Hash all leaf data in parallel using Poseidon hash function.
 *
 * Each thread independently computes one hash, providing maximum parallelism
 * for the initial tree layer construction.
 *
 * Thread Model:
 * - One thread per leaf (independent, no synchronization needed)
 * - Grid stride not used (assumes num_leaves <= max_grid_size * block_size)
 *
 * Memory Access Pattern:
 * - Input:  Coalesced reads (thread i reads input[i])
 * - Output: Coalesced writes (thread i writes output[i])
 * - Constants: Broadcast reads (all threads read same round constants)
 *
 * Launch Configuration:
 * - Block size: 32 (one warp) recommended for efficiency
 * - Grid size: (num_leaves + block_size - 1) / block_size
 *
 * @param input          Array of unhashed field elements [num_leaves]
 * @param output         Array for hashed results [num_leaves] (output buffer)
 * @param round_constants Poseidon round constants [107 elements] (constant memory)
 * @param num_leaves     Total number of leaves to hash
 *
 * Preconditions:
 * - All pointers must be valid device memory
 * - input and output must not alias
 * - round_constants must contain exactly 107 valid field elements
 *
 * Postconditions:
 * - output[i] = PoseidonHash(input[i]) for all i < num_leaves
 */
__global__ void hash_leaves(
    const Fp* input,
    Fp* output,
    const Fp* round_constants,
    const unsigned int num_leaves
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Bounds check: handle case where num_leaves is not multiple of block_size
    if (idx >= num_leaves) return;

    // Independent hash computation (no thread interaction)
    output[idx] = poseidon::hash_single(input[idx], round_constants);
}

/**
 * Compute parent layer from child node pairs using Poseidon hash.
 *
 * This kernel implements the core Merkle tree construction step: hashing pairs
 * of child nodes to produce the parent layer. Called repeatedly to build the
 * tree bottom-up until only the root remains.
 *
 * Thread Model:
 * - One thread per parent node (each thread processes one pair)
 * - Thread i computes: parents[i] = Hash(children[2*i], children[2*i+1])
 * - No synchronization needed (independent computations)
 *
 * Memory Access Pattern:
 * - Children: Non-coalesced reads (thread i reads children[2*i] and children[2*i+1])
 *   Note: Adjacent threads read interleaved data, but cache line reuse is good
 * - Parents: Coalesced writes (thread i writes parents[i])
 * - Constants: Broadcast reads (all threads read same round constants)
 *
 * Launch Configuration:
 * - Block size: 32 (one warp) recommended
 * - Grid size: (num_parents + block_size - 1) / block_size
 *
 * Example:
 *   Children: [A, B, C, D, E, F, G, H]  (8 nodes)
 *   Parents:  [Hash(A,B), Hash(C,D), Hash(E,F), Hash(G,H)]  (4 nodes)
 *
 *   Thread 0: reads children[0,1], writes parents[0]
 *   Thread 1: reads children[2,3], writes parents[1]
 *   Thread 2: reads children[4,5], writes parents[2]
 *   Thread 3: reads children[6,7], writes parents[3]
 *
 * @param children        Current layer nodes [2 * num_parents] (must be even length)
 * @param parents         Output parent layer [num_parents] (output buffer)
 * @param round_constants Poseidon round constants [107 elements]
 * @param num_parents     Number of parent nodes to compute (= len(children) / 2)
 *
 * Preconditions:
 * - All pointers must be valid device memory
 * - children must have exactly 2*num_parents elements
 * - children and parents must not alias
 * - round_constants must contain exactly 107 valid field elements
 *
 * Postconditions:
 * - parents[i] = PoseidonHash(children[2*i], children[2*i+1]) for all i < num_parents
 */
__global__ void compute_parents(
    const Fp* children,       // Current layer (either leaves or internal nodes)
    Fp* parents,              // Parent layer to compute
    const Fp* round_constants,
    const unsigned int num_parents
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Bounds check: handle case where num_parents is not multiple of block_size
    if (idx >= num_parents) return;

    // Calculate indices for left and right children
    // Each parent is hash of two consecutive children
    unsigned int left_idx = 2 * idx;
    unsigned int right_idx = 2 * idx + 1;

    // Compute parent hash (independent operation, no thread communication)
    parents[idx] = poseidon::hash_pair(children[left_idx], children[right_idx], round_constants);
}

} // extern "C"
