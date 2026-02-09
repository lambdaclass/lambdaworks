// CUDA kernels for Merkle tree construction using Poseidon hash
//
// The tree is built bottom-up, layer by layer:
// 1. hash_leaves: Hash all leaf data in parallel
// 2. compute_parents: Hash pairs of nodes to produce parent layer

#include "./poseidon.cuh"

using Fp = poseidon::Fp;

extern "C" {

// Hash all leaves in parallel
// Input: unhashed leaf data
// Output: hashed leaves
// round_constants: Poseidon round constants (107 elements)
__global__ void hash_leaves(
    const Fp* input,
    Fp* output,
    const Fp* round_constants,
    const unsigned int num_leaves
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_leaves) return;

    output[idx] = poseidon::hash_single(input[idx], round_constants);
}

// Compute parent nodes from pairs of children.
// Each thread hashes one pair: children[2*idx] and children[2*idx+1].
__global__ void compute_parents(
    const Fp* children,       // Current layer (either leaves or internal nodes)
    Fp* parents,              // Parent layer to compute
    const Fp* round_constants,
    const unsigned int num_parents
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_parents) return;

    unsigned int left_idx = 2 * idx;
    unsigned int right_idx = 2 * idx + 1;

    parents[idx] = poseidon::hash_pair(children[left_idx], children[right_idx], round_constants);
}

} // extern "C"
