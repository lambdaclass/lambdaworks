// CUDA kernels for Merkle tree construction using Poseidon hash
//
// The tree is built bottom-up, layer by layer:
// 1. hash_leaves: Hash all leaf data in parallel
// 2. build_layer: Hash pairs of nodes to produce parent layer
//
// Tree layout in memory (for 8 leaves):
// [internal nodes: 7][leaf hashes: 8]
// Index 0 is root, indices 1-6 are internal, 7-14 are leaves

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

// Build one layer of the Merkle tree by hashing pairs of children
// children: array of child nodes (current layer)
// parents: array to store parent nodes (next layer up)
// round_constants: Poseidon round constants
// num_parents: number of parent nodes to compute (= num_children / 2)
__global__ void build_layer(
    const Fp* children,
    Fp* parents,
    const Fp* round_constants,
    const unsigned int num_parents
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_parents) return;

    // Each parent is hash of two consecutive children
    unsigned int child_idx = idx * 2;
    parents[idx] = poseidon::hash_pair(children[child_idx], children[child_idx + 1], round_constants);
}

// Build entire Merkle tree in-place
// nodes: array with leaves at the end, internal nodes will be filled in
// Layout: [internal: num_leaves-1][leaves: num_leaves]
// round_constants: Poseidon round constants
// num_leaves: number of leaves (must be power of 2)
//
// This kernel is called once per layer, iterating from leaves to root
__global__ void build_tree_layer(
    Fp* nodes,
    const Fp* round_constants,
    const unsigned int layer_start,    // Index where this layer starts
    const unsigned int layer_size      // Number of nodes in this layer
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= layer_size) return;

    // Child indices in the nodes array
    unsigned int parent_idx = layer_start + idx;
    unsigned int left_child_idx = 2 * parent_idx + 1;
    unsigned int right_child_idx = 2 * parent_idx + 2;

    // Hash children to produce parent
    nodes[parent_idx] = poseidon::hash_pair(nodes[left_child_idx], nodes[right_child_idx], round_constants);
}

// Simpler approach: work on a flat array where:
// - First call: leaves are hashed to produce first layer of internal nodes
// - Subsequent calls: each layer produces the parent layer
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
