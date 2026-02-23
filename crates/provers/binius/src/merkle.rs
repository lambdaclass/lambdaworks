//! Merkle Tree implementation for Binius
//!
//! Provides a production-ready Merkle tree with inclusion proofs using
//! SHA-256 for hashing.

use crate::fields::tower::Tower;
use sha2::{Digest, Sha256};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MerkleNode([u8; 32]);

impl MerkleNode {
    pub fn new(data: &[u8]) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(data);
        let result = hasher.finalize();
        let mut node = [0u8; 32];
        node.copy_from_slice(&result);
        Self(node)
    }

    pub fn from_values(values: &[Tower]) -> Self {
        let mut data = Vec::with_capacity(values.len() * 16);
        for v in values {
            data.extend_from_slice(&v.value().to_le_bytes());
        }
        Self::new(&data)
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }

    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    pub fn to_array(&self) -> [u8; 32] {
        self.0
    }
}

impl Default for MerkleNode {
    fn default() -> Self {
        Self([0u8; 32])
    }
}

/// Merkle tree with proofs
pub struct MerkleTree {
    nodes: Vec<MerkleNode>,
    depth: usize,
    leaf_count: usize,
}

impl MerkleTree {
    pub fn build(leaves: &[Tower]) -> Option<Self> {
        if leaves.is_empty() {
            return None;
        }

        let leaf_count = leaves.len().next_power_of_two();
        let depth = (leaf_count as f64).log2() as usize + 1;

        // Build leaf hashes
        let mut nodes: Vec<MerkleNode> = leaves
            .iter()
            .map(|l| MerkleNode::new(&l.value().to_le_bytes()))
            .collect();

        // Pad to power of 2
        while nodes.len() < leaf_count {
            nodes.push(MerkleNode::default());
        }

        // Build tree from bottom up, storing in level order
        // For a complete binary tree with N leaves: total nodes = 2*N - 1
        // Level order indices: root=0, children=1,2, grandchildren=3,4,5,6, leaves at N-1...

        // But simpler: let's just build it level by level and store [leaves, parents..., root]
        let mut level = nodes.clone();
        let mut all_levels: Vec<Vec<MerkleNode>> = vec![level.clone()];

        while level.len() > 1 {
            let mut next_level = Vec::with_capacity(level.len() / 2);
            for i in (0..level.len()).step_by(2) {
                let left = &level[i];
                let right = if i + 1 < level.len() {
                    &level[i + 1]
                } else {
                    left
                };
                let mut combined = [0u8; 64];
                combined[..32].copy_from_slice(&left.0);
                combined[32..].copy_from_slice(&right.0);
                next_level.push(MerkleNode::new(&combined));
            }
            all_levels.push(next_level.clone());
            level = next_level;
        }

        // Now build final array: root first, then parents, then leaves
        // all_levels[last] = root, all_levels[last-1] = parents, ..., all_levels[0] = leaves
        let mut final_nodes = Vec::with_capacity(2 * leaf_count - 1);

        // Add from last to first
        for level in all_levels.into_iter().rev() {
            final_nodes.extend(level);
        }

        Some(Self {
            nodes: final_nodes,
            depth,
            leaf_count,
        })
    }

    pub fn root(&self) -> &MerkleNode {
        &self.nodes[0]
    }

    pub fn depth(&self) -> usize {
        self.depth
    }

    pub fn get_leaf(&self, index: usize) -> Option<&MerkleNode> {
        // Leaf nodes start at index: 2^(depth-1) - 1 = leaf_count - 1
        let leaf_start = (1 << (self.depth - 1)) - 1;
        self.nodes.get(leaf_start + index)
    }

    pub fn get_proof(&self, index: usize) -> Option<MerkleProof> {
        if index >= self.leaf_count {
            return None;
        }

        let mut proof = Vec::with_capacity(self.depth - 1);

        // Leaf nodes start at index: 2^(depth-1) - 1 = leaf_count - 1
        let leaf_start = (1 << (self.depth - 1)) - 1;
        let mut current_index = leaf_start + index;

        // Traverse up to root (but don't include root in proof)
        while current_index > 0 {
            let sibling_index = if current_index % 2 == 0 {
                // Even: sibling is to the left
                current_index - 1
            } else {
                // Odd: sibling is to the right
                current_index + 1
            };

            proof.push(self.nodes[sibling_index].clone());
            current_index /= 2;
        }

        Some(MerkleProof { proof, index })
    }

    pub fn verify_proof(&self, leaf: &Tower, proof: &MerkleProof) -> bool {
        let mut current_hash = MerkleNode::new(&leaf.value().to_le_bytes());
        let mut index = proof.index;

        for sibling in &proof.proof {
            current_hash = if index % 2 == 0 {
                let mut combined = [0u8; 64];
                combined[..32].copy_from_slice(&current_hash.0);
                combined[32..].copy_from_slice(&sibling.0);
                MerkleNode::new(&combined)
            } else {
                let mut combined = [0u8; 64];
                combined[..32].copy_from_slice(&sibling.0);
                combined[32..].copy_from_slice(&current_hash.0);
                MerkleNode::new(&combined)
            };
            index /= 2;
        }

        current_hash == *self.root()
    }
}

#[derive(Clone, Debug)]
pub struct MerkleProof {
    proof: Vec<MerkleNode>,
    index: usize,
}

impl MerkleProof {
    pub fn new(proof: Vec<MerkleNode>, index: usize) -> Self {
        Self { proof, index }
    }

    pub fn as_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.proof.len() * 32 + 4);
        bytes.extend_from_slice(&(self.index as u32).to_le_bytes());
        for node in &self.proof {
            bytes.extend_from_slice(&node.0);
        }
        bytes
    }

    pub fn len(&self) -> usize {
        self.proof.len()
    }

    pub fn is_empty(&self) -> bool {
        self.proof.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merkle_tree_build() {
        let leaves = vec![
            Tower::new(1, 3),
            Tower::new(2, 3),
            Tower::new(3, 3),
            Tower::new(4, 3),
        ];

        let tree = MerkleTree::build(&leaves).unwrap();
        assert!(!tree.root().as_bytes().is_empty());
        assert_eq!(tree.depth(), 3); // 4 leaves -> 3 levels
    }

    #[test]
    fn test_merkle_proof() {
        let leaves = vec![
            Tower::new(1, 3),
            Tower::new(2, 3),
            Tower::new(3, 3),
            Tower::new(4, 3),
        ];

        let tree = MerkleTree::build(&leaves).unwrap();

        // Debug: print tree structure
        println!("Leaf count: {}, Depth: {}", tree.leaf_count, tree.depth());
        println!("All nodes:");
        for (i, n) in tree.nodes.iter().enumerate() {
            println!("  nodes[{}] = {:02x?}", i, &n.0[..8]);
        }

        let proof = tree.get_proof(0).unwrap();
        println!("Proof len: {}", proof.len());

        // Verify manually
        let leaf = MerkleNode::new(&leaves[0].value().to_le_bytes());
        let mut current = leaf;
        let mut idx = 0;
        for (i, sibling) in proof.proof.iter().enumerate() {
            println!("Step {}: idx={}, sibling={:02x?}", i, idx, &sibling.0[..8]);
            current = if idx % 2 == 0 {
                let mut c = [0u8; 64];
                c[..32].copy_from_slice(&current.0);
                c[32..].copy_from_slice(&sibling.0);
                MerkleNode::new(&c)
            } else {
                let mut c = [0u8; 64];
                c[..32].copy_from_slice(&sibling.0);
                c[32..].copy_from_slice(&current.0);
                MerkleNode::new(&c)
            };
            println!("  -> {:02x?}", &current.0[..8]);
            idx /= 2;
        }
        println!("Computed: {:02x?}", &current.0[..8]);
        println!("Root:     {:02x?}", &tree.root().0[..8]);

        assert!(tree.verify_proof(&leaves[0], &proof));
    }

    #[test]
    fn test_merkle_proof_wrong_index() {
        let leaves = vec![
            Tower::new(1, 3),
            Tower::new(2, 3),
            Tower::new(3, 3),
            Tower::new(4, 3),
        ];

        let tree = MerkleTree::build(&leaves).unwrap();
        let proof = tree.get_proof(0).unwrap();

        // Verify with wrong leaf should fail
        assert!(!tree.verify_proof(&leaves[1], &proof));
    }
}
