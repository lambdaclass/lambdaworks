//! XMSS Merkle Tree implementation
//!
//! The XMSS tree is a binary Merkle tree where:
//! - Leaves are L-tree compressed WOTS+ public keys
//! - The root becomes the XMSS public key
//! - Authentication paths prove membership of leaves

use crate::address::{Address, AddressType};
use crate::hash::XmssHasher;
use crate::ltree::ltree;
use crate::params::{H, N};
use crate::wots::wots_pkgen;

/// Authentication path for XMSS signature verification
///
/// Contains H sibling nodes needed to recompute the root from a leaf.
#[derive(Clone, Debug)]
pub struct AuthPath {
    /// Sibling nodes from leaf to root (H elements)
    pub path: Vec<[u8; N]>,
}

impl AuthPath {
    /// Create a new authentication path
    pub fn new(path: Vec<[u8; N]>) -> Self {
        assert_eq!(path.len(), H, "Auth path must have {} elements", H);
        Self { path }
    }
}

/// Full XMSS tree containing all nodes
///
/// Stored in a flat array: nodes[height][index]
/// where height 0 contains the leaves.
#[derive(Clone)]
pub struct XmssTree {
    /// All tree nodes, indexed by [height][index]
    /// Height 0 = leaves, Height H = root
    nodes: Vec<Vec<[u8; N]>>,
}

impl XmssTree {
    /// Build a complete XMSS tree from seeds
    ///
    /// # Arguments
    /// * `hasher` - The hash function implementation
    /// * `secret_seed` - Secret seed for WOTS+ key generation
    /// * `public_seed` - Public seed
    ///
    /// # Returns
    /// The complete XMSS tree
    pub fn build<H: XmssHasher>(
        hasher: &H,
        secret_seed: &[u8; N],
        public_seed: &[u8; N],
    ) -> Self {
        let num_leaves = 1usize << crate::params::H;

        // Initialize tree structure
        let mut nodes: Vec<Vec<[u8; N]>> = Vec::with_capacity(crate::params::H + 1);

        // Generate leaves (level 0)
        let mut leaves = Vec::with_capacity(num_leaves);
        for i in 0..num_leaves {
            let leaf = compute_leaf(hasher, secret_seed, public_seed, i as u32);
            leaves.push(leaf);
        }
        nodes.push(leaves);

        // Build internal nodes bottom-up
        let mut address = Address::new();
        address.set_type(AddressType::HashTree);

        for height in 0..crate::params::H {
            let current_level = &nodes[height];
            let num_parents = current_level.len() / 2;
            let mut parent_level = Vec::with_capacity(num_parents);

            address.set_tree_height(height as u32);

            for i in 0..num_parents {
                address.set_tree_index(i as u32);
                let left = &current_level[2 * i];
                let right = &current_level[2 * i + 1];
                let parent = hasher.h(left, right, public_seed, &address);
                parent_level.push(parent);
            }

            nodes.push(parent_level);
        }

        Self { nodes }
    }

    /// Get the root of the tree (XMSS public key)
    pub fn root(&self) -> &[u8; N] {
        &self.nodes[crate::params::H][0]
    }

    /// Get the authentication path for a given leaf index
    ///
    /// # Arguments
    /// * `leaf_idx` - Index of the leaf (0 to 2^H - 1)
    ///
    /// # Returns
    /// The authentication path containing H sibling nodes
    pub fn auth_path(&self, leaf_idx: u32) -> AuthPath {
        let mut path = Vec::with_capacity(crate::params::H);
        let mut idx = leaf_idx as usize;

        for height in 0..crate::params::H {
            // Get sibling index
            let sibling_idx = if idx % 2 == 0 { idx + 1 } else { idx - 1 };

            path.push(self.nodes[height][sibling_idx]);

            // Move to parent index
            idx /= 2;
        }

        AuthPath::new(path)
    }

    /// Get a leaf node by index
    pub fn leaf(&self, idx: u32) -> &[u8; N] {
        &self.nodes[0][idx as usize]
    }
}

/// Compute a single leaf node
///
/// A leaf is the L-tree compression of a WOTS+ public key.
///
/// # Arguments
/// * `hasher` - The hash function implementation
/// * `secret_seed` - Secret seed for WOTS+ key generation
/// * `public_seed` - Public seed
/// * `idx` - Leaf index (which WOTS+ key pair)
///
/// # Returns
/// The leaf value (n bytes)
pub fn compute_leaf<H: XmssHasher>(
    hasher: &H,
    secret_seed: &[u8; N],
    public_seed: &[u8; N],
    idx: u32,
) -> [u8; N] {
    let mut address = Address::new();

    // Generate WOTS+ public key for this leaf
    address.set_type(AddressType::Ots);
    address.set_ots_address(idx);
    let wots_pk = wots_pkgen(hasher, secret_seed, public_seed, &mut address);

    // Compress with L-tree
    address.set_type(AddressType::LTree);
    address.set_ltree_address(idx);
    ltree(hasher, &wots_pk, public_seed, &mut address)
}

/// Compute root from a leaf and its authentication path
///
/// This is used during verification to recompute what the root
/// should be, given a claimed leaf value and auth path.
///
/// # Arguments
/// * `hasher` - The hash function implementation
/// * `leaf` - The leaf value
/// * `leaf_idx` - Index of the leaf
/// * `auth_path` - The authentication path
/// * `public_seed` - Public seed
///
/// # Returns
/// The computed root
pub fn compute_root<H: XmssHasher>(
    hasher: &H,
    leaf: &[u8; N],
    leaf_idx: u32,
    auth_path: &AuthPath,
    public_seed: &[u8; N],
) -> [u8; N] {
    let mut address = Address::new();
    address.set_type(AddressType::HashTree);

    let mut current = *leaf;
    let mut idx = leaf_idx;

    for height in 0..crate::params::H {
        address.set_tree_height(height as u32);
        address.set_tree_index(idx / 2);

        // Determine if we're left or right child
        current = if idx % 2 == 0 {
            // We're left child, sibling is right
            hasher.h(&current, &auth_path.path[height], public_seed, &address)
        } else {
            // We're right child, sibling is left
            hasher.h(&auth_path.path[height], &current, public_seed, &address)
        };

        idx /= 2;
    }

    current
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash::Sha256Hasher;

    #[test]
    fn test_tree_build() {
        let hasher = Sha256Hasher::new();
        let secret_seed = [0x01u8; N];
        let public_seed = [0x02u8; N];

        let tree = XmssTree::build(&hasher, &secret_seed, &public_seed);

        // Root should be a valid 32-byte value
        assert_eq!(tree.root().len(), N);
    }

    #[test]
    fn test_tree_deterministic() {
        let hasher = Sha256Hasher::new();
        let secret_seed = [0x01u8; N];
        let public_seed = [0x02u8; N];

        let tree1 = XmssTree::build(&hasher, &secret_seed, &public_seed);
        let tree2 = XmssTree::build(&hasher, &secret_seed, &public_seed);

        assert_eq!(tree1.root(), tree2.root());
    }

    #[test]
    fn test_auth_path_verification() {
        let hasher = Sha256Hasher::new();
        let secret_seed = [0x01u8; N];
        let public_seed = [0x02u8; N];

        let tree = XmssTree::build(&hasher, &secret_seed, &public_seed);

        // Test verification for leaf 0
        let leaf_idx = 0u32;
        let leaf = tree.leaf(leaf_idx);
        let auth_path = tree.auth_path(leaf_idx);

        let computed_root = compute_root(&hasher, leaf, leaf_idx, &auth_path, &public_seed);

        assert_eq!(computed_root, *tree.root());
    }

    #[test]
    fn test_auth_path_all_leaves() {
        let hasher = Sha256Hasher::new();
        let secret_seed = [0x01u8; N];
        let public_seed = [0x02u8; N];

        let tree = XmssTree::build(&hasher, &secret_seed, &public_seed);

        // Test a few leaves
        for leaf_idx in [0, 1, 10, 100, 500, 1023].iter() {
            let leaf = tree.leaf(*leaf_idx);
            let auth_path = tree.auth_path(*leaf_idx);

            let computed_root = compute_root(&hasher, leaf, *leaf_idx, &auth_path, &public_seed);

            assert_eq!(
                computed_root,
                *tree.root(),
                "Auth path failed for leaf {}",
                leaf_idx
            );
        }
    }

    #[test]
    fn test_different_leaves_same_root() {
        let hasher = Sha256Hasher::new();
        let secret_seed = [0x01u8; N];
        let public_seed = [0x02u8; N];

        let tree = XmssTree::build(&hasher, &secret_seed, &public_seed);

        // Different leaves should have different values but same root
        let leaf0 = tree.leaf(0);
        let leaf1 = tree.leaf(1);

        assert_ne!(leaf0, leaf1);

        // But both should verify to the same root
        let auth0 = tree.auth_path(0);
        let auth1 = tree.auth_path(1);

        let root0 = compute_root(&hasher, leaf0, 0, &auth0, &public_seed);
        let root1 = compute_root(&hasher, leaf1, 1, &auth1, &public_seed);

        assert_eq!(root0, root1);
        assert_eq!(root0, *tree.root());
    }

    #[test]
    fn test_wrong_leaf_idx_fails() {
        let hasher = Sha256Hasher::new();
        let secret_seed = [0x01u8; N];
        let public_seed = [0x02u8; N];

        let tree = XmssTree::build(&hasher, &secret_seed, &public_seed);

        let leaf = tree.leaf(0);
        let auth_path = tree.auth_path(0);

        // Using wrong leaf index should fail
        let computed_root = compute_root(&hasher, leaf, 1, &auth_path, &public_seed);

        assert_ne!(computed_root, *tree.root());
    }
}
