//! L-Tree implementation for XMSS
//!
//! The L-tree (Leaf tree) compresses a WOTS+ public key (LEN elements)
//! into a single n-byte value that becomes a leaf in the XMSS tree.
//!
//! Since LEN (67) is not a power of 2, the L-tree handles odd numbers
//! of nodes by promoting the leftover node to the next level.

use crate::address::{Address, AddressType};
use crate::hash::XmssHasher;
use crate::params::N;
use crate::wots::WotsPublicKey;

/// Compress a WOTS+ public key into a single leaf value using L-tree
///
/// The L-tree is a binary tree that hashes pairs of nodes together.
/// When there's an odd number of nodes at a level, the last node
/// is promoted to the next level unchanged.
///
/// # Arguments
/// * `hasher` - The hash function implementation
/// * `wots_pk` - The WOTS+ public key to compress
/// * `public_seed` - Public seed SEED
/// * `address` - L-tree address (will be modified)
///
/// # Returns
/// The compressed leaf value (n bytes)
pub fn ltree<H: XmssHasher>(
    hasher: &H,
    wots_pk: &WotsPublicKey,
    public_seed: &[u8; N],
    address: &mut Address,
) -> [u8; N] {
    address.set_type(AddressType::LTree);

    // Copy the public key elements as our working set
    let mut nodes: Vec<[u8; N]> = wots_pk.pk.clone();

    let mut height: u32 = 0;

    // Keep hashing until we have a single node
    while nodes.len() > 1 {
        address.set_tree_height(height);

        let mut parent_nodes = Vec::new();
        let mut i = 0;

        // Hash pairs of nodes
        while i + 1 < nodes.len() {
            address.set_tree_index(i as u32 / 2);
            let parent = hasher.h(&nodes[i], &nodes[i + 1], public_seed, address);
            parent_nodes.push(parent);
            i += 2;
        }

        // If odd number of nodes, promote the last one
        if i < nodes.len() {
            parent_nodes.push(nodes[i]);
        }

        nodes = parent_nodes;
        height += 1;
    }

    nodes[0]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash::Sha256Hasher;
    use crate::params::LEN;

    #[test]
    fn test_ltree_deterministic() {
        let hasher = Sha256Hasher::new();
        let public_seed = [0x42u8; N];

        // Create a dummy WOTS+ public key
        let mut pk_elements = Vec::with_capacity(LEN);
        for i in 0..LEN {
            let mut elem = [0u8; N];
            elem[0] = i as u8;
            pk_elements.push(elem);
        }
        let wots_pk = WotsPublicKey::new(pk_elements);

        let mut address1 = Address::new();
        address1.set_ltree_address(0);
        let result1 = ltree(&hasher, &wots_pk, &public_seed, &mut address1);

        let mut address2 = Address::new();
        address2.set_ltree_address(0);
        let result2 = ltree(&hasher, &wots_pk, &public_seed, &mut address2);

        assert_eq!(result1, result2);
    }

    #[test]
    fn test_ltree_different_inputs() {
        let hasher = Sha256Hasher::new();
        let public_seed = [0x42u8; N];

        // Create two different WOTS+ public keys
        let mut pk_elements1 = Vec::with_capacity(LEN);
        let mut pk_elements2 = Vec::with_capacity(LEN);
        for i in 0..LEN {
            let mut elem1 = [0u8; N];
            let mut elem2 = [0u8; N];
            elem1[0] = i as u8;
            elem2[0] = (i + 1) as u8;
            pk_elements1.push(elem1);
            pk_elements2.push(elem2);
        }
        let wots_pk1 = WotsPublicKey::new(pk_elements1);
        let wots_pk2 = WotsPublicKey::new(pk_elements2);

        let mut address = Address::new();
        address.set_ltree_address(0);

        let result1 = ltree(&hasher, &wots_pk1, &public_seed, &mut address);

        address.set_ltree_address(0);
        let result2 = ltree(&hasher, &wots_pk2, &public_seed, &mut address);

        assert_ne!(result1, result2);
    }

    #[test]
    fn test_ltree_output_size() {
        let hasher = Sha256Hasher::new();
        let public_seed = [0x42u8; N];

        let mut pk_elements = Vec::with_capacity(LEN);
        for i in 0..LEN {
            let mut elem = [0u8; N];
            elem[0] = i as u8;
            pk_elements.push(elem);
        }
        let wots_pk = WotsPublicKey::new(pk_elements);

        let mut address = Address::new();
        address.set_ltree_address(0);

        let result = ltree(&hasher, &wots_pk, &public_seed, &mut address);

        assert_eq!(result.len(), N);
    }

    #[test]
    fn test_ltree_different_addresses() {
        let hasher = Sha256Hasher::new();
        let public_seed = [0x42u8; N];

        let mut pk_elements = Vec::with_capacity(LEN);
        for i in 0..LEN {
            let mut elem = [0u8; N];
            elem[0] = i as u8;
            pk_elements.push(elem);
        }
        let wots_pk = WotsPublicKey::new(pk_elements);

        let mut address1 = Address::new();
        address1.set_ltree_address(0);
        let result1 = ltree(&hasher, &wots_pk, &public_seed, &mut address1);

        let mut address2 = Address::new();
        address2.set_ltree_address(1);
        let result2 = ltree(&hasher, &wots_pk, &public_seed, &mut address2);

        // Different L-tree addresses should give different results
        assert_ne!(result1, result2);
    }
}
