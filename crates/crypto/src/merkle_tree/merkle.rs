use core::fmt::Display;

use alloc::vec::Vec;

use super::{multiproof::MultiProof, proof::Proof, traits::IsMerkleTreeBackend, utils::*};

#[derive(Debug)]
pub enum Error {
    OutOfBounds,
}
impl Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Accessed node was out of bound")
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Error {}

/// The struct for the Merkle tree, consisting of the root and the nodes.
/// A typical tree would look like this
///                 root
///              /        \
///          leaf 12     leaf 34
///        /         \    /      \
///    leaf 1     leaf 2 leaf 3  leaf 4
/// The bottom leafs correspond to the hashes of the elements, while each upper
/// layer contains the hash of the concatenation of the daughter nodes.
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MerkleTree<B: IsMerkleTreeBackend> {
    pub root: B::Node,
    nodes: Vec<B::Node>,
}

const ROOT: usize = 0;

impl<B> MerkleTree<B>
where
    B: IsMerkleTreeBackend,
{
    /// Create a Merkle tree from a slice of data
    pub fn build(unhashed_leaves: &[B::Data]) -> Option<Self> {
        if unhashed_leaves.is_empty() {
            return None;
        }

        let hashed_leaves: Vec<B::Node> = B::hash_leaves(unhashed_leaves);

        //The leaf must be a power of 2 set
        let hashed_leaves = complete_until_power_of_two(hashed_leaves);
        let leaves_len = hashed_leaves.len();

        //The length of leaves minus one inner node in the merkle tree
        //The first elements are overwritten by build function, it doesn't matter what it's there
        let mut nodes = vec![hashed_leaves[0].clone(); leaves_len - 1];
        nodes.extend(hashed_leaves);

        //Build the inner nodes of the tree
        build::<B>(&mut nodes, leaves_len);

        Some(MerkleTree {
            root: nodes[ROOT].clone(),
            nodes,
        })
    }

    /// Returns a Merkle proof for the element/s at position pos
    /// For example, give me an inclusion proof for the 3rd element in the
    /// Merkle tree
    pub fn get_proof_by_pos(&self, pos: usize) -> Option<Proof<B::Node>> {
        let pos = pos + self.nodes.len() / 2;
        let Ok(merkle_path) = self.build_merkle_path(pos) else {
            return None;
        };

        self.create_proof(merkle_path)
    }

    /// Returns a batched inclusion proof for the elements at the given positions,
    /// using the Octopus technique to remove redundancies between the individual
    /// authentication paths.
    ///
    /// The returned proof is never larger than the concatenation of the
    /// individual proofs for the same positions, and is usually smaller: sibling
    /// nodes shared between two proven paths, as well as nodes that the verifier
    /// can recompute from other values in the proof, are not included.
    ///
    /// Positions are sorted and deduplicated, so the caller does not need to
    /// pre-process them. Returns `None` if no positions are given or if any
    /// position is out of the bounds of the tree.
    pub fn get_multiproof_by_positions(&self, positions: &[usize]) -> Option<MultiProof<B::Node>> {
        if positions.is_empty() {
            return None;
        }

        // `nodes` holds `2 * leaves_len - 1` entries: the first `leaves_len - 1`
        // are internal nodes and the rest are the (padded) leaves.
        let leaves_len = self.nodes.len().div_ceil(2);
        debug_assert!(leaves_len.is_power_of_two());

        // Unpadded leaf positions, sorted and without repetitions.
        let mut sorted_positions = positions.to_vec();
        sorted_positions.sort_unstable();
        sorted_positions.dedup();

        if sorted_positions
            .iter()
            .any(|position| *position >= leaves_len)
        {
            return None;
        }

        // We work with canonical heap indices over the complete tree, matching
        // the layout of `self.nodes`: the root is index 0, the children of a node
        // `j` are `2j + 1` and `2j + 2`, and the leaves occupy the indices
        // `[leaves_len - 1, 2 * leaves_len - 1)`. `sibling_index` and
        // `parent_index` operate on this indexing.
        //
        // `current` holds the proven node indices at the current level, sorted in
        // increasing order (which is also left-to-right order within a level).
        let mut current: Vec<usize> = sorted_positions
            .iter()
            .map(|position| position + leaves_len - 1)
            .collect();

        let mut nodes = Vec::new();

        // Walk up level by level until the root (canonical index 0) is the only
        // remaining node. The parent of a pair of proven siblings is derived by
        // the verifier, so it remains in the proven set and no hash is emitted
        // for it. Any node whose sibling is *not* proven needs that sibling hash
        // to be transmitted.
        while current.first() != Some(&0) {
            let mut next = Vec::new();

            let mut i = 0;
            while i < current.len() {
                if i + 1 < current.len() && current[i + 1] == sibling_index(current[i]) {
                    // Both siblings are proven: no hash needed.
                    i += 2;
                } else {
                    // The sibling is not proven, so it must be provided.
                    nodes.push(self.nodes[sibling_index(current[i])].clone());
                    i += 1;
                }

                let parent = parent_index(current[i - 1]);
                // `next` stays sorted and free of repetitions: the proven
                // indices are sorted, and the only way two consecutive proven
                // nodes share a parent is the sibling pair handled above.
                if next.last() != Some(&parent) {
                    next.push(parent);
                }
            }

            current = next;
        }

        Some(MultiProof {
            leaves_len,
            positions: sorted_positions,
            nodes,
        })
    }

    /// Creates a proof from a Merkle pasth
    fn create_proof(&self, merkle_path: Vec<B::Node>) -> Option<Proof<B::Node>> {
        Some(Proof { merkle_path })
    }

    /// Returns the Merkle path for the element/s for the leaf at position pos
    fn build_merkle_path(&self, mut pos: usize) -> Result<Vec<B::Node>, Error> {
        let mut merkle_path = Vec::new();

        while pos != ROOT {
            let Some(node) = self.nodes.get(sibling_index(pos)) else {
                // out of bounds, exit returning the current merkle_path
                return Err(Error::OutOfBounds);
            };
            merkle_path.push(node.clone());

            pos = parent_index(pos);
        }

        Ok(merkle_path)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use lambdaworks_math::field::{element::FieldElement, fields::u64_prime_field::U64PrimeField};

    use crate::merkle_tree::{merkle::MerkleTree, test_merkle::TestBackend};

    const MODULUS: u64 = 13;
    type U64PF = U64PrimeField<MODULUS>;
    type FE = FieldElement<U64PF>;

    #[test]
    fn build_merkle_tree_from_a_power_of_two_list_of_values() {
        let values: Vec<FE> = (1..5).map(FE::new).collect();
        let merkle_tree = MerkleTree::<TestBackend<U64PF>>::build(&values).unwrap();
        assert_eq!(merkle_tree.root, FE::new(7)); // Adjusted expected value
    }

    #[test]
    // expected | 8 | 7 | 1 | 6 | 1 | 7 | 7 | 2 | 4 | 6 | 8 | 10 | 10 | 10 | 10 |
    fn build_merkle_tree_from_an_odd_set_of_leaves() {
        const MODULUS: u64 = 13;
        type U64PF = U64PrimeField<MODULUS>;
        type FE = FieldElement<U64PF>;

        let values: Vec<FE> = (1..6).map(FE::new).collect();
        let merkle_tree = MerkleTree::<TestBackend<U64PF>>::build(&values).unwrap();
        assert_eq!(merkle_tree.root, FE::new(8)); // Adjusted expected value
    }

    #[test]
    fn build_merkle_tree_from_a_single_value() {
        const MODULUS: u64 = 13;
        type U64PF = U64PrimeField<MODULUS>;
        type FE = FieldElement<U64PF>;

        let values: Vec<FE> = vec![FE::new(1)]; // Single element
        let merkle_tree = MerkleTree::<TestBackend<U64PF>>::build(&values).unwrap();
        assert_eq!(merkle_tree.root, FE::new(2)); // Adjusted expected value
    }

    #[test]
    fn build_empty_tree_should_not_panic() {
        assert!(MerkleTree::<TestBackend<U64PF>>::build(&[]).is_none());
    }
}
