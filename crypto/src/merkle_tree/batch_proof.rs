use std::collections::HashMap;

use alloc::vec::Vec;
#[cfg(feature = "alloc")]
use lambdaworks_math::traits::Serializable;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use lambdaworks_math::{errors::DeserializationError, traits::Deserializable};

use crate::merkle_tree::utils::get_parent_pos;

use super::{merkle::NodePos, traits::IsMerkleTreeBackend, utils::get_sibling_pos};

/// Stores a merkle path to some leaf.
/// Internally, the necessary hashes are stored from root to leaf in the
/// `merkle_path` field, in such a way that, if the merkle tree is of height `n`, the
/// `i`-th element of `merkle_path` is the sibling node in the `n - 1 - i`-th check
/// when verifying.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BatchProof<T: PartialEq + Eq> {
    pub auth: HashMap<NodePos, T>,
}

impl<T: PartialEq + Eq> BatchProof<T> {
    pub fn verify<B>(&self, root_hash: B::Node, hashed_leaves: HashMap<NodePos, B::Node>) -> bool
    where
        B: IsMerkleTreeBackend<Node = T>,
    {
        let root_pos = 0;

        // Iterate the levels starting from the leaves, and build the upper level only using
        // the provided leaves and the auth map.
        let mut current_level = hashed_leaves;
        loop {
            let mut parent_level = HashMap::<NodePos, B::Node>::new();

            for (pos, node) in current_level.iter() {
                // Levels are expected to have tuples of nodes. If the first one was
                // already processed and parent was set, skip the sibling.
                let parent_pos = get_parent_pos(*pos);
                if parent_level.contains_key(&parent_pos) {
                    continue;
                }

                // Get the sibling node from the current level.
                // If doesn't exist, then it must have been provided in the batch auth.
                // If neither, then verification fails.
                let sibling_pos = get_sibling_pos(*pos);
                let sibling_node = if current_level.contains_key(&sibling_pos) {
                    current_level.get(&sibling_pos).unwrap()
                } else if self.auth.contains_key(&sibling_pos) {
                    self.auth.get(&sibling_pos).unwrap()
                } else {
                    // no sibling to hash with! Return error.
                    panic!();
                };

                let parent_node = B::hash_new_parent(node, sibling_node);

                // Root must match the provided root hash.
                if parent_pos == root_pos {
                    return parent_node == root_hash;
                }

                // Create a new element for the next, upper level
                parent_level.insert(parent_pos, parent_node);
            }

            // We didn't create any parents, and we didn't reach the root neither. Verification fails.
            if parent_level.is_empty() {
                return false;
            }

            // Issue the next level in the next iteration
            current_level = parent_level;
        }
    }
}

// #[cfg(feature = "alloc")]
// impl<T> Serializable for BatchProof<T>
// where
//     T: Serializable + PartialEq + Eq,
// {
//     fn serialize(&self) -> Vec<u8> {
//         self.merkle_path
//             .iter()
//             .flat_map(|node| node.serialize())
//             .collect()
//     }
// }

// impl<T> Deserializable for BatchProof<T>
// where
//     T: Deserializable + PartialEq + Eq,
// {
//     fn deserialize(bytes: &[u8]) -> Result<Self, DeserializationError>
//     where
//         Self: Sized,
//     {
//         let mut auth = Vec::new();
//         for elem in bytes[0..].chunks(8) {
//             let node = T::deserialize(elem)?;
//             auth.push(node);
//         }
//         Ok(Self { auth })
//     }
// }

#[cfg(test)]
mod tests {

    use std::collections::{HashMap, HashSet};

    #[cfg(feature = "alloc")]
    use super::BatchProof;
    use alloc::vec::Vec;
    use lambdaworks_math::field::{element::FieldElement, fields::u64_prime_field::U64PrimeField};
    #[cfg(feature = "alloc")]
    use lambdaworks_math::traits::{Deserializable, Serializable};

    use crate::merkle_tree::{
        merkle::{MerkleTree, NodePos},
        test_merkle::TestBackend as TB,
        traits::IsMerkleTreeBackend,
    };

    /// Small field useful for starks, sometimes called min i goldilocks
    /// Used in miden and winterfell
    // This field shouldn't be defined inside the merkle tree module
    pub type Ecgfp5 = U64PrimeField<0xFFFF_FFFF_0000_0001_u64>;
    pub type Ecgfp5FE = FieldElement<Ecgfp5>;
    pub type TestBackend = TB<Ecgfp5>;
    pub type TestMerkleTreeEcgfp = MerkleTree<TestBackend>;
    #[cfg(feature = "alloc")]
    pub type TestBatchProofEcgfp5 = BatchProof<Ecgfp5FE>;

    const MODULUS: u64 = 13;
    type U64PF = U64PrimeField<MODULUS>;
    type FE = FieldElement<U64PF>;

    // #[test]
    // #[cfg(feature = "alloc")]
    // fn serialize_batch_proof_and_deserialize_using_be_it_get_a_consistent_batch_proof() {
    //     let merkle_path = [Ecgfp5FE::new(2), Ecgfp5FE::new(1), Ecgfp5FE::new(1)].to_vec();
    //     let original_batch_proof = TestBatchProofEcgfp5 { merkle_path };
    //     let serialize_batch_proof = original_batch_proof.serialize();
    //     let batch_proof: TestBatchProofEcgfp5 =
    //         BatchProof::deserialize(&serialize_batch_proof).unwrap();

    //     for (o_node, node) in original_batch_proof
    //         .merkle_path
    //         .iter()
    //         .zip(batch_proof.merkle_path)
    //     {
    //         assert_eq!(*o_node, node);
    //     }
    // }

    // #[test]
    // #[cfg(feature = "alloc")]
    // fn serialize_batch_proof_and_deserialize_using_le_it_get_a_consistent_batch_proof() {
    //     let merkle_path = [Ecgfp5FE::new(2), Ecgfp5FE::new(1), Ecgfp5FE::new(1)].to_vec();
    //     let original_batch_proof = TestBatchProofEcgfp5 { merkle_path };
    //     let serialize_batch_proof = original_batch_proof.serialize();
    //     let batch_proof: TestBatchProofEcgfp5 =
    //         BatchProof::deserialize(&serialize_batch_proof).unwrap();

    //     for (o_node, node) in original_batch_proof
    //         .merkle_path
    //         .iter()
    //         .zip(batch_proof.merkle_path)
    //     {
    //         assert_eq!(*o_node, node);
    //     }
    // }

    // Creates following tree:
    //
    //          20
    //       /      \
    //      6       14
    //     / \      / \
    //    2   4    6   8
    //
    // Proves inclusion of leaves whose indices are passed into 'leaf_indices' array.
    // If it's [0, 3], then the test will create proof and verify inclusion of leaves with indices 0 and 3,
    // that are, 2 and 8.
    //
    // The test uses a test backend whose hash function is just an element added to itself.
    // So if leaf_values = [1,2,3,4], then actual leaf values will be [2,4,6,8], making the root 20.
    #[test]
    fn batch_proof_pen_and_paper_example() {
        let leaf_values: Vec<Ecgfp5FE> = (1..u64::pow(2, 2) + 1).map(Ecgfp5FE::new).collect();
        let merkle_tree: MerkleTree<TestBackend> = TestMerkleTreeEcgfp::build(&leaf_values);

        let nodes_len = merkle_tree.nodes_len();
        let first_leaf_pos = nodes_len / 2;

        // Leaves to prove inclusion for
        let leaf_indices = [0, 3];
        let leaf_positions: Vec<NodePos> = leaf_indices
            .iter()
            .map(|leaf_index| leaf_index + first_leaf_pos)
            .collect();

        // Build an authentication map for the first 10 leaves
        let batch_proof = merkle_tree.get_batch_proof(&leaf_positions).unwrap();

        let leaves: HashMap<NodePos, Ecgfp5FE> = leaf_positions
            .iter()
            .map(|pos| {
                (
                    *pos,
                    TestBackend::hash_data(&leaf_values[*pos - first_leaf_pos].clone()),
                )
            })
            .collect();

        assert!(batch_proof.verify::<TestBackend>(merkle_tree.root, leaves));
    }

    // #[test]
    // fn create_a_merkle_tree_with_10000_elements_and_verify_that_an_element_is_part_of_it() {
    //     let values: Vec<Ecgfp5FE> = (1..10000).map(Ecgfp5FE::new).collect();
    //     let merkle_tree = TestMerkleTreeEcgfp::build(&values);

    //     let proven_leaf_range = 4..usize::pow(2, 8);

    //     let batch_proof = merkle_tree
    //         .get_batch_proof(&proven_leaf_range.clone().collect::<Vec<_>>())
    //         .unwrap();

    //     let leaves = HashMap::from_iter(
    //         merkle_tree
    //             .get_batch_with_positions(
    //                 &proven_leaf_range
    //                     .map(|e| NodePos::from(e))
    //                     .collect::<Vec<_>>(),
    //             )
    //             .iter()
    //             .map(|(pos, elem)| (*pos, elem.clone())),
    //     );

    //     assert!(batch_proof.verify::<TestBackend<Ecgfp5>>(merkle_tree.root, leaves));
    // }
}
